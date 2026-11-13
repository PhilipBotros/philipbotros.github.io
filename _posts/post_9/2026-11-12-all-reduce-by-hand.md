---
layout: post
title: "Rebuilding NCCL's all-reduce: a CUDA worklog from T4s to B200s"
date: 2026-11-12
published: false
cover: /images/post_9/cover.png
background: /images/post_9/cover.png
og_image: /images/post_9/cover.png
mathjax: true
excerpt: <br> Rebuilding the techniques NCCL uses for all-reduce in CUDA, from transports and channels to protocols and algorithm choice, and seeing how A100s, H100s and B200s change which ones matter.
---
<link rel="stylesheet" href="/assets/post_9/figures.css">
<script src="/assets/post_9/figures.js" defer></script>

After working on [matrix multiplication](/2026/08/11/matmul-worklog.html) and reading the [JAX Scaling Book](https://jax-ml.github.io/scaling-book/), I wanted to understand GPU communication at a deeper level. I was curious what the different ways of moving data between GPUs are, and how the topology of a machine affects performance. Implementing an all-reduce by hand seemed like a good way to find out. I started with two T4s over PCIe and eventually worked up to eight A100s, then ran the same kernels on H100s and B200s to see how the optimisations carried over.

In distributed training, we use collectives to exchange data and combine results across GPUs. Frameworks like PyTorch usually hand this work to [NCCL](https://developer.nvidia.com/nccl), NVIDIA's Collective Communications Library. All-reduce is a good one to implement. Every GPU starts with its own vector, and all of them need to finish with the elementwise sum. One use is combining gradients in data-parallel training, where each GPU processes a different slice of the batch. There, the vector is usually a bucket of gradient tensors flattened into one buffer.

As with the matmul post, this is a worklog of what I tried and the bottlenecks I ran into. We'll start with a copy and an add kernel, then gradually move more of the work onto the GPUs, comparing each version with NCCL. Everything here happens inside one machine.

*Note: I used Claude Code heavily for this project, for writing and debugging the kernels, running the benchmark sweeps and editing this post.*

<h2> 1. Getting data from one GPU to another </h2>

Before choosing an all-reduce algorithm, let's look at what determines how quickly two GPUs inside one machine can exchange data:

1. **The bandwidth of the physical path.** Two GPUs can exchange data over PCIe, either through a shared PCIe switch or through the CPU's PCIe controller. Modern training GPUs like the A100, H100 and B200 have NVLink, a direct GPU-to-GPU connection with roughly ten times the bandwidth.

2. **Whether the GPUs can access each other's memory directly.** Without peer access, the data has to be copied into host memory and then on to the destination GPU, adding an extra transfer. Within one process, `cudaDeviceEnablePeerAccess` enables peer access. Across processes, CUDA IPC shares an allocation. Two of the cloud VMs we'll use don't allow peer access at all, so this extra trip is unavoidable there.

3. **How the data is moved.** With peer access, a copy engine, dedicated hardware that copies memory without running a kernel, can move the data into local memory before a kernel uses it. Alternatively, the kernel's own threads can load from the other GPU and compute on the values as they arrive. Without peer access, we use the copy engine to stage the data through host memory.

4. **The cost of synchronisation.** GPU 0 can't read GPU 1's vector until GPU 1 has finished writing it, and GPU 1 can't overwrite it while GPU 0 is still reading. Either the host enforces this order with streams and events, or the kernels signal each other directly by setting flags in each other's memory. For small messages, this coordination takes longer than the transfer itself.

<figure class="ar-fig" data-fig="topology">
  <figcaption>The paths between GPUs are PCIe through the host (grey), NVLink through NVSwitch (green), and a network card on each side between machines (orange).</figcaption>
</figure>

<h3> 1.1 Copy engine or SMs </h3>

For an all-reduce, we also need to add the values we receive to our local values. Here's what each approach involves for one contribution from GPU 1 to GPU 0:

<figure class="ar-fig" data-fig="movers">
  <div class="ar-head"><div class="ar-title">One transfer from GPU 1 to GPU 0</div></div>
  <figcaption>Copy, then add, or read the peer's memory inside the add kernel. Press next to step through both.</figcaption>
</figure>

Both versions use the same add kernel. Each thread reads one value from each input and writes their sum. The second pointer can refer to a local copy or to the other GPU's memory:

```cuda
__global__ void add(const float* a, const float* b, float* out, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) out[i] = a[i] + b[i];
}
```

In the examples below, GPU 0 is the current device and `stream` belongs to it. `local`, `tmp` and `out` are on GPU 0, and `peer` points to GPU 1's input. Both inputs are ready, and GPU 0 already has peer access to GPU 1.

With the **copy engine**, GPU 1's data has to land in GPU 0's memory before the add kernel can read it. We copy it into `tmp`, then add on the same stream:

```cuda
cudaMemcpyPeerAsync(tmp, 0, peer, 1, N * sizeof(float), stream);
add<<<(N + 255) / 256, 256, 0, stream>>>(local, tmp, out, N);
// Both inputs to add are now in GPU 0's memory.
```

The copy leaves the SMs available for other work, but we pay for the temporary buffer and another pass over memory in the add kernel.

With **SM loads**, we pass the peer's pointer straight to the kernel:

```cuda
add<<<(N + 255) / 256, 256, 0, stream>>>(local, peer, out, N);
// Reading b[i] inside add fetches the value from GPU 1.
```

The kernel reads the remote value and adds it directly in registers, avoiding `tmp` and the separate copy submission. It uses SM resources for the transfer, though, and needs enough independent memory requests to cover the time spent waiting for remote data.

It's easy to assume the second version is faster since it does less work, but we still need to get the data across the connection. A connection with bandwidth $B$ and round-trip latency $L$ needs roughly $B \times L$ bytes in flight to stay busy. If our threads issue a load and then wait for it before issuing another, we may leave most of that bandwidth unused. We'll come back to this on the A100s.

<h2> 2. What all-reduce does </h2>

Suppose we're training with data parallelism, and each GPU has computed gradients for its own batch of examples. We want to add the corresponding values together and give every GPU a copy of the result. Each GPU can then divide by the number of GPUs to get the average gradient, so they all make the same model update.

For a small example, suppose GPU 0 holds `[1, 2]` and GPU 1 holds `[3, 4]`. Adding the matching entries gives `[4, 6]`, and both GPUs need to finish with that vector.

To compare ways of doing this, we'll use $P$ for the number of GPUs and $N$ for the number of values in each GPU's vector. We're using FP32 values, which take 4 bytes each, so the input size per GPU is $S = 4N$ bytes. In the code, `N` always counts elements rather than bytes.

<figure class="ar-fig" data-fig="allreduce">
  <div class="ar-head"><div class="ar-title">All-reduce on four GPUs</div></div>
</figure>

The additions are cheap compared with moving the operands. We mainly need to reduce remote traffic and the time spent coordinating GPUs.

<h3> 2.1 Two algorithms </h3>

How should we organise the transfers? We want to move as few bytes as possible, but we also want to avoid making the GPUs wait for each other. For a large vector, moving the data takes most of the time. For a small one, launching the work and waiting for the other GPUs can cost more than the transfer itself.

Let's compare two algorithms with a simple model. Give each sequential communication step a fixed cost $\alpha$ and each GPU an effective receive bandwidth $B$.

The simplest way to extend our two-input add to more GPUs is to read an input from every GPU and sum them locally. This is the **direct** algorithm (often called *one-shot* all-reduce). Every GPU computes the full result for itself. There are no intermediate reduction rounds, but each GPU reads $(P-1) \cdot S$ remote bytes, which grows with every GPU you add:

$$T_\text{direct} \approx \alpha + (P-1)\cdot\frac{S}{B}.$$

We count this as one communication step because there are no dependent rounds.

<figure class="ar-fig" data-fig="direct">
  <div class="ar-head"><div class="ar-title">Direct read on four GPUs</div></div>
  <figcaption>Hover or tap a GPU to see its reads.</figcaption>
</figure>

The direct version is simple, but reading every other GPU's entire input gets expensive as we add GPUs. The **ring** avoids this by splitting the vector into $P$ chunks and passing partial sums between neighbours. NCCL uses it for large messages on many systems, including the A100 node here. Newer NVSwitch systems can also reduce inside the switch, which we'll look at later.

The ring builds the all-reduce out of two smaller collectives, a reduce-scatter followed by an all-gather. In **reduce-scatter**, each GPU receives one chunk from its left neighbour per step and adds it into its own copy. After $P-1$ steps, every GPU owns one fully summed chunk. The **all-gather** phase passes those finished chunks around for another $P-1$ steps so every GPU gets the full result. In total, each GPU receives

$$2 \cdot \frac{P-1}{P} \cdot S \text{ bytes},$$

and supplies the same amount to its neighbour, so the per-GPU volume approaches $2S$ as GPU count grows, at the price of $2(P-1)$ sequential steps:

$$T_\text{ring} \approx 2(P-1)\left(\alpha + \frac{S/P}{B}\right).$$

<figure class="ar-fig" data-fig="ring">
  <figcaption>Ring all-reduce on four GPUs.</figcaption>
</figure>

The direct algorithm pays one fixed cost and transfers more data. The ring transfers less data but pays a fixed cost at every step. We can write both using the same two terms:

$$T \approx n\,\alpha + \frac{V}{B},$$

with $n$ sequential steps and $V$ bytes received per GPU:

| | steps $n$ | bytes $V$ |
|---|---|---|
| direct | $1$ | $(P-1)\,S$ |
| ring | $2(P-1)$ | $2\frac{P-1}{P}\,S$ |

The optimisations below either make the steps fewer or cheaper, move fewer bytes, or use more bandwidth. Notice that with two GPUs both algorithms receive $S$ bytes, so the ring's extra step doesn't save us any traffic.

With more GPUs, we'd expect direct to win for small messages, where the fixed cost dominates, and the ring for large ones, where the transfer time does. The graph below plots both formulas. The marker shows the [default gradient bucket size](https://docs.pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html) of PyTorch's DistributedDataParallel (DDP), 25 MiB (`bucket_cap_mb=25`).

<figure class="ar-fig" data-fig="model">
  <div class="ar-head"><div class="ar-title">Cost model: ring against direct read</div></div>
  <figcaption>Change the GPU count and link to see the crossover move.</figcaption>
</figure>

<div class="ar-note" markdown="1">

**Note: PCIe and NVLink.** PCIe is the general-purpose interconnect used by GPUs, network cards and other devices. GPUs exchange data over it through PCIe switches or the CPU's PCIe controller. A PCIe 4.0 ×16 connection provides roughly 32 GB/s in each direction.

NVLink adds connections designed for GPU-to-GPU communication, either directly or through NVSwitch. An A100 has up to 300 GB/s in each direction across its NVLink connections, roughly ten times its PCIe bandwidth. The advertised 600 GB/s counts both directions together ([NVIDIA's architecture overview](https://developer.nvidia.com/blog/nvidia-ampere-architecture-in-depth/)).

For scale, Llama 3 8B's feed-forward down-projection matrix has shape 4,096 × 14,336, which is about 59 million parameters, or 117 MB of BF16 gradients. For a local batch of 2,048 tokens, its forward and backward matrix multiplications take about 722 GFLOPs, or 2.3 ms at the A100's peak BF16 throughput. A ring all-reduce over eight GPUs sends 1.75 times the gradient size from each GPU, about 205 MB. Assuming BF16 gradient communication and full interconnect bandwidth, that gives idealised transfer times of roughly 6.4 ms over PCIe and 0.7 ms over NVLink. The PCIe estimate is nearly three times the matrix's compute time at peak throughput. NVLink makes communication much easier to overlap with the remaining backward pass.

</div>

<h2> 3. Starting on PCIe </h2>

I started with two GPUs over PCIe before moving to NVLink.

Every time in this post is the median of 100 runs, or 20 for messages of 64 MB and up. The clock starts at the first call and stops once every GPU has finished.

<h3> 3.1 Two T4s </h3>

The first VM had two T4s on PCIe 3.0 ×16 through a host bridge, with peer access available. The link is rated at roughly 16 GB/s per direction. The simplest all-reduce copies each input to the other GPU and then runs the add kernel on both sides. With serialised copies this takes 33 µs at 4 KB.

<figure class="ar-fig" data-fig="machine" data-machine="t4">
  <div class="ar-head"><div class="ar-title">Machine: two T4s</div></div>
</figure>

The two copies don't depend on each other, so we can give each GPU its own stream and make the copies asynchronous. PCIe is full duplex, meaning data can travel in both directions at once. If the rest of the path supports that too, overlapping the copies should get us closer to the time of a single transfer. At 4 KB, this brings the total down to 19 µs.

Overlapping the copies helps at 4 KB, but does it also let us move large inputs in both directions at full speed? To separate the transfer from the addition, I timed the copies on their own. A single 256 MB copy takes 27 ms. Two copies in opposite directions start within 10 µs of each other, but both take 54 ms, roughly **9.9 GB/s combined**. The two transfers appear to share a limit somewhere on this VM's path. One candidate is [ACS redirecting peer traffic up through the host bridge](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/troubleshooting/gpu_troubleshooting.html#pci-access-control-services-acs), but I couldn't check that on this VM.

What about passing the peer pointer straight into the add kernel? That removes the separate copy submission and the temporary buffer. It cuts 4 KB latency to **14 µs**, but for large messages it's actually slower. One GPU reading gets 7.05 GB/s, and both reading get about 3.6 GB/s each. So there's another shared limit, about **7.2 GB/s for SM reads**. I tried grid-stride loops and `float4` loads, but neither made much difference.

NCCL chose host staging despite CUDA peer access. Sending data directly between the GPUs should avoid the intermediate host buffers, so I tried forcing P2P with `NCCL_P2P_LEVEL=SYS`. It saves 3–9 µs at small sizes and nothing at bulk, so changing the transport doesn't remove the large-message limit on this VM.

**4 KB: 14 µs with SM reads, 256 MB: 4.7 GB/s with async copies** (NCCL: 28 µs, 4.0 GB/s)

<figure class="ar-fig" data-fig="chart" data-chart="t4" data-mode="rel">
  <div class="ar-head"><div class="ar-title">Two T4s over PCIe 3.0</div></div>
  <figcaption>Hover for GB/s and times.</figcaption>
</figure>

<h3> 3.2 Four L4s without peer access </h3>

With two GPUs, each one only needs the other's input. On four GPUs, the ring starts to save traffic. Each GPU receives $1.5S$ against direct's $3S$. The price is six dependent communication steps.

The next VM was a **g6.12xlarge** with four L4s on PCIe 4.0. That gives us a higher theoretical link bandwidth than the T4's PCIe 3.0, but there was a catch. It didn't allow CUDA peer access between any pair. Both NCCL and my copy ring therefore stage through host memory, though through different paths. NCCL uses its shared-memory transport and CPU proxies, while the ring submits staged copy-engine transfers.

The ring still follows the reduce-scatter and all-gather schedule from section 2. For each reduce-scatter step, we copy a neighbour's partial sum into a local buffer and launch an add kernel. During all-gather, we only copy the finished chunks. We use streams and events so a step only starts once the previous one has finished. All of it is submitted from one host thread.

<figure class="ar-fig" data-fig="machine" data-machine="l4">
  <div class="ar-head"><div class="ar-title">Machine: four L4s</div></div>
</figure>

<div class="ar-note" markdown="1">

**Note: algorithm and bus bandwidth.** For the multi-GPU comparisons, we'll report **algorithm bandwidth**, the message size per GPU divided by the time. NCCL's tools also print [*bus bandwidth*](https://github.com/NVIDIA/nccl-tests/blob/master/doc/PERFORMANCE.md), which multiplies that by $2(P-1)/P$ to count the number of bytes a ring moves per GPU, so it can be set against link speed. That factor is 1.5 on four GPUs and 1.75 on eight. Apply it when comparing these numbers with bus bandwidth reported elsewhere. Message sizes are powers of two, so 256 MB is $2^{28}$ bytes, and GB/s means $10^9$ bytes per second.

</div>

At 256 MB the copy ring's bandwidth is **1.7x NCCL's**. NVIDIA's `nccl-tests` gets 3.8 GB/s against 3.7 in my benchmark, so my timing code doesn't explain NCCL's result. A g5.12xlarge with four A10Gs, also without peer access, gave the same picture at 2x.

At 4 KB, though, NCCL takes **42 µs** against the ring's **239 µs**. Each GPU needs six staged copies and three add kernels, plus event dependencies. One host thread submits all of it, 24 staged copies and 12 add kernels across the four GPUs. There's hardly any data to move, but the host still has to issue every step. NCCL runs the GPU work in one kernel per GPU. It still leads at 1 MB, and the copy ring pulls ahead at 4 MB.

**4 KB: 239 µs, 256 MB: 6.4 GB/s** (NCCL: 42 µs, 3.7 GB/s)

<figure class="ar-fig" data-fig="chart" data-chart="l4" data-mode="rel">
  <div class="ar-head"><div class="ar-title">Four L4s over PCIe 4.0, no peer access</div></div>
  <figcaption>“Copy all inputs, then sum” is the direct algorithm with staged copies.</figcaption>
</figure>

The host is spending most of the small-message time submitting work. We'd like to move that work into one kernel, but that requires peer access. Before moving on, I tried one more PCIe VM that has peer access.

<h3> 3.3 Four L4s with peer access </h3>

Would peer access have changed the result? I ran the same code on a second VM with four L4s, where every pair allows peer access.

Pairs of GPUs behave much like the T4s, but once a third GPU takes part the peer path collapses:

| 4x L4 with peer access, 256 MB | GB/s |
|---|---|
| copy ring over P2P | 0.55 |
| ring with SM reads | 0.60 |
| direct read | 0.30 |
| NCCL forced onto P2P | 0.28 |
| NCCL, staging through host memory | 3.2 |
| copy ring, staging through host memory | 3.2 |

Any three of the four GPUs show the same slowdown, so it isn't one slow GPU or link. NCCL avoids it by staging through shared host memory, and my copy ring matches it with peer access turned off. I have no idea why the peer path slows down.

Small messages don't have that problem. At 4 KB, direct read takes **23 µs**, against NCCL's 54 µs and the copy ring's 208 µs. Each GPU reads the other inputs inside one kernel, so the host launches one kernel per GPU instead of submitting every staged copy.

<h2> 4. Moving to NVLink: eight A100s </h2>

So far every transfer has gone over PCIe. The **p4d.24xlarge** has eight A100s that are also connected by NVLink. Each GPU's links go to NVSwitch, which connects every GPU to every other. Each GPU has [300 GB/s per direction](https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/a100/pdf/nvidia-a100-datasheet.pdf) shared across peers. Every GPU can access every other's memory, and NCCL uses its P2P transport.

<figure class="ar-fig" data-fig="machine" data-machine="a100">
  <div class="ar-head"><div class="ar-title">Machine: eight A100s</div></div>
</figure>

The faster connection should help large messages. At small sizes, we still have the CPU calls and the ring's dependent steps. There are fourteen of them now that we're using eight GPUs.

Now that every GPU can read its neighbours' memory, we can do the transfers and additions inside our kernels. We'll keep the ring algorithm and first try to get its bandwidth close to NCCL's, then work on the time spent coordinating each step. These experiments use one host process until section 4.6.

<h3> 4.1 Host-driven ring </h3>

I ran the copy ring unchanged on two, four and eight A100s.

Its 4 KB latency rises from 58 µs on two GPUs to 219 µs on four and **1149 µs on eight**, staying roughly flat through 16 MB. One host thread submits $P \cdot 2(P-1) = 112$ copies on eight GPUs. At about 10 µs per submission, that accounts for nearly the whole 1.1 ms. NCCL takes about 104 µs.

**Copy ring, 4 KB: 1149 µs, 256 MB: 90 GB/s** (NCCL: 104 µs, 116 GB/s)

At 256 MB, that's about **14x the 6.4 GB/s** the copy ring managed on four L4s with host staging over PCIe, even with twice as many GPUs participating. This compares different machines and transfer paths, rather than isolating NVLink's contribution, but it shows how much the PCIe setup was holding us back. Small-message latency barely changes at the same GPU count: the four-A100 copy ring takes **219 µs at 4 KB**, against **239 µs on four L4s**. The faster link helps bulk transfers; it doesn't remove the host's submission work.

On the T4, peer reads saved submission work but lost bandwidth. NVLink gives the SMs a different path to peer memory, so the T4 result needn't carry over. Between two A100s they reach 233 GB/s, against 164 GB/s for a copy-engine pull. An eight-GPU ring that reads the neighbour inside its add kernel now beats the copy ring at bulk, but the host still submits every step.

**Remote-read ring, 4 KB: 821 µs, 256 MB: 118 GB/s** (NCCL: 104 µs, 116 GB/s)

<figure class="ar-fig" data-fig="chart" data-chart="a100h" data-mode="lat">
  <div class="ar-head"><div class="ar-title">Eight A100s over NVLink, host-driven</div></div>
</figure>

The chart also includes both direct baselines from section 2, copying all inputs before summing and reading them remotely in one kernel. Remote direct read already takes just 45 µs at 4 KB, ahead of NCCL. We'll return to it in section 4.7 after tuning the ring. The host-driven ring is close to NCCL at bulk, but still about eight times slower at 4 KB.

<h3> 4.2 NCCL uses a fraction of the SMs </h3>

Our simple add kernel is slightly faster than NCCL at bulk, which I didn't expect, so I did a bit of digging into NCCL. NCCL handles transfers, additions and synchronisation in one kernel per GPU and splits the work into *channels*. A channel is one copy of the ring working on its own slice of the vector, with its own buffers and flags, and each channel runs in one thread block. With `NCCL_DEBUG=INFO`, NCCL reports **24 channels** on this machine, so it occupies **at most 24 of the A100's 108 SMs**. Our add kernels launch one thread per element, which for these sizes means thousands of blocks spread across all 108 SMs.

NCCL's small footprint is deliberate. In training, the all-reduce runs at the same time as the backward pass, and every SM it occupies is one the compute kernels can't use. Matching NCCL's bandwidth with the whole GPU isn't a fair comparison, so on the A100 we'll hold our kernels to 24 blocks of 512 threads, NCCL's default block size. Can we sustain the same bandwidth that way?

<h3> 4.3 More loads in flight with float4 </h3>

At 24 x 512, the scalar grid-stride kernel drops from roughly **120 to 42 GB/s**. In this version, each thread loads one FP32 value from each input, adds and stores, then advances to its next element. Reducing the thread count also reduces the number of independent loads we can have in flight. To relate that to the $B \times L$ estimate from section 1.1, we can approximate the outstanding payload as

$$\text{bytes in flight} \approx \text{resident threads} \times \text{outstanding loads per thread} \times \text{bytes per load}.$$

If all 24 x 512 threads have one remote 4-byte load outstanding, that's 48 KB in flight. The 42 GB/s is algorithm bandwidth, and the eight-GPU ring puts 1.75 times that on the link, about 74 GB/s. Dividing the bytes in flight by that rate gives an effective latency of about 0.67 µs. At the same latency, four independent 16-byte loads per thread would put 768 KB in flight, enough for about 1,180 GB/s, well above the link's 300 GB/s per direction. By this rough estimate, the scalar kernel is limited by how few loads it keeps in flight, and wider, independent loads should recover the bandwidth.

A `float4` groups four FP32 values into a 16-byte load. Instead of loading one value, adding and storing it before moving on, each thread issues four independent `float4` loads from each input first. This is also how NCCL's kernels are written:

```cuda
// a4 is local, b4 points into the neighbour's memory. Each holds n4 float4s.
int stride = gridDim.x * blockDim.x;                 // total number of threads
int first = blockIdx.x * blockDim.x + threadIdx.x;   // this thread's first element

// Each thread takes every stride-th element, four of them per iteration.
// The loop runs while all four are still inside the array.
for (int i = first; i + 3 * stride < n4; i += 4 * stride) {
  int i0 = i, i1 = i + stride, i2 = i + 2 * stride, i3 = i + 3 * stride;

  // Issue all eight loads before the first store.
  float4 a0 = a4[i0], a1 = a4[i1], a2 = a4[i2], a3 = a4[i3];
  float4 b0 = b4[i0], b1 = b4[i1], b2 = b4[i2], b3 = b4[i3];

  // f4add adds two float4s lane-wise.
  c4[i0] = f4add(a0, b0);
  c4[i1] = f4add(a1, b1);
  c4[i2] = f4add(a2, b2);
  c4[i3] = f4add(a3, b3);
}
```

With float4 loads, the 24 x 512 launch is back to the same bandwidth as launching across all 108 SMs. Four loads in flight per thread are enough for 24 blocks to reach the limit the full grid hits.

**256 MB at 24 x 512: 42 GB/s scalar, 120 GB/s with float4** (NCCL: 116 GB/s)

<figure class="ar-fig" data-fig="chart" data-chart="sm" data-mode="rel">
  <div class="ar-head"><div class="ar-title">Eight A100s, remote-read ring at NCCL's launch budget</div></div>
</figure>

The bandwidth is back without a bigger grid, but small-message latency is still 6 to 10x NCCL's, because the host still submits every step of the ring.

<h3> 4.4 One kernel per GPU </h3>

Wider loads recovered the large-message bandwidth, but at 4 KB the host-driven ring is still about 8x slower than NCCL. Each step moves only 512 bytes per GPU, yet the CPU has to submit a separate copy or kernel and the events that order it. Making those tiny transfers faster won't remove that submission cost.

We can reduce the host's work to one kernel launch per GPU by putting the whole ring inside the kernel. That means the GPUs also have to enforce the dependencies themselves: each GPU waits for its neighbour and signals when its own data is ready, through a flag in memory. The fourteen dependent steps remain, but the CPU no longer has to submit each one.

Here is the reduce-scatter half of the host-driven ring. Here `off(c)` and `len(c)` give the start and length of chunk `c`. The first step reads a neighbour's original input, and later steps read the partial sum it produced:

```cuda
// Host code: one CPU thread queues every step for every GPU.
for (int st = 0; st < P - 1; st++)                 // 7 reduce-scatter steps on eight GPUs
  for (int r = 0; r < P; r++) {                    // for each GPU
    int src = (r - 1 + P) % P, c = (r - st - 1 + P) % P;
    const float* from = (st == 0 ? in[src] : out[src]) + off(c);
    cudaSetDevice(r);
    if (st > 0) cudaStreamWaitEvent(s[r], ev[src][st - 1]);    // wait for the left neighbour's previous step
    add4<<<24, 512, 0, s[r]>>>(in[r] + off(c), from, out[r] + off(c), len(c));   // this step's work
    cudaEventRecord(ev[r][st], s[r]);                           // mark this step done
  }
// all-gather: 7 more steps, with a peer copy instead of add4
```

These calls enqueue work, and events enforce the dependencies on the GPUs. But one CPU thread still makes 112 launches and copies plus over 200 event calls per collective. At 4 KB each step moves only 512 bytes, so submission dominates.

Putting the entire ring into one kernel per GPU leaves the host with eight launches:

```cuda
for (int r = 0; r < P; r++) {                      // host: all that is queued per collective
  cudaSetDevice(r);
  ring_fused_kernel<<<24, 512, 0, s[r]>>>(args, r, P, N, base);
}
```

The step loop moves into the kernel unchanged. The wait and the "done" marker the host used to queue around each step are now done by the GPU itself:

```cuda
__global__ void ring_fused_kernel(FusedArgs a, int me, int P, int N, unsigned base) {
  int src = (me - 1 + P) % P, b = blockIdx.x;
  for (int st = 0; st < P - 1; st++) {                  // reduce-scatter
    int c = (me - st - 1 + P) % P;
    const float* from = (st == 0 ? a.in[src] : a.out[src]) + off(c);
    if (st > 0) wait_step(a.flag[src] + b, base + st);  // was cudaStreamWaitEvent
    add_slice(a.in[me] + off(c), from, a.out[me] + off(c), len(c));   // was the add4 launch
    signal_step(a.flag[me] + b, base + st + 1);         // was cudaEventRecord
  }
  // all-gather: P-1 more steps, copy_slice instead of add_slice (was cudaMemcpyPeerAsync)
}
```

The rounds and data dependencies stay the same. We use one flag per block. Block $b$ handles the same slice on every GPU, so it only waits for the matching block on its neighbour, not the whole GPU. These are the channels from section 4.2, each coordinating independently.

That leaves the two operations used in the kernel above, `wait_step` and `signal_step`. Each flag is a step counter in the signalling GPU's memory. To wait, one thread repeatedly reads its neighbour's counter over NVLink until the required step is complete, while the other threads wait at a block barrier. To signal, the block finishes its writes and publishes the new counter value:

```cuda
__device__ void wait_step(const unsigned* f, unsigned v) {     // f is on the neighbour
  if (threadIdx.x == 0)                                        // one thread polls for the block
    while (load_acquire(f) < v) { }                            // ld.acquire.sys, repeated until the neighbour reaches step v
  __syncthreads();                                             // the other threads wait here for thread 0
}
__device__ void signal_step(unsigned* f, unsigned v) {         // f is local
  __syncthreads();                                             // every thread has issued its writes
  if (threadIdx.x == 0) store_release(f, v);                   // st.release.sys, publishes the new step
}
```

Without the acquire load and the release store, the neighbour could see the new flag before the data it describes, and read stale values.

A waiting block keeps spinning on its SM, so the grid must be small enough for every block to run at once. Otherwise a block can wait for one that never gets scheduled. The 24-block grid fits easily on the A100's 108 SMs, at least on the idle GPUs I measured here.

Reducing host submissions to $P$ kernel launches cut 4 KB latency from **840 µs** for the host-driven ring at 24 x 512 to **261 µs**. That first fused version called `__threadfence_system()` in all 512 threads before writing the flag and reached **119 GB/s** at 256 MB. Replacing those fences with the barrier and one release store from thread 0, as shown above, brought it to **126 GB/s**.

| 8x A100, one process | 16 MB | 256 MB | 1 GB |
|------|-----|-----|-----|
| ring, fused kernel | **63 GB/s** | **126 GB/s** | **131 GB/s** |
| NCCL | 52 GB/s | 116 GB/s | 128 GB/s |

**4 KB: 261 µs, 256 MB: 126 GB/s** (NCCL: 104 µs, 116 GB/s)

At bulk the fused ring now matches or beats NCCL. At 4 KB it's still 2.5x behind. With eight GPUs and a 4 KB input each chunk is only 512 bytes, so almost none of that is transfer time. Every one of the fourteen steps still publishes a flag and polls for one.

<figure class="ar-fig" data-fig="timeline" data-stages="host,fused">
  <div class="ar-head"><div class="ar-title">Host-driven ring vs fused kernel</div></div>
  <figcaption>Grey is the host issuing work, blue is a GPU running a step. The bars underneath compare the measured times.</figcaption>
</figure>

<h3> 4.5 The LL protocol </h3>

So far, a receiver waits for a separate flag before reading the data, similar to what NCCL calls its *Simple* protocol. For small messages, NCCL switches to its *LL* (low latency) protocol, which packs each FP32 value with a flag so the data itself shows when it's ready. Our version pushes `{data, flag, data, flag}` words into the receiver's memory. The receiver polls locally until both flags match the expected step, then uses the accompanying data.

<figure class="ar-fig" data-fig="protocols">
  <div class="ar-head"><div class="ar-title">Data and flags in NCCL's protocols</div></div>
  <figcaption>So far our ring works like Simple, and this section builds LL. LL128 is a third option NCCL uses over NVLink.</figcaption>
</figure>

```cuda
// Sender: publish two values with their readiness flags.
float2 v = src2[i];
st_v4(slot_on_nxt + i, as_uint(v.x), flag, as_uint(v.y), flag);

// Receiver: poll its local slot until both values are ready.
uint4 w;
do { w = ld_v4(my_slot + i); } while (w.y != flag || w.w != flag);
float2 r = make_float2(as_float(w.x), as_float(w.z));
```

Pairing every value with a flag doubles the bytes we send, so we only use LL for chunks up to 1 MB and keep separate flags above that. With LL and the single release store, the final kernel drops from 261 to 82 µs at 4 KB. That's the first time the ring is ahead of NCCL at small sizes.

**4 KB: 82 µs, 256 MB: 126 GB/s** (NCCL: 104 µs, 116 GB/s)

<figure class="ar-fig" data-fig="chart" data-chart="fused" data-mode="lat">
  <div class="ar-head"><div class="ar-title">Eight A100s, from host-driven ring to fused kernel, 24 x 512</div></div>
</figure>

It passes the output checks, but it relies on each 8-byte data/flag pair arriving together, an assumption NCCL's LL also makes but the [PTX memory model](https://docs.nvidia.com/cuda/parallel-thread-execution/#memory-operations-on-vector-data-types) doesn't guarantee for vector stores.

<h3> 4.6 One process per GPU </h3>

So far I'd used one host thread to submit work to all eight GPUs, both for my kernels and for NCCL. Typical DDP training uses one process per GPU, which lets the processes submit their work concurrently. I wanted to see how much of the remaining small-message latency came from this difference in how the benchmark launches work.

In the existing NCCL benchmark, the thread calls `ncclAllReduce` for each GPU inside an `ncclGroupStart`/`ncclGroupEnd` pair, then synchronises the eight devices one after another. With one process per GPU, each rank makes its own call and waits on its own stream, so this changes the NCCL baseline as well as my kernels' timings.

We fork before CUDA initialisation, map peer buffers with CUDA IPC, and initialise NCCL with `ncclCommInitRank`:

```cuda
// parent: no CUDA calls before the fork, a CUDA context doesn't survive it
sh = (Shared*)mmap(nullptr, sizeof(Shared), PROT_READ | PROT_WRITE,
                   MAP_SHARED | MAP_ANONYMOUS, -1, 0);
for (int rank = 0; rank < P; rank++)
  if (fork() == 0) return worker(rank);

// worker: export our buffer, wait for the others, map theirs
cudaSetDevice(rank);
cudaMalloc(&in[rank], bytes);
cudaIpcGetMemHandle(&sh->hin[rank], in[rank]);
if (rank == 0) ncclGetUniqueId(&sh->id);
barrier();   // wait until every rank has exported its handle
for (int r = 0; r < P; r++)
  if (r != rank)
    cudaIpcOpenMemHandle((void**)&in[r], sh->hin[r], cudaIpcMemLazyEnablePeerAccess);
ncclCommInitRank(&comm, P, sh->id, rank);
```

With concurrent submission, NCCL's 4 KB time falls from **104 to 34 µs**, a **3x speedup!** Our fused LL ring falls from **82 to 42 µs**, roughly cutting latency in half. The ring now loses to NCCL at small sizes and only pulls ahead from 16 MB.

**4 KB: 42 µs, 256 MB: 128 GB/s** (NCCL: 34 µs, 118 GB/s)

<h3> 4.7 Picking the algorithm by size </h3>

For the final optimisation, we choose the algorithm based on the message size: direct read for small inputs and the fused ring for large ones. We've made the ring faster, but it still needs fourteen dependent rounds on eight GPUs. For small inputs, we can avoid those rounds entirely: each GPU reads every other GPU's input and computes the full sum itself, using the direct-read algorithm from section 2. Each GPU reads more remote data, $7S$ instead of the ring's $1.75S$, but when the input is only a few KB, avoiding the rounds can matter more than moving fewer bytes.

NCCL makes this kind of choice too, with more options and a more detailed model. Its [tuning model](https://developer.nvidia.com/blog/understanding-nccl-tuning-to-accelerate-gpu-to-gpu-communication/) estimates which algorithm and protocol will be fastest for the message size, number of GPUs and machine topology. That means our NCCL baseline isn't one fixed algorithm across the whole sweep. Our direct-read/ring cutoff is a much simpler version of the same idea.

The cutoff between the two kernels is based on the total remote data each GPU reads, $(P-1)S$. If that is at most 8 MiB, we use direct read; otherwise we use the ring. On eight GPUs, that puts the cutoff at an input size of about 1.14 MiB per GPU.

The direct-read kernel below loads a value from every GPU before summing, using the independent-load idea from section 4.3. The `auto_dispatch` function makes the choice between this kernel and the fused ring. As with the earlier direct-read benchmark, input readiness and buffer lifetime are enforced outside the timed call; we'll account for that limitation when comparing the results.

```cuda
__global__ void sum_all4(Ptrs in, float4* out, int P, int n4) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = gridDim.x * blockDim.x;
  for (int i = tid; i < n4; i += stride) {
    float4 x[MAXP];
    #pragma unroll
    for (int p = 0; p < MAXP; p++) if (p < P) x[p] = ((const float4*)in.p[p])[i];   // issue every load first
    float4 v = x[0];
    for (int p = 1; p < MAXP; p++) if (p < P) v = f4add(v, x[p]);
    out[i] = v;
  }
}
void auto_dispatch(int N) {   // direct read while traffic is small, ring after that
  if ((long)(P - 1) * N * sizeof(float) <= (8L << 20)) direct(N); else ring_fused(N);
}
```

Direct read has no flags at all. My benchmark makes every input ready before timing and keeps the buffers alive until all readers finish, while NCCL does that coordination inside its call. With that head start, direct read is twice as fast as NCCL at 4 KB. So this isn't a drop-in training speedup.

**4 KB: 16 µs, 256 MB: 128 GB/s** (NCCL: 34 µs, 118 GB/s)

NCCL also has [low-latency symmetric-memory kernels](https://developer.nvidia.com/blog/enabling-fast-inference-and-resilient-training-with-nccl-2-27/) for specialised use cases, but those weren't used in these runs.

The combined dispatch is ahead of NCCL at every tested size except around the crossover. At 1 and 4 MB on eight GPUs and 4 MB on four, NCCL leads by 3 to 15%.

<figure class="ar-fig" data-fig="chart" data-chart="mp" data-mode="lat">
  <div class="ar-head"><div class="ar-title">Eight A100s, one process per GPU</div></div>
</figure>

Here's the progression on eight A100s, expressed as a percentage of NCCL's speed: **100% matches NCCL, and higher is faster**. Each row uses the NCCL baseline from the same sweep and process setup. The first six changes use one host process; the last two use one process per GPU.

| Change | 4 KB speed (% of NCCL) ↑ | 256 MB speed (% of NCCL) ↑ |
|---|---:|---:|
| Host-driven copy ring | 9% | 78% |
| Read the peer inside the add kernel | 13% | 102% |
| Limit the scalar kernel to 24 × 512 threads | 14% | 36% |
| Issue independent `float4` loads | 14% | 104% |
| Fuse the ring into one kernel per GPU, with per-thread fences | 40% | 104% |
| Use LL for small chunks and a single release store for the separate-flag path | 127% | 109% |
| Submit from one process per GPU | 82% | 109% |
| Choose direct read for small inputs, fused ring for large ones | 218% | 109% |
| NCCL baseline | 100% | 100% |

Speed is calculated as NCCL time divided by our time, using the unrounded measurements. For example, 109% means 9% faster than NCCL.

<h2> 5. Newer GPUs: what changes on the H100 and B200 </h2>

I then ran the same kernels on H100s and B200s, keeping the A100's 24 x 512 launch budget to see how the tuning carried over.

The H100 provides 1.5x the A100's NVLink bandwidth, and the B200 provides 3x. If our kernels can keep the links busy, large-message times should fall roughly in proportion. These systems also support reduction inside the switch, called NVLink SHARP (NVLS), giving NCCL another way to move fewer bytes. We'll come back to this in section 5.2.

<figure class="ar-fig" data-fig="machine" data-machine="hopper">
  <div class="ar-head"><div class="ar-title">Machine: eight H100s or B200s</div></div>
</figure>

| 8 GPUs, one node | A100 | H100 | B200 |
|---|---|---|---|
| NVLink per GPU, per direction | 300 GB/s | 450 GB/s | 900 GB/s |
| In-switch reduction (NVLS) | no | yes | yes |
| NCCL channels for the ring | 24 | 24 | 32 |

The first surprise was that the B200 ran our fused ring no faster than the A100. Both reach 126 GB/s at 256 MB, and the B200 has three times the link.

<h3> 5.1 Grid size and loads on the B200 </h3>

Our kernel still uses the A100's launch configuration: 24 blocks issuing four 16-byte loads per thread. The B200 has three times the NVLink bandwidth, so we may need more outstanding loads to use it. On the A100, shrinking the grid left too few loads outstanding. Keeping that grid fixed while increasing link bandwidth could leave us with the same bottleneck.

At 24 x 512 the fused ring reaches only **126 GB/s** at 256 MB, compared with NCCL's **338**. To test whether more concurrency would help, I swept the fused kernel over block counts and loads per thread.

One thing I didn't expect came up before the sweep. Rewriting the inner loop so the four loads go through a small array and `#pragma unroll`, instead of four named variables, took the 24 x 512 number from **126 to 160 GB/s** on its own, with the same four loads per thread. On the full grid both versions reach 317 to 321 GB/s, so it only shows when the grid is too small to hide it.

SASS is the GPU's machine code, and it shows where the difference comes from. In the original loop the compiler issues three of the four remote loads, then the adds for the first two, and only then the fourth load. That last load can't go out until the first two have come back. With the array version all four remote loads are issued before the first add.

Here are the loads and adds from both loops, with the address arithmetic left out. The `STRONG.GPU` loads are the remote ones:

<div class="language-sass highlighter-rouge"><div class="highlight"><pre class="highlight"><code><span class="c">// four named variables</span>
<span class="nf">LDG.E.128</span>            R4,  desc[UR20][R12.64]   <span class="c">// local</span>
<span class="k">LDG.E.128.STRONG.GPU</span> R8,  desc[UR20][R8.64]    <span class="c">// remote 1</span>
<span class="k">LDG.E.128.STRONG.GPU</span> R16, desc[UR20][R26.64]   <span class="c">// remote 2</span>
<span class="nf">LDG.E.128</span>            R20, desc[UR20][R24.64]   <span class="c">// local</span>
<span class="k">LDG.E.128.STRONG.GPU</span> R12, desc[UR20][R42.64]   <span class="c">// remote 3</span>
<span class="nf">LDG.E.128</span>            R24, desc[UR20][R24.64]   <span class="c">// local</span>
<span class="n">FADD</span> R4,  R4,  R8                              <span class="c">// needs remote 1, three more follow</span>
<span class="nf">LDG.E.128</span>            R8,  desc[UR20][R44.64]   <span class="c">// local</span>
<span class="n">FADD</span> R18, R22, R18                             <span class="c">// needs remote 2, three more follow</span>
<span class="k">LDG.E.128.STRONG.GPU</span> R20, desc[UR20][R22.64]   <span class="c">// remote 4</span>
</code></pre></div></div>

<div class="language-sass highlighter-rouge"><div class="highlight"><pre class="highlight"><code><span class="c">// array with #pragma unroll</span>
<span class="nf">LDG.E.128</span>            R8,  desc[UR10][R18.64]   <span class="c">// local</span>
<span class="k">LDG.E.128.STRONG.GPU</span> R12, desc[UR10][R16.64]   <span class="c">// remote 1</span>
<span class="nf">LDG.E.128</span>            R20, desc[UR10][R30.64]   <span class="c">// local</span>
<span class="k">LDG.E.128.STRONG.GPU</span> R24, desc[UR10][R28.64]   <span class="c">// remote 2</span>
<span class="k">LDG.E.128.STRONG.GPU</span> R16, desc[UR10][R48.64]   <span class="c">// remote 3</span>
<span class="k">LDG.E.128.STRONG.GPU</span> R28, desc[UR10][R28.64]   <span class="c">// remote 4</span>
<span class="n">FADD</span> R8,  R8,  R12                             <span class="c">// needs remote 1, three more follow</span>
<span class="nf">LDG.E.128</span>            R12, desc[UR10][R50.64]   <span class="c">// local</span>
<span class="n">FADD</span> R22, R22, R26                             <span class="c">// needs remote 2, three more follow</span>
<span class="nf">LDG.E.128</span>            R24, desc[UR10][R26.64]   <span class="c">// local</span>
</code></pre></div></div>

The table uses the array version throughout.

| fused ring, 256 MB | 4 loads | 8 loads | 16 loads |
|---|---|---|---|
| 24 blocks | 160 GB/s | 209 | 240 |
| 32 blocks | 211 | 263 | 314 |
| 48 blocks | 273 | 280 | 297 |
| 96 blocks | 314 | 316 | 291 |
| 148 blocks (full grid) | 317 | 315 | 302 |

Increasing either knob helps, and the best configurations reach about **315 GB/s**, close to the 327 GB/s NCCL's own ring gets with NVLS disabled. NCCL's logs on this node show 32 channels for its ring, against 24 on the A100, and at 32 blocks, 16 loads per thread gets there too.

**B200, 256 MB: 126 → 317 GB/s** (NCCL ring: 327 GB/s, NCCL with NVLS: 338 GB/s)

The estimate from section 4.3 says 24 x 512 with four loads should already saturate the link, yet it only reaches 160 GB/s, so that model is too simple. More blocks works better than deeper unrolling at the same bytes in flight. Deep unrolling also hurts the smaller sizes. At 16 MB on 24 blocks, 16 loads per thread gets 38 GB/s against 55 with four.

The H100 doesn't have this problem. The same sweep there only moves the fused ring between **181 and 186 GB/s** at 256 MB, whatever the block count or loads per thread. One configuration didn't run at all. The 16-load version at 24 x 512 failed to launch on the H100 with a register error, although the same source launched on the B200.

At 1 GB, the tuned rings reach about 131, 192 and 365 GB/s on A100, H100 and B200. Multiplied by the ring's 1.75 traffic factor, that's 229, 336 and 639 GB/s on the link, or roughly 76%, 75% and 71% of each GPU's per-direction NVLink rating.

<h3> 5.2 In-switch reduction (NVLS) </h3>

Up to this point, every addition has happened on a GPU. The switch forwards the partial sums, and our kernels combine them. The H100 and B200 systems here support **[NVLink SHARP (NVLS)](https://arxiv.org/abs/2507.04786)**, which lets the switch participate in the reduction. Contributions can be summed inside NVSwitch and multicast back. InfiniBand SHARP applies the same idea between machines.

This reduces the data each GPU sends and receives to a little over $S$, compared with the eight-GPU ring's $1.75S$. Our peer-read kernel can't use that reduction hardware. To check how much of NCCL's lead comes from it, we can disable NVLS and measure again.

On the H100 it's most of the lead. At 256 MB, NCCL reaches **232 GB/s** compared with our ring's **179**. Disabling NVLS reduces NCCL to **190 GB/s**. After the grid sweep from 5.1, the two plain rings are close to level.

**H100, 256 MB: 186 GB/s** (NCCL ring: 190 GB/s, NCCL with NVLS: 232 GB/s)

On the B200 it matters less. Disabling NVLS costs NCCL **3%** at 256 MB and **8%** at 1 GB. The tuned ring is already close to NCCL's plain ring, so NVLS is most of the remaining gap.

<h3> 5.3 Small-message times barely improve </h3>

At 4 KB, there's very little transfer time to save. Here are the same kernels with one process per GPU:

| 4 KB, 8 GPUs, one process per GPU | A100 | H100 | B200 |
|---|---|---|---|
| direct read, external coordination | 16 µs | 13 µs | 18 µs |
| NCCL | 34 µs | 27 µs | 38 µs |

Advertised link bandwidth triples from A100 to B200, but the 4 KB times barely change. At this size there's almost nothing to transfer, so the time goes to launching kernels and synchronising GPUs, and a faster link doesn't help with either. The B200 logs show NCCL using a single-channel LL ring here.

A faster link also makes direct read's extra bytes cheaper, so the model says direct read should stay ahead up to larger sizes. I kept the A100's 8 MB threshold, and the result is mixed. On the B200 the dispatch stays ahead of NCCL through 1 MB, while on the H100 NCCL takes over from 1 MB.

The curves below combine the original 24-block results below 16 MB with the tuned full-grid results from 16 MB onward. All use one host process. The graph shows algorithm bandwidth in GB/s, so we can compare the two GPU types directly.

<figure class="ar-fig" data-fig="chart" data-chart="newerfinal" data-mode="bw">
  <div class="ar-head"><div class="ar-title">H100 and B200: fused rings against NCCL</div></div>
  <figcaption>Eight GPUs, one host process. Both use 24 × 512 below 16 MB; from 16 MB, H100 uses 132 × 512 and B200 uses 148 × 512. Solid lines show our fused rings; dashed lines show default NCCL on each GPU type. The tuned sweeps and NCCL algorithm checks were measured separately.</figcaption>
</figure>

<h2> 6. Summary </h2>

On the A100s, we matched or beat NCCL's large-message bandwidth by keeping enough loads in flight and moving the ring into one kernel per GPU. The B200 needed more blocks and loads to get close, and on the H100 most of NCCL's lead came from reducing inside the switch. Here's where the experiments ended up:

| Technique | Result | Limit of the comparison |
|---|---|---|
| Choose the transport | On the two VMs without peer access, staged copy engines reach 1.7–2x NCCL's bulk bandwidth | Depends on the VM and path. On the second L4 VM they only match NCCL |
| Keep enough bytes in flight | At 24 x 512 on A100, independent wide loads recover 42 → 120 GB/s | B200 needs more blocks or loads to reach about 317 GB/s |
| Coordinate inside one kernel | A100 4 KB time falls from 840 to 261 µs | One host process, still fourteen ring rounds |
| Use direct read for small messages | A100 4 KB time is 16 µs against NCCL's 34 | One process per GPU, with readiness and buffer lifetime enforced outside timing |
| Reduce inside the switch | Disabling NVLS costs NCCL most of its H100 lead and 3–8% on B200 | Our peer-read kernel doesn't use NVLS |

<figure class="ar-fig" data-fig="bars">
  <div class="ar-head"><div class="ar-title">Algorithm bandwidth at 256 MB per GPU, GB/s</div></div>
</figure>

On a new VM, I'd first check peer access, one transfer, two concurrent transfers, the same with a third GPU involved, and NCCL's selected transport. On the PCIe machines, the link speed on the spec sheet told me very little.

There's a lot I didn't get to. The next steps would be using `multimem` instructions to reduce inside the switch, trying LL128, and going between machines. I'd recommend reading [Demystifying NCCL](https://arxiv.org/abs/2507.04786) and Aleksa Gordić's [Inside TPU and GPU Clusters: The Anatomy of Collective Communication](https://www.aleksagordic.com/blog/collective-operations) for more on all three.

<h2> Resources </h2>

- [Demystifying NCCL: An In-depth Analysis of GPU Communication Protocols and Algorithms](https://arxiv.org/abs/2507.04786)
- [Inside TPU and GPU Clusters: The Anatomy of Collective Communication](https://www.aleksagordic.com/blog/collective-operations) - Aleksa Gordić
- [JAX Scaling Book](https://jax-ml.github.io/scaling-book/)
- [NCCL](https://github.com/NVIDIA/nccl) and [nccl-tests](https://github.com/NVIDIA/nccl-tests) - NVIDIA
- [Enabling fast inference and resilient training with NCCL 2.27](https://developer.nvidia.com/blog/enabling-fast-inference-and-resilient-training-with-nccl-2-27/) - NVIDIA
- [Code and results for this post](https://github.com/PhilipBotros/nccl-allreduce)
