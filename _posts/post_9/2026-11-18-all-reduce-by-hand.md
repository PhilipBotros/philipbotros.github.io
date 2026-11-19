---
layout: post
title: "All-reduce by hand: chasing NCCL from T4s to B200s"
date: 2026-11-18
published: true
cover: /images/post_9/cover.png
background: /images/post_9/cover.png
og_image: /images/post_9/cover.png
mathjax: true
excerpt: <br> Building and benchmarking a CUDA all-reduce across five GPU generations, tracing performance to memory concurrency, synchronisation, compiler scheduling and in-switch reduction.
---
<link rel="stylesheet" href="/assets/post_9/figures.css">
<script src="/assets/post_9/figures.js" defer></script>

Most codebases treat [NCCL](https://developer.nvidia.com/nccl) as a black box, yet recent systems write their own device-side communication so it can overlap tightly with compute. [DeepSeek-V3](https://arxiv.org/html/2412.19437v2#S3.SS2.SSS2) carves out 20 of each H800's 132 SMs for custom communication kernels, and [ParallelKittens](https://hazyresearch.stanford.edu/blog/2025-11-17-fluffy-kittens) fuses GEMMs and collectives in one kernel. I wanted to know what goes into code like that: how one GPU's SMs load from another GPU's HBM, how ranks synchronize without the CPU, and what limits throughput over PCIe and NVLink. For the broader theory of collectives and cluster topologies, I recommend the [JAX Scaling Book](https://jax-ml.github.io/scaling-book/) and Aleksa Gordić’s [Inside TPU and GPU Clusters](https://www.aleksagordic.com/blog/collective-operations).

After writing a single-GPU [matrix multiplication worklog](/2026/08/11/matmul-worklog.html), I did the same for multi-GPU all-reduce, starting with `cudaMemcpy` and a four-line add kernel on two T4s, working up to a fused ring on eight B200s, and timing every step against NCCL. Here are the final 256 MB results:

| Eight GPUs, 256 MB per GPU | Hand-written ring | NCCL ring | Default NCCL (NVLS on H100/B200) |
|---|---:|---:|---:|
| A100 | 126 GB/s | 115 GB/s (Ring/Simple) | 116 GB/s |
| H100 | 181 GB/s | 190 GB/s | 232 GB/s |
| B200 | 314 GB/s | 324 GB/s | 338 GB/s |

The hand-written ring uses as many blocks as NCCL's ring has channels: 24 on A100 and H100, and 32 on B200.

Code and benchmark logs are at [`PhilipBotros/nccl-allreduce`](https://github.com/PhilipBotros/nccl-allreduce).

## 1. Cost model and measurement setup

All-reduce sums input buffers elementwise across $P$ GPUs and leaves a full copy of the result on every GPU. In data-parallel training, for instance, each GPU holds gradients from its local microbatch, and all-reduce gives every rank the summed gradients before the optimizer step.

The simplest algorithm is **one-shot** all-reduce: every GPU reads every peer's input directly and sums them locally. This duplicates the additions across all $P$ GPUs and scales per-GPU read traffic linearly with $P$, but it finishes in a single communication round with no intermediate steps.

<figure class="ar-fig" data-fig="direct">
  <div class="ar-head"><div class="ar-title">One-shot all-reduce on four GPUs</div></div>
  <figcaption>Hover or tap a GPU to see its reads.</figcaption>
</figure>

A **ring** all-reduce trades extra serial steps for lower bandwidth usage. We split the $S$-byte buffer into $P$ chunks. In each step of the first phase (reduce-scatter), every GPU reads one chunk from its predecessor, adds its local contribution, and makes the partial sum available to its successor. After $P-1$ steps, each GPU holds the final sum for one chunk. A second phase of $P-1$ steps (all-gather) circulates those finished chunks around the ring without further additions.

Here is the full schedule on four GPUs:

<figure class="ar-fig" data-fig="ring">
  <figcaption>Reduce-scatter followed by all-gather on four GPUs. The ring is a schedule, and the physical links don't have to form a ring.</figcaption>
</figure>

Let's compare the step count and per-GPU received bytes for $P$ GPUs and $S$ bytes per GPU:

| Algorithm | Dependent rounds | Received per GPU |
|---|---:|---:|
| One-shot: read every peer and sum locally | 1 | $(P-1)S$ |
| Ring: reduce-scatter, then all-gather | $2(P-1)$ | $2\frac{P-1}{P}S$ |

A basic $\alpha$-$\beta$ model gives us a sanity check for every experiment:

$$T \approx n\alpha + \frac{V}{B},$$

where $n$ is the number of dependent steps, $\alpha$ is the fixed latency per step, $V$ is the bytes received per GPU, and $B$ is achieved link bandwidth. On eight GPUs ($P = 8$), a ring receives $2 \times \frac{7}{8}S = 1.75S$ bytes across 14 steps. One-shot receives $7S$ bytes in 1 step. For large buffers, $V/B$ dominates and the ring wins by 4x on traffic. For tiny buffers, paying $14\alpha$ costs much more than reading extra bytes.

This model ignores copy engines, compiler scheduling, and protocol metadata, so we shouldn't take it too literally. Still, every optimization we make should clearly improve $n$, $\alpha$, $V$, or $B$.

<figure class="ar-fig" data-fig="model">
  <div class="ar-head"><div class="ar-title">Cost model: ring against one-shot</div></div>
  <figcaption>Change the GPU count and link to see the crossover move. The bandwidth and per-step costs here are illustrative.</figcaption>
</figure>

Throughout the post we'll track two sizes from each sweep: **4 KB**, where launch and synchronization latency dominate, and **256 MB**, where bulk transfer throughput dominates.

Following [`nccl-tests`](https://github.com/NVIDIA/nccl-tests/blob/master/doc/PERFORMANCE.md), I report **algorithm bandwidth** $S/T$ in GB/s, using power-of-two buffer sizes. For an eight-GPU ring, multiplying algorithm bandwidth by $1.75$ gives the nominal payload traffic on the link (what `nccl-tests` calls bus bandwidth, excluding protocol metadata). On an A100 with 300 GB/s of unidirectional NVLink bandwidth per GPU, an eight-GPU ring tops out at $300 / 1.75 \approx 171\text{ GB/s}$ of algorithm bandwidth.

Timings are medians over 100 iterations (20 for $\ge 64$ MB), using the same harness for each kernel and its NCCL baseline.

## 2. Copy paths and a detour through PCIe

Before building a full ring on NVLink, there are two mechanics choices to sort out first: how one GPU reads another's memory, and what the host's PCIe topology actually lets through.

### Copy engines vs. direct peer loads

Once peer-to-peer (P2P) access is enabled via `cudaDeviceEnablePeerAccess`, a GPU has two ways to read a neighbour's memory:

1. **Copy engine + local kernel:** Queue an asynchronous `cudaMemcpyPeerAsync` on a hardware DMA copy engine to pull the neighbour's data into a local scratch buffer, then run a local add kernel.
2. **Direct peer loads:** Pass the neighbour's device pointer straight into the add kernel so SM threads load remote memory directly over PCIe or NVLink.

In CUDA, the add kernel itself looks identical either way:

```cuda
__global__ void add(const float* a, const float* b, float* out, int N) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < N) out[i] = a[i] + b[i];
}
```

<figure class="ar-fig" data-fig="movers">
  <div class="ar-head"><div class="ar-title">One transfer from GPU 1 to GPU 0</div></div>
  <figcaption>Copy, then add, or read the peer's memory inside the add kernel. Press next to step through both.</figcaption>
</figure>

With a copy engine, `b` is the local scratch buffer. With direct peer loads, `b` points into the other GPU's memory, so `b[i]` becomes a load over PCIe or NVLink straight into registers. That skips the separate copy launch and the round trip through local HBM.

The catch is that SM threads are now responsible for saturating the link. Copy engines leave SMs completely free for compute, whereas direct peer loads use SMs and need enough active warps and independent loads in flight to hide remote memory latency. In return, the transfer is issued from inside the kernel, so the same kernel can overlap compute and communication. And if the cloud hypervisor disables P2P altogether, direct peer loads are unavailable. On these VMs, peer copies still work by staging through host memory.

### A quick detour through virtualised PCIe

Before renting eight-GPU NVLink nodes, I got the basics working on cheap PCIe instances.

First was a dual-T4 VM. With $P = 2$, ring and one-shot move the exact same bytes, which makes it easy to test transfer mechanics in isolation. Copying the buffers sequentially and running `add` took **33 µs at 4 KB**. Putting each GPU on its own CUDA stream so the two directions overlapped dropped that to **19 µs**. Passing the peer pointer directly into `add` brought it down to **14 µs**.

At 256 MB, the T4 VM got weird. One 256 MB peer copy took 27 ms. Running two opposing copies at the same time took 54 ms: profiler traces showed both starting within 10 µs of each other and splitting a single direction's bandwidth. Direct peer reads inside the kernel were even slower, even after switching to grid-stride loops and `float4` loads. Both directions appeared to share a bottleneck somewhere in the VM's virtualised PCIe path.

Four-GPU L4 instances were rougher still. My first one (`g6.12xlarge`, PCIe 4.0) disabled P2P completely, forcing all traffic through host DRAM. NCCL fell back to its shared-memory (`SHM`) transport, while my copy ring queued host-staged copies, add kernels, and CUDA events at every step.

<figure class="ar-fig" data-fig="machine" data-machine="l4">
  <div class="ar-head"><div class="ar-title">Machine: four L4s</div></div>
</figure>

At 256 MB, the host-staged copy ring hit **6.4 GB/s vs. NCCL's 3.7 GB/s** (`nccl-tests` measured 3.8 GB/s on the same box, and a four-A10G VM without P2P showed a similar ~2x gap). At 4 KB, however, my host-staged ring took **239 µs vs. NCCL's 42 µs**. Having one CPU thread submit 24 staged copies, 12 kernels, and cross-stream events killed small-message latency.

A second four-L4 VM did enable P2P, and the peer-read ring ran below 1 GB/s on each three-GPU subset despite faster two-GPU transfers. On four GPUs, forcing NCCL onto P2P also put it below 1 GB/s. Falling back to host staging got both to 3.2 GB/s.

The takeaway from PCIe: cloud VM topologies frequently violate what you'd expect from GPU spec sheets. On a new machine, check `cudaDeviceCanAccessPeer`, unidirectional and bidirectional copy bandwidth, multi-GPU contention, and `NCCL_DEBUG=INFO` before debugging kernel code. Better yet, just move to NVLink!

## 3. Eight A100s: from a copy ring to a fused kernel

Moving to an eight-A100 node (`p4d.24xlarge`) gives us full all-to-all P2P over NVSwitch at 300 GB/s per GPU per direction.

<figure class="ar-fig" data-fig="machine" data-machine="a100">
  <div class="ar-head"><div class="ar-title">Machine: eight A100s</div></div>
</figure>

### Kernel 1: host-driven copy ring (baseline)

Running the host-driven copy-engine ring out of the box reached **90 GB/s** at 256 MB, compared to **116 GB/s** for NCCL (78%). At 4 KB it took **1,149 µs** vs. NCCL's **103 µs** in that sweep. Across eight GPUs and 14 ring steps, the CPU was making hundreds of CUDA runtime calls. In fact, running the four-GPU copy ring on A100 took 219 µs at 4 KB vs. 239 µs on the PCIe L4s. NVLink doesn't help when you're completely bound by host API overhead.

### Kernel 2: direct peer loads inside the add kernel

Dropping `cudaMemcpyPeerAsync` and passing the predecessor's buffer pointer directly into the add kernel gets rid of the local scratch buffer and removes one peer-copy submission per GPU per reduce-scatter step. At 256 MB, bandwidth jumped from **90 GB/s to 118 GB/s**, slightly ahead of NCCL's 116 GB/s.

### But wait, are we really beating NCCL?

Beating NCCL at 256 MB with a four-line scalar add kernel made me suspicious, so I profiled NCCL with Nsight Systems to check its launch configuration. My `add` kernel was launching enough blocks to fill all 108 SMs on every A100. NCCL was launching just **24 blocks**.

In training, communication overlaps with backward-pass GEMMs. An all-reduce that occupies all 108 SMs can compete with a concurrent matmul. That is why DeepSeek-V3, for instance, caps communication at 20 SMs on H800, and why NCCL uses a small number of blocks (one per "channel"). Matching NCCL's block count doesn't automatically prove equal L2 or memory-controller contention alongside a GEMM, but using the whole GPU while NCCL uses 24 blocks isn't a fair fight. So for the remaining A100 ring experiments, I capped the kernel at **24 blocks of 512 threads**.

Converting the scalar `add` kernel into a grid-stride loop at `24 × 512` tanked 256 MB throughput from 118 GB/s down to **42 GB/s**.

### Kernel 3: `float4` vector loads and 4x loop unrolling

Let's work out why 24 blocks bottlenecked at 42 GB/s with our current kernel. By Little's Law, the data we keep in flight has to cover the link for as long as a load takes:

$$Q \ge B_{\text{link}} L.$$

At `24 × 512` we have $12,288$ threads. If each has one 4-byte remote `float` load outstanding, that is $12,288 \times 4\text{ B} = 48\text{ KB}$ in flight. At 300 GB/s, 48 KB covers only about 0.16 µs, and a load over NVLink takes closer to a microsecond. Switching to 16-byte `float4` loads and unrolling the loop 4x, so each thread issues four independent remote loads before touching the results, raises that to $12,288 \times 64\text{ B} = 768\text{ KB}$, which covers about 2.6 µs.

Let's look at the inner loop:

```cuda
// a4 is local, b4 points into the neighbour's memory. Each holds n4 float4s.
int stride = gridDim.x * blockDim.x;
int first = blockIdx.x * blockDim.x + threadIdx.x;

for (int i = first; i + 3 * stride < n4; i += 4 * stride) {
  int i0 = i, i1 = i + stride, i2 = i + 2 * stride, i3 = i + 3 * stride;

  // Issue independent loads before the adds and stores.
  float4 a0 = a4[i0], a1 = a4[i1], a2 = a4[i2], a3 = a4[i3];
  float4 b0 = b4[i0], b1 = b4[i1], b2 = b4[i2], b3 = b4[i3];

  c4[i0] = f4add(a0, b0);
  c4[i1] = f4add(a1, b1);
  c4[i2] = f4add(a2, b2);
  c4[i3] = f4add(a3, b3);
}
```

Still running at `24 × 512`, 256 MB bandwidth jumped straight back from **42 GB/s to 120 GB/s**, matching the 108-SM kernel. (Keep that 768 KB number in mind. When we port to B200 in section 4, this simple model breaks down.)

<figure class="ar-fig" data-fig="chart" data-chart="sm" data-mode="rel">
  <div class="ar-head"><div class="ar-title">Eight A100s: recovering bandwidth at 24 × 512 threads</div></div>
  <figcaption>One host process. Relative speed against NCCL across the message-size sweep.</figcaption>
</figure>

### Kernel 4: fusing all 14 steps into one persistent kernel

At 256 MB we were now matching NCCL, but at 4 KB the unrolled ring still took **840 µs** vs. NCCL's **104 µs**, more than 8x slower! At 4 KB across eight GPUs, each chunk is only 512 bytes. The GPUs were spending almost the entire 840 µs waiting on the host loop:

```cuda
// Reduce-scatter, driven from the host. All-gather is P-1 more steps of copies.
for (int step = 0; step < P - 1; step++) {
  for (int gpu = 0; gpu < P; gpu++) {
    int prev  = (gpu - 1 + P) % P;         // the GPU this one reads from
    int chunk = (gpu - step - 1 + P) % P;  // the chunk this GPU reduces in this step
    cudaSetDevice(gpu);  // host switches to this GPU
    if (step > 0) cudaStreamWaitEvent(stream[gpu], ready[prev][step - 1], 0);  // wait for prev's last step
    add4<<<24, 512, 0, stream[gpu]>>>(in[gpu] + off(chunk), from(prev, step, chunk),
                                      out[gpu] + off(chunk), len(chunk));  // add prev's chunk to ours
    cudaEventRecord(ready[gpu][step], stream[gpu]);  // mark this step done
  }
}
```

Across all 14 steps, the host submits 56 add kernels and 56 all-gather copies. It also records 104 events and issues 152 stream waits. Note that this loop costs us twice. One thread submits every GPU's work in turn, and every step needs its own launch plus an event from the neighbour. Here’s how the timeline changes when the host launches once per GPU and the GPUs coordinate all 14 steps themselves:

<figure class="ar-fig" data-fig="timeline" data-stages="host,fused">
  <div class="ar-head"><div class="ar-title">Moving the ring's step loop onto the GPU</div></div>
  <figcaption>The schematic uses the 1,149 µs copy-ring baseline and illustrative step durations. The host-driven version submits each step. The fused version launches once per GPU and coordinates the steps through memory.</figcaption>
</figure>

To get the host out of the loop, we launch one kernel per GPU and move the 14-step loop inside it. Each host synchronization call gets replaced by a memory operation on a monotonically increasing step counter: `cudaEventRecord` becomes a local release store (`st.release.sys`), and `cudaStreamWaitEvent` becomes a spin-loop polling the predecessor's flag over NVLink with an acquire load (`ld.acquire.sys`):

```cuda
// every GPU's buffers and flags, indexed by GPU
struct FusedArgs { const float* in[MAXP]; float* out[MAXP]; unsigned* flag[MAXP]; };

__global__ void ring_fused_kernel(FusedArgs a, int gpu, int P) {
  int prev = (gpu - 1 + P) % P;                       // the GPU this one reads from
  unsigned* my_flag   = a.flag[gpu]  + blockIdx.x;    // one flag per block
  unsigned* prev_flag = a.flag[prev] + blockIdx.x;    // a flag counts finished steps

  for (int step = 0; step < P - 1; step++) {          // reduce-scatter
    int chunk = (gpu - step - 1 + P) % P;             // the chunk this GPU reduces now
    const float* mine   = a.in[gpu] + off(chunk);
    const float* theirs = (step == 0 ? a.in[prev] : a.out[prev]) + off(chunk);
    float* out = a.out[gpu] + off(chunk);

    if (step > 0) wait_step(prev_flag, step);         // was cudaStreamWaitEvent
    add_slice(mine, theirs, out, len(chunk));         // was the add4 launch
    signal_step(my_flag, step + 1);                   // was cudaEventRecord
  }
  // all-gather: P-1 more steps, copy_slice instead of add_slice
}

__device__ void wait_step(const unsigned* flag, unsigned count) {  // flag is on another GPU
  if (threadIdx.x == 0)                               // thread 0 polls for the whole block
    while (load_acquire(flag) < count) { }            // ld.acquire.sys, over NVLink
  __syncthreads();                                    // the other threads wait here
}

__device__ void signal_step(unsigned* flag, unsigned count) {      // flag is on this GPU
  __syncthreads();                                    // every thread has issued its writes
  if (threadIdx.x == 0) store_release(flag, count);   // st.release.sys
}
```

On step 0, each GPU reads from its predecessor's `in` buffer. On later steps it reads partial sums from `out`. Notice also that every block has its own flag: each of the 24 blocks owns an independent slice of every chunk and only synchronizes with the same block on its neighbours. The listing leaves out two details of the real kernel. The counters keep increasing across calls, so flags never need resetting, and each GPU also waits for its successor to finish reading before it reuses a buffer.

Two details are critical for correctness here:

1. **System-scoped memory ordering (`sys`).** Plain `volatile` loads/stores or `.gpu`-scoped fences aren't enough across NVLink. Without `st.release.sys` and `ld.acquire.sys`, a peer GPU can observe the updated flag before the payload writes land in memory and read stale floats.
2. **Co-resident blocks.** Because `wait_step` spins on the SM, every block in the grid has to be scheduled at the same time. Otherwise spinning consumers can starve producers and deadlock. Twenty-four blocks fit easily on an A100's 108 SMs when the GPU is idle (if you share the GPU with a big compute kernel, you have to manage residency carefully).

Fusing the ring cut 4 KB latency from 840 µs to **261 µs**. At 256 MB it changed nothing at first. In my first draft, every thread called `__threadfence_system()` before the flag was raised, which forces its writes to be visible to other GPUs first, and bandwidth stayed at 119 GB/s. Replacing those 512 fences per block with `__syncthreads()` and the single `st.release.sys` store shown above raised it to **126 GB/s**.

### Kernel 5: packing flags inline with data (LL protocol)

At 261 µs, 4 KB was still 2.5x slower than NCCL (104 µs). To get some inspiration, I had a look at what NCCL does for small messages: it switches to a different protocol, called low-latency (**LL**). To see why that helps, let's trace what happens on each of the 14 steps when data and flags live separately:

1. The sender writes its slice to local HBM, hits `__syncthreads()`, and does a release store to its flag.
2. Thread 0 on the receiver polls that remote flag over NVLink, wakes its block with `__syncthreads()`, and only *then* do threads issue remote NVLink reads to pull the payload.

Every step pays twice: once to learn the data is ready, and again to fetch the data.

LL collapses those two trips into a single push. Instead of receiver-pull with separate flags, the sender packs 8 bytes of data (`float2`) with two copies of a 4-byte step tag into a single 16-byte store (`{data0, flag, data1, flag}`) and writes it directly over NVLink into a scratch slot on the *receiving* GPU. Each receiving thread polls its local slot until both tags match `flag`. As soon as the loop exits, the data is already sitting in registers:

```cuda
// Sender: push two floats + readiness tags into the successor's memory.
float2 v = src2[i];
st_v4(slot_on_next + i, as_uint(v.x), flag, as_uint(v.y), flag);

// Receiver: poll local slots until both halves carry this step's tag.
uint4 w;
do { w = ld_v4(my_slot + i); } while (w.y != flag || w.w != flag);
float2 r = make_float2(as_float(w.x), as_float(w.z));
```

<figure class="ar-fig" data-fig="protocols">
  <div class="ar-head"><div class="ar-title">Separate flags and flags carried with the data</div></div>
  <figcaption>The separate-flag path resembles NCCL's Simple protocol, and this section adds LL. LL128 is a third option NCCL uses over NVLink.</figcaption>
</figure>

Inline flags double wire traffic (8 bytes of tags for every 8 bytes of payload), so we only use LL for chunks up to 1 MB and use separate flags above that.

Adding LL dropped 4 KB latency from 261 µs down to **82 µs**, beating NCCL's **104 µs**.

### Sanity check: one process per GPU

I wanted to time the same submission pattern used in a typical multi-GPU training job. With one worker per GPU, each process can submit its own kernels. My single-process harness submitted to the eight devices in turn, grouped the NCCL calls, and synchronized each stream.

I wrote a second multiprocess harness that `fork`s eight worker processes before CUDA init, exchanges peer buffers with `cudaIpcGetMemHandle` / `cudaIpcOpenMemHandle`, aligns ranks on a host barrier, and gives each rank its own `ncclCommInitRank`:

| Eight A100s, 4 KB | Single host process | One process per GPU |
|---|---:|---:|
| Fused LL ring | 82 µs | 42 µs |
| NCCL | 104 µs | 34 µs |

My 4 KB win disappeared. Giving each GPU its own process cut my fused ring's latency nearly in half, from 82 µs to 42 µs, while NCCL sped up 3x from 104 µs to 34 µs. (At 256 MB nothing meaningful changed: **128 GB/s vs. NCCL's 118 GB/s**.) Concurrent submission helped NCCL more than it helped my kernel launch loop.

Note that this also changes how to read the Kernel 4 result. Part of the drop from 840 µs came from no longer submitting every step from one thread, and one process per GPU would have removed that part too.

### Kernel 6: one-shot all-reduce for small messages

Paying 13 extra ring hops at 4 KB just to avoid reading an extra 21 KB makes no sense, so for small buffers ($(P - 1)S \le 8\text{ MB}$) we drop the ring completely and launch a one-shot kernel that reads `float4`s directly from every peer.

That cuts 4 KB latency on eight A100s from 42 µs to 16 µs vs. NCCL's 34 µs, with the caveat that the benchmark already has all peer inputs ready at launch, whereas `ncclAllReduce` pays to synchronize readiness on every call. In the 1–4 MB range before switching back to the ring, NCCL still leads by 3–15%.

<figure class="ar-fig" data-fig="chart" data-chart="mp" data-mode="lat">
  <div class="ar-head"><div class="ar-title">Eight A100s: one process per GPU</div></div>
</figure>

### Summary of the A100 progression

Here is every step on eight A100s, shown as speed relative to NCCL in the same harness (higher is faster). The first six rows use one host process and the last two use one process per GPU:

| Kernel | Target term | 4 KB | 256 MB |
|---|---|---:|---:|
| 1. Host-driven copy ring (baseline) | | 9% | 78% |
| 2. Direct peer loads (108 SMs) | $B$ | 13% | 102% |
| Cap scalar kernel at 24 × 512 | | 14% | 36% |
| 3. `float4` loads + 4x unrolling (24 × 512) | $B$ | 14% | 104% |
| 4. Fused ring (`__threadfence_system`) | $\alpha$ | 40% | 104% |
| 5. Thread-0 `st.release.sys` + LL protocol | $\alpha$ | 127% | 109% |
| Switch harness to one process per GPU | | 82% | 109% |
| 6. One-shot fast path (inputs ready) | $n$ | 218% | 109% |

## 4. H100 and B200: `ptxas` scheduling, NVLink concurrency, and NVLS

Next, I took the exact `24 × 512` A100 fused ring and ran it on eight-GPU H100 and B200 nodes.

<figure class="ar-fig" data-fig="machine" data-machine="hopper">
  <div class="ar-head"><div class="ar-title">Machine: eight H100s or B200s</div></div>
</figure>

Let's compare the hardware numbers across the three NVSwitch generations:

| 8 GPUs, single node | A100 | H100 | B200 |
|---|---|---|---|
| Unidirectional NVLink per GPU | 300 GB/s | 450 GB/s | 900 GB/s |
| 8-GPU ring algorithm ceiling ($B / 1.75$) | 171 GB/s | 257 GB/s | 514 GB/s |
| In-switch reduction (NVLS) | no | yes | yes |
| NCCL ring channels | 24 | 24 | 32 |

On B200, NVLink is 3x faster than on A100 (900 GB/s vs. 300 GB/s). Yet running our A100 kernel at 256 MB produced **126 GB/s**, the exact same number as A100, while default NCCL hit **338 GB/s** using NVLS, which performs the reduction in the NVSwitch (more on this below). Keeping the grid at `24 × 512` and 4 loads per thread while tripling link speed made me wonder whether the grid had enough outstanding loads.

### Kernel 7: an accidental 27% from `ptxas`

To sweep the unroll factor as a template parameter `U`, I replaced the four hardcoded variables (`b0`, `b1`, `b2`, `b3`) with `float4 b[U]` and a `#pragma unroll` loop. When I re-ran the `U = 4`, `24 × 512` baseline at 256 MB on B200 as a sanity check, bandwidth jumped from **126 GB/s to 160 GB/s** (+27%), and a back-to-back A/B run reproduced the gap.

I dumped the SASS (the GPU's machine code) for both kernels with `cuobjdump --dump-sass` to see what had changed. In the named-variable version, `ptxas` emitted three remote NVLink loads (`LDG.E.128.STRONG.GPU`), then scheduled `FADD`s using the first two loads *before* issuing the fourth remote load. The warp ends up stalling on those first NVLink round-trips before the fourth load even gets sent. With `float4 b[U]` and `#pragma unroll`, `ptxas` emits all four remote loads up front before the first `FADD`.

<details markdown="1">
<summary>The generated SASS</summary>

Address arithmetic is omitted. `STRONG.GPU` marks the remote loads:

<div class="language-sass highlighter-rouge"><div class="highlight"><pre class="highlight"><code><span class="c">// four named variables</span>
<span class="nf">LDG.E.128</span>            R4,  desc[UR20][R12.64]  <span class="c">// local</span>
<span class="k">LDG.E.128.STRONG.GPU</span> R8,  desc[UR20][R8.64]   <span class="c">// remote 1</span>
<span class="k">LDG.E.128.STRONG.GPU</span> R16, desc[UR20][R26.64]  <span class="c">// remote 2</span>
<span class="nf">LDG.E.128</span>            R20, desc[UR20][R24.64]  <span class="c">// local</span>
<span class="k">LDG.E.128.STRONG.GPU</span> R12, desc[UR20][R42.64]  <span class="c">// remote 3</span>
<span class="nf">LDG.E.128</span>            R24, desc[UR20][R24.64]  <span class="c">// local</span>
<span class="n">FADD</span> R4,  R4,  R8                             <span class="c">// needs remote 1, three more follow</span>
<span class="nf">LDG.E.128</span>            R8,  desc[UR20][R44.64]  <span class="c">// local</span>
<span class="n">FADD</span> R18, R22, R18                            <span class="c">// needs remote 2, three more follow</span>
<span class="k">LDG.E.128.STRONG.GPU</span> R20, desc[UR20][R22.64]  <span class="c">// remote 4</span>
</code></pre></div></div>

<div class="language-sass highlighter-rouge"><div class="highlight"><pre class="highlight"><code><span class="c">// array with #pragma unroll</span>
<span class="nf">LDG.E.128</span>            R8,  desc[UR10][R18.64]  <span class="c">// local</span>
<span class="k">LDG.E.128.STRONG.GPU</span> R12, desc[UR10][R16.64]  <span class="c">// remote 1</span>
<span class="nf">LDG.E.128</span>            R20, desc[UR10][R30.64]  <span class="c">// local</span>
<span class="k">LDG.E.128.STRONG.GPU</span> R24, desc[UR10][R28.64]  <span class="c">// remote 2</span>
<span class="k">LDG.E.128.STRONG.GPU</span> R16, desc[UR10][R48.64]  <span class="c">// remote 3</span>
<span class="k">LDG.E.128.STRONG.GPU</span> R28, desc[UR10][R28.64]  <span class="c">// remote 4</span>
<span class="n">FADD</span> R8,  R8,  R12                            <span class="c">// needs remote 1, three more follow</span>
<span class="nf">LDG.E.128</span>            R12, desc[UR10][R50.64]  <span class="c">// local</span>
<span class="n">FADD</span> R22, R22, R26                            <span class="c">// needs remote 2, three more follow</span>
<span class="nf">LDG.E.128</span>            R24, desc[UR10][R26.64]  <span class="c">// local</span>
</code></pre></div></div>

</details>

### Kernel 8: sweeping block count and unroll depth on B200

With the array version, I swept block counts and remote `float4` loads per thread (`U = 4, 8, 16`) at 512 threads/block on B200:

| B200 fused ring, 256 MB | 4 loads/thread | 8 loads/thread | 16 loads/thread |
|---|---:|---:|---:|
| 24 blocks | 160 GB/s | 209 GB/s | 240 GB/s |
| 32 blocks | 211 GB/s | 263 GB/s | 314 GB/s |
| 48 blocks | 273 GB/s | 280 GB/s | 297 GB/s |
| 96 blocks | 314 GB/s | 316 GB/s | 291 GB/s |
| 148 blocks (full grid) | 317 GB/s | 315 GB/s | 302 GB/s |

Increasing either blocks or loads per thread brings B200 to a ceiling around **314–317 GB/s**. In particular, `32 blocks × 16 loads` hits 314 GB/s, matching the 32 blocks NCCL uses for its B200 ring. Cranking `U` up to 16 isn't free everywhere, though: throughput dips slightly at 96–148 blocks, and on 16 MB buffers at 24 blocks it drops to 38 GB/s vs. 55 GB/s for `U = 4`.

Our simple Little's Law estimate from Kernel 3 also breaks down here. At 900 GB/s, `24 blocks × 4 loads` (768 KB) covers ~0.87 µs, which was already tight, yet reaching 314 GB/s took over 5x as much nominal concurrency (`32 × 16`, or 4 MB). Spreading work across SMs also beat unrolling deeper per thread: `48 blocks × 4 loads` and `24 blocks × 8 loads` both put 1.5 MB in flight, yet 48 blocks hit 273 GB/s while 24 blocks stopped at 209 GB/s. Individual SMs seem to hit a local ceiling before the link itself saturates.

On H100, the same sweep stayed flat at **181–186 GB/s**, since 24 blocks were already enough to reach the register-load ceiling on H100's 450 GB/s links.

H100 and B200 have a way around that ceiling, the [Tensor Memory Accelerator (TMA)](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/async-copies.html#using-the-tensor-memory-accelerator-tma). A kernel can ask it to copy a whole tile into shared memory asynchronously, without holding registers for every load in flight. I didn't use it here because the A100 has no TMA, and I wanted to run the same kernel on all three generations.

<figure class="ar-fig" data-fig="sweep">
  <div class="ar-head"><div class="ar-title">B200 and H100: bandwidth against block count</div></div>
  <figcaption>Eight GPUs, 256 MB per GPU, one host process. The B200 has 148 SMs and the H100 has 132. Diamonds show the NVLS-disabled baseline at the ring channel count. The B200 forced-Ring run is reported separately below.</figcaption>
</figure>

### Isolating in-switch reduction (NVLS)

Even after the sweep, H100 topped out at **186 GB/s** while default NCCL hit **232 GB/s**. Adding blocks didn't move the number, so I checked NCCL’s choice of algorithm next.

On the Hopper and Blackwell nodes in these runs, NCCL defaulted to [NVLink SHARP (NVLS)](https://arxiv.org/abs/2507.04786) for large buffers. Instead of forwarding chunks around a GPU-to-GPU ring, kernels issue `multimem.ld_reduce` and `multimem.st` instructions against multicast addresses, and the NVSwitch ASICs perform the reduction inside the switch fabric and broadcast the result. That drops per-GPU link traffic on eight GPUs from $1.75S$ (ring) down to roughly $1S$. Because our kernel reads peer HBM directly and adds on SMs, it always pays the $1.75S$ ring factor.

Forcing NCCL onto its ring algorithm (`NCCL_ALGO=Ring`) made the algorithmic comparison direct. Without NVLS, NCCL's H100 throughput dropped from 232 GB/s to **190 GB/s**, right next to my ring's 186 GB/s. On B200, `NCCL_ALGO=Ring` ran at **324 GB/s** against my ring’s 317 GB/s. Disabling NVLS separately (`NCCL_NVLS_ENABLE=0`) gave 327 GB/s. On H100, forcing Ring closed 42 GB/s of the 46 GB/s gap.

Pushing buffer size to 1 GB, the tuned rings reach **131 GB/s** (A100), **192 GB/s** (H100), and **367 GB/s** (B200, 148 blocks and four loads per thread). Multiplying by 1.75 gives 229, 336, and 643 GB/s of nominal NVLink payload throughput, or 76%, 75%, and 71% of rated unidirectional NVLink bandwidth across all three generations.

<figure class="ar-fig" data-fig="chart" data-chart="newerfinal" data-mode="bw">
  <div class="ar-head"><div class="ar-title">H100 and B200: fused rings against NCCL</div></div>
  <figcaption>Eight GPUs, one host process (without one-shot dispatch). Solid lines show the fused rings (24 × 512 below 16 MB; tuned full-grid 132 × 512 on H100 and 148 × 512 on B200 from 16 MB up), and dashed lines show default NCCL. The full-grid H100 curve reaches 183 GB/s at 256 MB. The best H100 result, 186 GB/s, used 96 blocks and eight loads per thread.</figcaption>
</figure>

### Small-message latency across generations

Let's also check 4 KB latency across all three NVSwitch generations (one process per GPU):

| Eight GPUs, 4 KB | A100 | H100 | B200 |
|---|---:|---:|---:|
| One-shot (inputs ready beforehand) | 16 µs | 13 µs | 18 µs |
| NCCL | 34 µs | 27 µs | 38 µs |

4 KB takes ~14 ns on a 300 GB/s link and ~5 ns at 900 GB/s, so tripling NVLink bandwidth barely registers. Practically the whole 13–18 µs is kernel launch and cross-GPU memory latency.

## Wrapping up

That's about as far as I took this. I wanted to understand how custom GPU collectives actually work, and half the time went into things I didn't expect: broken cloud PCIe topologies, single-process launch bottlenecks, and `ptxas` scheduling quirks. Link bandwidth itself was rarely the problem.

There's obviously more you'd need to make any of this useful inside a training loop, like testing it next to a concurrent GEMM or trying TMA and `multimem` on Hopper and Blackwell.

### References

- [Code and benchmark sweeps for this post](https://github.com/PhilipBotros/nccl-allreduce)
- [Demystifying NCCL: An In-depth Analysis of GPU Communication Protocols and Algorithms](https://arxiv.org/abs/2507.04786)
- [Inside TPU and GPU Clusters: The Anatomy of Collective Communication](https://www.aleksagordic.com/blog/collective-operations), Aleksa Gordić
- [Loads and Loads of Fluffy Kittens](https://hazyresearch.stanford.edu/blog/2025-11-17-fluffy-kittens), Hazy Research
- [DeepSeek-V3 Technical Report](https://arxiv.org/abs/2412.19437), section 3.2.2
- [How to Scale Your Model (JAX Scaling Book)](https://jax-ml.github.io/scaling-book/)
- [NCCL source](https://github.com/NVIDIA/nccl) and [nccl-tests bandwidth definitions](https://github.com/NVIDIA/nccl-tests/blob/master/doc/PERFORMANCE.md)
