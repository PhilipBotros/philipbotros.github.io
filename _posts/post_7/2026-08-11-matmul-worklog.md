---
layout: post
title: "How to get a matmul kernel into the top percentile (worklog)"
date: 2026-08-11
cover: /images/post_7/perc_99.png
background: /images/post_7/perc_99.png
og_image: /images/post_7/register_tiling_kernel.png
mathjax: true
excerpt: <br> A worklog on taking a CUDA matmul kernel from naive to the top 1% on a T4, building intuition for GPU performance along the way.
---
Around Christmas last year I was looking for a fun side project and ended up looking more into CUDA. I've found [LeetGPU](https://leetgpu.com) to be extremely helpful in having a sandbox to test kernels for different problems on different hardware without having to spin up your own VM. I ended up with a [challenge](https://leetgpu.com/challenges/matrix-multiplication) to get to the top percentile of a mat mul and this is what I learned.

More than anything, this is a worklog. The goal is not to present a definitive GEMM optimization guide, but to document how I learned to reason about GPU performance by iteratively improving a kernel and understanding the bottlenecks encountered along the way. The focus is less on absolute performance and more on building intuition for how memory hierarchy, data reuse, and execution resources interact to determine performance.

There are already excellent deep dives into CUDA GEMM implementations such as [How to Optimize a CUDA Matmul Kernel for cuBLAS-like Performance](https://siboehm.com/articles/22/CUDA-MMM) by Simon Boehm and [Matrix Multiplication on GPU](https://www.aleksagordic.com/blog/matmul) by Aleksa Gordic. Rather than focusing on the CUDA programming model itself, this post focuses on the performance bottlenecks that shaped each optimization step.

{:refdef: style="text-align: center;"}
![Matmul Optimization Journey]({{ "/images/post_7/perc_99.png"}}){: width="50%" style="border-radius: 12px;" }
{: refdef}

*Note: As you can see the kernels were written around Christmas last year, I'm sure Claude/Codex will one shot a better one now.*

<h2> GPU bottlenecks </h2>

Before we start implementing the kernels it's a good idea to reason about the hardware.

The two fundamental properties of a compute system are:
1. Operation throughput (how many operations per second it can execute)
2. Data supply rate (how quickly operands can be delivered to those operations)

The time of your algorithm is dominated by:

$$\text{Time} \geq \max(\text{compute_time},\ \text{data_movement_time})$$

In other words, performance is bounded by either computation or communication.

If we look at modern GPUs, they have a ridiculous amount of FLOPs and it's growing much faster than the amount of memory and memory bandwidth. This phenomenon has been dubbed the "memory wall" as characterized by [this paper](https://arxiv.org/abs/2403.14123). Their graph shows why this is the case:
{:refdef: style="text-align: center;"}
![Memory Wall]({{ "/images/post_7/mem_wall.png"}})
{: refdef}

Stephen Jones from Nvidia gives a good explanation in [this lecture](https://www.youtube.com/watch?v=3l10o0DYJXg). Fundamentally, SRAM needs 6 transistors per bit so fast memory takes up a lot of space on the die and memory bandwidth is increasingly constrained by physical limits, the speed of light becomes a factor when moving data across chips. There's no reason at this point to believe this is going to change soon so the focus on memory optimization will stay relevant.

This has been a known problem in computing for a while and memory hierarchy systems have been devised to partly overcome this issue. CPUs rely on hardware-managed caches (L1/L2/L3), while GPUs combine caches with programmer-managed shared memory to keep data close to the compute units. TPUs take this to the extreme and drop caches almost entirely in favor of a large compiler-managed scratchpad, with the compiler (XLA) scheduling all data movement so operands stream deterministically into the compute units. Given the predictable nature of data movement for the dominant ML operations, this makes a lot of sense (and saves a lot of die area).

As we'll mostly be focusing on optimising the use of the memory hierarchy, let's have a look at the hierarchy of an Nvidia GPU. The numbers shown are representative of a modern data-center GPU, exact capacities vary per architecture (the T4 we'll be using has 16 GB of GDDR6 and 64 KB of shared memory per SM for example).
{:refdef: style="text-align: center;"}
![Memory Hierarchy]({{ "/images/post_7/cache_hierarchy.png"}})
{: refdef}

As you can see there is quite an elaborate memory hierarchy divided into programmer and hardware managed levels. On one side you have the caches, where the hardware stores recently used data. On the other side you're provided with shared memory which is under our control, and registers which the compiler allocates for us, to minimize round trips to very expensive global memory. There's an inverse relationship between size and speed and the difference between global memory to shared memory to registers is 20-30x per step!

Given this hierarchy, the challenge becomes restructuring algorithms so data remains in fast memory as long as possible. This typically means breaking problems into tiles that fit into shared memory and registers. Matrix multiplication is a classic example, and modern algorithms like FlashAttention apply the same principle by restructuring attention into SRAM-sized tiles.

However, locality alone is not enough. Some memory accesses are unavoidable, and GPUs address this through massive parallelism. By keeping many more warps in flight than execution units, the scheduler can switch to ready work whenever others stall on memory.

Together, locality and parallelism form the two fundamental strategies for achieving high performance on GPUs:

1.	Reduce memory traffic through locality and reuse
2.	Hide remaining latency through massive parallelism

<h2> Setting the target </h2>

Before diving in, let's establish some targets. Performance is measured with M=8192, N=6144, K=4096 on a T4 GPU. Given a complexity of O(MNK) for a matrix multiplication and a single FMA counting as 2 operations the total work is $2 \times 8192 \times 6144 \times 4096 \approx$ **412 billion FLOPs**. The T4's peak FP32 throughput is **8.1 TFLOPS**, giving a theoretical minimum runtime of

$$\frac{412 \times 10^9}{8.1 \times 10^{12}} \approx 50.9 \text{ ms.}$$

The two input matrices total $(8192 \times 6144 + 6144 \times 4096) \times 4 \approx$ **302 MB** and writing the output adds another $8192 \times 4096 \times 4 \approx$ **134 MB**, giving an arithmetic intensity (FLOPs per byte transferred) of $\sim$**945 FLOPs/byte**. The T4's memory bandwidth is **320 GB/s**, so its [roofline](https://modal.com/gpu-glossary/perf/roofline-model) crossover is $8.1 \times 10^{12} / 320 \times 10^{9} \approx$ **25.3 FLOPs/byte**, so at these dimensions the problem is firmly **compute-bound**, assuming we reuse data well enough to only move each matrix roughly once. The floor is set by FP32 throughput, not memory bandwidth, though as we'll see shortly individual kernels can still be very much memory-bound.

In practice, cuBLAS typically achieves 85-95% of peak FP32, putting it around **54-60 ms** for this problem.


<h2>Starting Point: Fully Naive Implementation</h2>

Given matrices $A \in \mathbb{R}^{M \times N}$ and $B \in \mathbb{R}^{N \times K}$, the output matrix $C \in \mathbb{R}^{M \times K}$ is defined element-wise as:

$$C_{i,j} = \sum_{n=1}^{N} A_{i,n} \cdot B_{n,j}, \quad i \in [1, M],\; j \in [1, K].$$

Note that I follow LeetGPU's convention here where $N$ is the reduction dimension, whereas most GEMM literature multiplies $M \times K$ by $K \times N$ and reduces over $K$.

The most basic matrix multiplication kernel looks like this:
```cuda
__global__ void matrix_multiplication_kernel(const float* A, const float* B, float* C, int M, int N, int K) {
    int row = blockDim.y * blockIdx.y + threadIdx.y;
    int col = blockDim.x * blockIdx.x + threadIdx.x;
    if (row < M && col < K) {
        float acc = 0.0f;
        for (int i = 0; i < N; ++i) {
            acc += A[row * N + i] * B[i * K + col];
        }
        C[row * K + col] = acc;
    }
}
```

Each thread computes one output element of the matrix C by loading an entire row from A and an entire column from B from global memory. If we assume they're both square with a size of N, this means every active thread loads 4N bytes from A and 4N bytes from B given FP32 inputs (ignoring the write back to global memory). Every thread does N multiplications and N additions. 

{:refdef: style="text-align: center;"}
![Naive matmul kernel]({{ "/images/post_7/naive_kernel.png"}})
{: refdef}

A good way of measuring algorithmic performance is arithmetic intensity. This translates to the number of FLOPs performed for every byte loaded from global memory. The higher this number the more we are bottlenecked by compute speed versus global memory bandwidth. The formula for this is:

$$ AI = \frac{\text{FLOPs}}{\text{Bytes Transferred}} $$

Arithmetic Intensity for this kernel:


**2N FLOPs / 8N bytes = 0.25 FLOPs/byte.** This is extremely low, for every FLOP we need to load 4 bytes! The T4 GPU has a peak bandwidth of ~320 GB/s and peak compute of ~8.1 TFLOPS. At 0.25 FLOPs/byte, bandwidth limits us to $0.25 \times 320 = 80$ GFLOPS, which is less than 1% of peak compute. In practice the caches absorb part of this traffic, threads in the same half-warp read the same row of A for example, which is why the measured runtime below works out to roughly 430 GFLOPS instead. The conclusion still stands though, we are heavily memory-bound and far away from peak compute. We'll aim to close this gap over the course of this post.

**Runtime: 956.41 ms percentile 16.9**

Unsurprisingly we are in the bottom quartile of performance given this naive implementation, time to do better!

<h2>Optimization 1: Tiled Matmul with Shared Memory</h2>
The starting point to reduce global memory traffic is a textbook tiled matrix multiplication. We divide the problem into tiles and sequentially load tiles into shared memory and compute partial accumulations until the full matrices have been processed. The inner sum can be split into consecutive chunks of size $T$:

$$C_{ij} = \sum_{n=1}^{N} A_{in} \cdot B_{nj} = \sum_{t=0}^{\lceil N/T \rceil - 1} \sum_{k=0}^{T-1} A_{i,\, tT+k} \cdot B_{tT+k,\, j}.$$

Each inner sum is a partial accumulation over a single tile. Because addition is associative over the reals, we can split the reduction dimension into smaller chunks and accumulate partial dot products without changing the final result (FP32 reassociation can change rounding, but this loop sums in the same ascending order as the naive kernel so the result is identical). At every tile step, each thread loads one element into shared memory, we synchronize, then each thread computes one (partial) output element. Graphically this looks like:
{:refdef: style="text-align: center;"}
![Tiled matmul kernel]({{ "/images/post_7/tiled_matmul_kernel.png"}})
{: refdef}

The code changes are relatively simple: we need to initialize shared memory, and at every step cooperatively load a tile and do the partial multiplication.
```cuda
#define TILE_SIZE 16

__global__ void matrix_multiplication_kernel(const float* A, const float* B, float* C, int M, int N, int K) {
    __shared__ float tile_A[TILE_SIZE][TILE_SIZE];
    __shared__ float tile_B[TILE_SIZE][TILE_SIZE];

    int row = blockDim.y * blockIdx.y + threadIdx.y;
    int col = blockDim.x * blockIdx.x + threadIdx.x;

    float acc = 0.0f;
    int num_tiles = (N + TILE_SIZE - 1) / TILE_SIZE;

    for (int t = 0; t < num_tiles; t++) {
        // Each thread loads one element
        tile_A[threadIdx.y][threadIdx.x] = A[row * N + t * TILE_SIZE + threadIdx.x];
        tile_B[threadIdx.y][threadIdx.x] = B[(t * TILE_SIZE + threadIdx.y) * K + col];
        __syncthreads();

        for (int k = 0; k < TILE_SIZE; k++) {
            acc += tile_A[threadIdx.y][k] * tile_B[k][threadIdx.x];
        }
        __syncthreads();
    }
    C[row * K + col] = acc;
}
```
Note that this version drops the bounds checks for brevity and assumes the matrix dimensions are divisible by TILE_SIZE (which holds for our benchmark sizes).

The win here over the fully naive implementation is that we're reusing data across threads via shared memory. Without tiling, each thread would load an entire row of A and column of B from global memory. With tiling, we cooperatively load tiles into shared memory and reuse these across the block.

Each thread still computes a single output element of $C$ and therefore performs approximately $2N$ FLOPs. The key difference is that global memory loads are now amortized across the block. At each tile iteration, every active thread loads one FP32 value from $A$ and one from $B$, for a total of $8$ bytes from global memory. Since the dot product spans $\lceil N / T \rceil$ tiles, each thread loads $8 \times \lceil N / T \rceil$ bytes in total. This results in a per-thread arithmetic intensity of

$$\frac{2N}{8 \lceil N / T \rceil} \approx \frac{T}{4}.$$

For a tile size of $T = 16$, this gives an arithmetic intensity of $\mathbf{4.0}$ **FLOPs per byte**. This is a 16x increase over the naive kernel!

An interesting observation is that pushing the tile size to the maximum supported by a single block (1024 threads, i.e. $T = 32$) doubles the arithmetic intensity to 8.0 FLOPs/byte, yet performance on a T4 actually decreases. This highlights an important caveat of roofline-style reasoning: increasing arithmetic intensity does not automatically translate to higher throughput.

Even at 8.0 FLOPs/byte we are still below the T4's crossover, and neither version gets anywhere near its bandwidth ceiling, so the limit lies elsewhere. It's not occupancy either, both configurations can keep the same 32 warps resident on a Turing SM. One thing that does change is block independence. A 32×32 tile fills the SM with a single 1024-thread block, so at every `__syncthreads()` the warps that arrive early can only sit and wait, there is no other block whose warps could run in the meantime. With 16×16 tiles, several independent blocks live on the same SM and the scheduler can run warps from another block while one waits at a barrier. My best guess is that this loss of scheduling flexibility explains the slowdown.

**Runtime: 593.53 ms percentile 76.1**

But there's still a problem: each thread only computes one output, and we're only getting one FMA (fused multiply-add) per two shared memory loads. We can do better.

<h2>Optimization 2: Thread Coarsening</h2>
The key for the next optimization is that each thread can compute multiple outputs. If a single thread computes a $T_m \times T_n$ tile of outputs instead of a single element (sometimes called a micro-tile), we can reuse loaded values (inside the fast thread registers) across those outputs. Remember that registers are ~1 cycle away and shared memory ~20-30 cycles so this is worth optimizing. Just adding micro-tiling while keeping the original kernel did not work! A few changes had to be made to get it to work correctly which I'll go over now. We now define a set of different parameters first:

```cuda
#define BM 64   // Block tile rows
#define BN 64   // Block tile cols
#define BK 8    // Reduction tile
#define TM 4    // Thread tile rows
#define TN 4    // Thread tile cols
```
We are now covering a block of 64x64 where every thread computes a micro-tile of 4x4. We're still using threadblocks of 16x16 like in the previous kernel but the effective size has quadrupled. Furthermore, we decouple the reduction dimension (BK) from the block dimensions. Once we expand the output tile to 64×64, keeping the reduction tile square would drastically increase shared memory usage and register pressure. Using a smaller BK is necessary to make this larger tile shape work without collapsing occupancy. 

{:refdef: style="text-align: center;"}
![Register tiling kernel]({{ "/images/post_7/register_tiling_kernel.png"}})
{: refdef}

We now have to set up thread specific accumulator tiles (in registers) and we can do a form of **cooperative loading**. Indexing becomes slightly more involved as we now have another dimension to account for (micro-tile).  


```cuda
__global__ void matmul_coarsened_tiled_basic(
    const float* A, const float* B, float* C,
    int M, int N, int K_dim
) {
    int ty = threadIdx.y; 
    int tx = threadIdx.x;

    int block_row = blockIdx.y * BM;
    int block_col = blockIdx.x * BN;

    int row_base = block_row + ty * TM;
    int col_base = block_col + tx * TN;

    // Block tiles: loaded from global memory into shared memory
    __shared__ float tile_A[BM][BK]; // 64x8
    __shared__ float tile_B[BK][BN]; // 8x64

    // Micro-tiles: per-thread accumulators stored in registers
    float acc[TM][TN];
    for (int i = 0; i < TM; i++) {
        for (int j = 0; j < TN; j++) {
            acc[i][j] = 0.0f;
        }
    }

    int num_tiles = (N + BK - 1) / BK;
```

I want to zoom in on the loading pattern, we now have 16x16=256 threads in a block needing to load 64x8 values for A and 8x64 for B = 1024. Previously every active thread was simply loading a single element of both tiles like so, as the tiles were the size of the block:
```cuda
// Each thread loads one element
tile_A[threadIdx.y][threadIdx.x] = A[row * N + t * TILE_SIZE + threadIdx.x];
tile_B[threadIdx.y][threadIdx.x] = B[(t * TILE_SIZE + threadIdx.y) * K + col];
__syncthreads();
```
Since every thread now has multiple elements to load we have a choice to make inside our kernel on how to design this.
We could theoretically have every thread load 2 values from A and 2 values from B but the indexing will become quite cumbersome.

Alternatively, we can have only a subset of the threads participate (i.e. tx < BK), which keeps the mapping simple: tx directly indexes the BK columns and each participating thread loads a short vertical strip of TM elements. This keeps the A loads coalesced (lanes tx 0-7 read consecutive addresses) and reduces the amount of index arithmetic and branching in the load path. The B loads are not ideal here, neighboring lanes end up 4 floats apart, but we'll fix the loading pattern properly in the next optimization.

{:refdef: style="text-align: center;"}
![Cooperative Loading]({{ "/images/post_7/coop_loading.png"}})
{: refdef}

```cuda
    for (int t = 0; t < num_tiles; t++) {
        int k0 = t * BK;

        // ----------------------------
        // Load A tile: BM x BK
        // Use tx to cover BK columns (BK=8), so only tx<8 participates.
        // Each participating thread loads TM rows.
        // Total loads = (BM/TM) * BK * TM = BM*BK
        // ----------------------------
        if (tx < BK) {
            for (int i = 0; i < TM; i++) {
                int a_row = block_row + ty * TM + i;
                int a_col = k0 + tx;
                float v = 0.0f;
                if (a_row < M && a_col < N) {
                    v = A[a_row * N + a_col];
                }
                tile_A[ty * TM + i][tx] = v;
            }
        }

        // ----------------------------
        // Load B tile: BK x BN
        // Use ty to cover BK rows (BK=8), so only ty<8 participates.
        // Each participating thread loads TN columns.
        // Total loads = BK * (BN/TN) * TN = BK*BN
        // ----------------------------
        if (ty < BK) {
            for (int j = 0; j < TN; j++) {
                int b_row = k0 + ty;
                int b_col = block_col + tx * TN + j;
                float v = 0.0f;
                if (b_row < N && b_col < K_dim) {
                    v = B[b_row * K_dim + b_col];
                }
                tile_B[ty][tx * TN + j] = v;
            }
        }

        __syncthreads();
```

After loading the tile, each thread pulls its slice of the shared tile into register fragments and computes the outer product:

```cuda
float acc[TM][TN] = {0.0f};

for (int k = 0; k < BK; k++) {
    float a_frag[TM];
    float b_frag[TN];

    // Load fragments into registers
    for (int i = 0; i < TM; i++)
        a_frag[i] = tile_A[ty * TM + i][k];
    for (int j = 0; j < TN; j++)
        b_frag[j] = tile_B[k][tx * TN + j];

    // Outer product - this is where register reuse happens
    for (int i = 0; i < TM; i++)
        for (int j = 0; j < TN; j++)
            acc[i][j] += a_frag[i] * b_frag[j];
}
```

The difference is in the outer product. We load 4 values from A and 4 values from B (8 loads total), then perform 16 FMAs. Each a_frag[i] is used 4 times, each b_frag[j] is used 4 times. Compare this to the naive version: 2 loads → 1 FMA. Now we have 8 loads → 16 FMAs.

Outside of adding micro-tiling we needed to decouple the block reduction dimension and add cooperative loading. The good thing is this was the largest refactor and the skeleton of the kernel stays roughly the same from now.
<!-- 
The third kernel uses a larger block tile ($64 \times 64$) and further optimizes by having each thread compute multiple results ($4 \times 4 = 16$ outputs) using registers. This increases data reuse at two levels: the block level (Global $\rightarrow$ Shared) and the thread level (Shared $\rightarrow$ Registers). -->

Every thread now does $2N \times TM \times TN$ FLOPs. The cooperative loading is spread unevenly across the block, some threads load 8 elements per step, some 4 and some none, but averaged over the 256 threads the 1024 tile elements work out to 4 elements (16 bytes) per thread per step, or $16 \times \lceil N / BK \rceil$ bytes across the full reduction.

{:refdef: style="text-align: center;"}
![Register Tiling Traffic Comparison]({{ "/images/post_7/register_tiling.png"}})
{: refdef}

**Global Memory Intensity (The Roofline AI)**

The intensity relative to global memory depends on the Block Tile size ($BM \times BN$).

FLOPs per block iteration: $BM \times BN \times BK \times 2 = 64 \times 64 \times 8 \times 2 = 65{,}536$.

Bytes per block iteration: $(BM \times BK + BN \times BK) \times 4 = (64 \times 8 + 64 \times 8) \times 4 = 4{,}096$.

Calculation:

$$\frac{64 \times 64}{2(64 + 64)} = \frac{4096}{256} = \mathbf{16.0}.$$

Note that this quadrupled the global arithmetic intensity compared to the previous kernel (16 vs 4 FLOPs/byte at the global level). That's because the global memory AI depends on the block tile dimensions ($BM \times BN$), which quadrupled (from $16 \times 16$ to $64 \times 64$), while the number of threads per block stayed the same ($16 \times 16$). The per-thread AI moved in lockstep, from $4$ to $\mathbf{16}$ FLOPs/byte. On top of that comes the reuse inside the outer product, where every fragment value fetched from shared memory is used $TM$ (or $TN$) times — this reuse happens in registers, which is where the real win is.

**Runtime: 194.28 ms percentile 92.5**

<h2>Optimization 3: Coalesced Global Memory Loading</h2>
Now we have a decent arithmetic intensity and keep data local it's time to look at the memory traffic. Remember the B load from the previous kernel:

```cuda
// Bad: at any fixed j, neighboring lanes are TN floats apart
tile_B[ty][tx * TN + j] = B[(k0 + ty) * K_dim + block_col + tx * TN + j];
```
The problem is what happens at any given moment. At a fixed j, thread 0 is loading column 0, thread 1 is loading column 4, thread 2 is loading column 8. Global memory is fetched in 32 byte sectors, so with neighboring lanes 4 floats apart we only use a quarter of every sector we touch. The key thing to remember is that global memory coalescing is decided at the warp level, not per thread. In CUDA, a load instruction is executed by 32 threads in lockstep, and the hardware coalesces the 32 addresses requested by the warp into as few memory transactions as possible.

That means a pattern can look “nice” within each thread (each thread walks consecutive elements), yet still be wasteful if neighboring lanes leave gaps between their addresses at the same instruction.
The fix is to make threads cooperate on the load: assign a linear thread id and have the block stride collectively through the tile. Then, for each load instruction, consecutive lanes tend to fetch consecutive addresses, which yields coalesced transactions:

```cuda
int tid = ty * blockDim.x + tx;
int num_threads = blockDim.x * blockDim.y;

for (int i = tid; i < BM * BK; i += num_threads) {
    int a_row = i / BK;
    int a_col = i % BK;
    int global_row = blockIdx.y * BM + a_row;
    int global_col = k_offset + a_col;
    tile_A[a_row][a_col] = A[global_row * N + global_col];
}
```
The same scheme loads the B tile.

```
// BM=64, BK=8, 256 threads, tile has 512 elements → 2 iterations per thread
// First iteration (i = tid):
Thread 0   loads tile_A[0][0]  → A[row+0, k+0]
Thread 1   loads tile_A[0][1]  → A[row+0, k+1]
...
Thread 7   loads tile_A[0][7]  → A[row+0, k+7]
Thread 8   loads tile_A[1][0]  → A[row+1, k+0]
...
Thread 255 loads tile_A[31][7] → A[row+31, k+7]

// Second iteration (i = tid + 256):
Thread 0   loads tile_A[32][0] → A[row+32, k+0]
...
Thread 255 loads tile_A[63][7] → A[row+63, k+7]
```

Now at any given moment, thread 0 loads address 0, thread 1 loads address 1, thread 2 loads address 2. The memory controller can service the warp with a few fully utilized transactions instead of wasting most of every fetch, coalescing is about not paying for bytes you don't use.

{:refdef: style="text-align: center;"}
![Global Memory Coalescing]({{ "/images/post_7/coalescing.png"}}){: width="75%" }
{: refdef}

As you can see the speedup is very modest here. The effect became larger once the tile sizes grew in the later kernels, where coalesced loading was worth about 0.7 percentile points (96.3 vs 97.0).

**Runtime: 193.43 ms percentile 92.6**

<h2>Optimization 4: Double Buffering</h2>

The next optimization tries to overlap memory loading with computation, a form of software pipelining. With double buffering, we use two sets of shared memory and load the next tile while computing on the current one. The loads themselves are ordinary synchronous loads as the T4 has no hardware support for asynchronous copies, so we rely on the compiler and warp scheduler to actually overlap the independent work (more on this below).

{:refdef: style="text-align: center;"}
![Double Buffering]({{ "/images/post_7/double_buffering.png"}}){: width="75%" }
{: refdef}

In code this looks roughly like:

```cuda
__shared__ float tile_A[2][BM][BK];
__shared__ float tile_B[2][BK][BN];

// Load first tile
load_tile(tile_A[0], tile_B[0], t=0);
__syncthreads();

for (int t = 0; t < num_tiles; t++) {
    int curr = t % 2;
    int next = (t + 1) % 2;

    // Load next tile while computing current
    if (t + 1 < num_tiles)
        load_tile(tile_A[next], tile_B[next], t + 1);

    compute(tile_A[curr], tile_B[curr]);
    __syncthreads();
}
```
This gave a modest speedup by hiding some memory latency behind computation. This did not hold when the tile size was increased! Also note that the T4 does not support async copies (`cp.async`), the loads go through registers and compete for the same issue slots, although the warp scheduler can still overlap them with compute from other warps as usual. On newer architectures (Ampere and beyond), `cp.async` offloads global-to-shared memory copies to dedicated hardware, freeing the warp to execute compute instructions while the copy completes in the background.

**Runtime: 190.58 ms percentile 92.7**

<h2>Optimization 5: Larger Tiles</h2>

At this point I increased the tile sizes to better utilize the T4's resources:

```cuda
// Before
#define BM 64
#define BN 64
#define BK 8
#define TM 4
#define TN 4
// Each thread: 16 outputs

// After
#define BM 128
#define BN 128
#define BK 12
#define TM 8
#define TN 8
// Each thread: 64 outputs
```
**Runtime: 169.00 ms percentile 94.1**

The T4 has 64 KB of shared memory per SM, and CUDA caps register usage at 255 registers per thread. With 128×128 tiles and 8×8 thread tiles, we're using about 24 KB of shared memory (12 KB, doubled by the double buffering) and ~100 registers per thread, both well within limits.

More work per thread means better register reuse and fewer synchronization points. Also note increasing the tile sizes is not free! We now can schedule fewer warps per SM as every warp takes up more register space, which can lead to a lower occupancy and reduce the kernel's performance. We'll see an example of this at the end where we run our kernel on newer GPU architectures. Turns out we can crank it up even further:
```
#define BM 192
#define BN 192
#define BK 16
#define TM 12
#define TN 12
// Each thread: 144 outputs
```
**Runtime: 136.86 ms percentile 95.7**

<h2>Optimization 6: Removing double buffering with larger tiles</h2>

After increasing tile sizes, double buffering no longer improved performance and actually hurt it. One likely suspect is resource pressure: double buffering doubles shared memory usage, which can reduce the number of resident blocks per SM and therefore lower occupancy. With 144 accumulators per thread however, register pressure alone may already limit the SM to a single resident block, so take this as an educated guess.

Also, as noted earlier, the T4 does not support `cp.async`, meaning loads and compute cannot truly overlap through dedicated hardware. Without that capability, the overlap we do get no longer outweighed the extra shared memory cost at these tile sizes.

**Runtime: 123.21 ms percentile 97.0**

<h2>Optimization 7: Strided Thread Layout</h2>

Now we have the reads coalesced we want to do the same with the writes.

With the contiguous layout, each thread computes a contiguous block of outputs (given TM=TN=12):

```
Thread (0,0) computes: C[0:12, 0:12]
Thread (1,0) computes: C[0:12, 12:24]
// Thread (0,0) handles outputs at (0,0), (0,1), (0,2), ..., (11,10), (11,11)
// Thread (1,0) handles outputs at (0,12), (0,13), (0,14), ..., (11,22), (11,23)

```

When writing results, thread 0 writes to column 0, thread 1 writes to column 12. These addresses are far apart, resulting in scattered writes.

The “strided thread layout” change is not a separate store routine — it’s a change in how tx maps to column indices. That mapping affects both (1) which columns each thread reads from shared B inside the k-loop, and (2) which columns each thread writes to global C at the end. The win comes from making the warp’s accesses contiguous.


The strided layout changes the mapping so threads handle outputs spaced a block dimension (16) apart:

```cuda
#define BM 192
#define BN 192
#define BK 16
#define TM 12
#define TN 12

int row = threadIdx.y + BM * blockIdx.y;
int col = threadIdx.x + BN * blockIdx.x;

// Thread (0,0) handles outputs at (0,0), (0,16), (0,32), ..., (16,0), (16,16), ...
// Thread (1,0) handles outputs at (0,1), (0,17), (0,33), ..., (16,1), (16,17), ...
```
Now when writing row 0, thread 0 writes column 0, thread 1 writes column 1, thread 2 writes column 2. Adjacent threads write adjacent addresses.

{:refdef: style="text-align: center;"}
![Strided Thread Layout]({{ "/images/post_7/strided_layout.png"}}){: width="75%" }
{: refdef}

**Runtime: 111.75 ms percentile 98.7**

<h2>Optimization 8: Compiler optimizations</h2>

Two final compiler-level tweaks helped push the kernel into the top 1%:

**`__restrict__`**: By marking input pointers as `__restrict__`, we tell the compiler that these pointers do not alias. This removes the need for conservative reloads and allows the compiler to keep values in registers across loop iterations. In a compute-heavy kernel like GEMM, this directly reduces redundant global memory traffic and enables more aggressive instruction scheduling.

```cuda
__global__ void matmul(const float* __restrict__ A,
                       const float* __restrict__ B,
                       float* __restrict__ C, ...) {
```

**`#pragma unroll`**: The innermost loops have a small, compile-time-known trip count (e.g. BK, TM, TN). Forcing unrolling eliminates loop control overhead and, more importantly, exposes independent instructions to the compiler. This increases instruction-level parallelism, improves register reuse, and gives the scheduler more freedom to overlap arithmetic with memory operations.

```cuda
#pragma unroll
for (int k = 0; k < BK; k++) {
    #pragma unroll
    for (int i = 0; i < TM; i++)
        #pragma unroll
        for (int j = 0; j < TN; j++)
            acc[i][j] += a_frag[i] * b_frag[j];
}
```

**Runtime: 108.89 ms percentile 99.2**

<h2>Things That Didn't Help</h2>

Not everything I tried improved performance:

- **float4 vectorized loads**: My attempt crashed on what I suspect was an alignment or indexing issue I didn't chase down. Since the final configuration ended up at BK=16 anyway, this is worth revisiting.
- **Double buffering with large tiles** (see Optimization 6): The added shared memory pressure may have reduced occupancy, and without `cp.async` on T4 the extra complexity didn’t translate into meaningful overlap.
- **Bank conflict padding**: Consistently increased runtime by ~10%, suggesting bank conflicts were not the dominant bottleneck and the padding likely hurt locality or indexing efficiency instead.

<h2>Summary</h2>

{:refdef: style="text-align: center;"}
![Runtime and percentile improvements across optimization steps]({{ "/images/post_7/opt_journey.png"}})
{: refdef}

Coming back to the target we set at the start, the final kernel runs in 108.89 ms, which works out to roughly 47% of the T4's peak FP32 throughput. cuBLAS, at 85-95% of peak, would finish the same problem in 54-60 ms. So while we made it into the top percentile of submissions, there is still a 2x gap to a fully tuned library, which would require techniques like vectorized loads and warp-level tiling to close.

The biggest lesson from this exercise was that GPU performance is largely about how data moves through the memory hierarchy. Most meaningful speedups came not from reducing computation, but from improving how data is reused, accessed, and kept close to the compute units.

Looking back, almost every optimization fell into one of three categories:

1. **Increasing data reuse** (tiling and register blocking)
2. **Improving memory access efficiency** (coalesced loads and stores)
3. **Hiding latency** (balancing tile sizes and occupancy)

The strided thread layout was particularly counterintuitive at first, but it nicely illustrates how small changes in how work is mapped to threads can have large effects on memory efficiency.

<h2>Extra: Transfer to newer GPUs</h2>
As the T4 is ancient, let's try this exact kernel on the H100 and B200. As an illustration the specs compared to a T4:

| Spec | T4 | H100 SXM | B200 |
|------|-----|----------|------|
| Architecture | Turing (2018) | Hopper (2022) | Blackwell (2024) |
| FP32 TFLOPS | 8.1 | 67 | ~80 |
| Memory | 16 GB GDDR6 | 80 GB HBM3 | 192 GB HBM3e |
| Memory BW | 320 GB/s | 3.35 TB/s | 8 TB/s |
| SMs | 40 | 132 | 148 |

Running the exact same kernel gives us the **88.7th** percentile on the H100 and **81.5th** on the B200, with runtimes of **14.45 ms** and **12.48 ms** respectively, roughly an **8x** speedup over the T4. This is one of the nice things about CUDA: the programming model abstracts how work is scheduled and executed, so the same kernel automatically benefits from more SMs, wider memory paths, and better schedulers. Of course, the kernel is nowhere near optimal on these GPUs, but it highlights how well CUDA’s abstraction layer holds up across generations.

The percentile drop also makes sense. Tile parameters tuned for the T4 are not ideal here. The block tiles (BM, BN) and the per-thread tiles (TM, TN) together trade off arithmetic intensity (how much math per byte loaded), register pressure and occupancy (how many warps can stay resident). The T4 and H100/B200 sit in very different regimes, so the optimal balance shifts. Decreasing BK to 8 on the newer GPUs already recovers some performance, bringing the H100 to **91.9** and the B200 to **86.3**. Note that BK cancels out of the block-level arithmetic intensity entirely (both the FLOPs and the bytes per step scale with it), so the gain can't come from better data reuse on paper. It points to an execution or resource effect instead, the smaller fully unrolled inner loop for example. Another reminder that the roofline model is not the whole story.

To fully utilise these architectures you would want to use their native features such as TMA for hardware-accelerated tile loads, newer tensor core instructions, and larger shared memory capacities. I'd recommend reading [Modular's Blackwell matmul series](https://www.modular.com/blog/matrix-multiplication-on-nvidias-blackwell-part-1-introduction) and [this worklog on outperforming cuBLAS on the H100](https://cudaforfun.substack.com/p/outperforming-cublas-on-h100-a-worklog) for more information on this.

<h2>Resources</h2>
- [How to Optimize a CUDA Matmul Kernel for cuBLAS-like Performance](https://siboehm.com/articles/22/CUDA-MMM) - Simon Boehm
- [Matrix Multiplication on GPU](https://www.aleksagordic.com/blog/matmul) - Aleksa Gordic
- [Matrix Multiplication on NVIDIA's Blackwell](https://www.modular.com/blog/matrix-multiplication-on-nvidias-blackwell-part-1-introduction) - Modular
- [Outperforming cuBLAS on H100, a worklog](https://cudaforfun.substack.com/p/outperforming-cublas-on-h100-a-worklog) - Pranjal Shankhdhar
- [CUDA resources I found helpful](https://github.com/PhilipBotros/cudafun/tree/main/resources)