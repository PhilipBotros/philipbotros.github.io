---
layout: post
title: "How to get to the top percentile of a mat mul kernel (worklog)"
date: 2026-01-08
cover: /images/post_6/cover.jpg
background: /images/post_6/bg.jpg
mathjax: true
excerpt: <br> A look at the CPython execution model and showing two ways to run CUDA from Python, using ctypes for raw access and PyTorch custom operators for deeper integration.
---
This post documents my journey optimizing a CUDA matrix multiplication kernel, starting from a naive tiled implementation and ending up in the top 1% of performance on a T4 GPU on LeetGPU (LINK).
{:refdef: style="text-align: center;"}
![Matmul Optimization Journey]({{ "/images/post_matmul/optimization_journey.png"}})
{: refdef}

Add picture of percentile stuff here,

There's loads of highly illustrative posts about the internals of GPUs, see (1) and (2).
To recap, at a high level, fast GPU kernels are built by keeping data local, maximizing arithmetic intensity, and exposing enough parallelism to hide memory latency.

To give a measure of the differences:
Global Memory: (Huge capacity, but 400–800 cycles away).

Shared Memory: (Small, but only ~20–30 cycles away).

Registers: (Tiny, but ~1 cycle away).

Maybe a better illustration above (flash-attention) ?

<h2>Starting Point: Fully Naive Implementation</h2>

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

Each thread computes one output element of the matrix C by loading an entire row from A and an entire column from B from global memory. If we assume they're both square with a size of N, this means every active thread loads 4N bytes from A and 4N bytes from B given FP32 inputs (ignoring the write back to HBM FIX THIS). Every thread then does N multiplications and N additions. A good way of measuring algorithmic performance is arithmetic intensity. This translates to the number of FLOPs performed for every byte loaded from global memory. The higher this number the more we are bottlenecked by compute speed versus HBM bandwidth. The formula for this is:

$$ AI = \frac{\text{FLOPs}}{\text{Bytes Transferred}} $$

Arithmetic Intensity:
2N FLOPs / 8N bytes = 0.25 FLOPs/byte. This is extremely low as for every FLOP we now have to load 4 bytes! The T4 GPU has a peak bandwidth of ~320 GB/s and peak compute of ~8.1 TFLOPS. At 0.25
FLOPs/byte, you'd need 32.4 TB/s of memory bandwidth to reach peak compute so you're memory-bound by a factor of ~100×.

This is a lot as every load comes from slow global memory, and there's no data reuse across threads. If two threads need the same element from A or B, they both load it separately. (REWRITE)

**Runtime: 956.41 ms percentile 16.9**

Unsurprisingly we are in the bottom quartile of performance given this naive implementation, time to do better!

<h2>Optimization 1: Tiled Matmul with Shared Memory</h2>
The starting point to reduce global memory traffic is a textbook tiled matrix multiplication. We divide the problem into tiles and sequentially load tiles into shared memory and compute partial accumulations until the full matrices have been processed. At every tile step, each thread loads one element into shared memory, we synchronize, then each thread computes one (partial) output element.

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
The win here over the fully naive implementation is that we're reusing data across threads via shared memory. Without tiling, each thread would load an entire row of A and column of B from global memory. With tiling, we cooperatively load tiles into shared memory and reuse these across the block.

Each thread still computes a single output element of $C$ and therefore performs approximately $2N$ FLOPs. The key difference is that global memory loads are now amortized across the block. At each tile iteration, every active thread loads one FP32 value from $A$ and one from $B$, for a total of $8$ bytes from global memory. Since the dot product spans $\lceil N / T \rceil$ tiles, each thread loads $8 \times \lceil N / T \rceil$ bytes in total. This results in a per-thread arithmetic intensity of $$\frac{2N}{8 \lceil N / T \rceil} \approx \frac{T}{4}.$$ For a tile size of $T = 16$, this gives an arithmetic intensity of $4.0$ FLOPs per byte. This is a 16x increase over the naive kernel!

An interesting observation is that pushing the tile size to the maximum supported by a single block (1024 threads, i.e. T = 32) doubles the arithmetic intensity to 8.0 FLOPs/byte, yet performance on a T4 actually decreases. This highlights an important caveat of roofline-style reasoning: increasing arithmetic intensity does not automatically translate to higher throughput.

In this regime, the kernel is no longer limited by global memory bandwidth. Instead, hardware characteristics begin to dominate: large blocks reduce scheduling flexibility, often limiting the SM to a single resident block, which in turn reduces the total number of active warps available to hide latency. Additionally, larger tiles typically increase register pressure and synchronization cost, further constraining occupancy. The net effect is that, despite improved data reuse, the GPU is less able to keep its execution pipelines busy.

But there's still a problem: each thread only computes one output, and we're only getting one FMA (fused multiply-add) per two shared memory loads. We can do better.

**Runtime: 593.53 ms percentile 76.1**

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

We now have to set up thread specific accumulator tiles (in registers) and we can do a form of cooperative loading(BOLD). Indexing becomes slightly more involved as we now have another dimension to account for (micro-tile).  


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

    __shared__ float tile_A[BM][BK]; // 64x8
    __shared__ float tile_B[BK][BN]; // 8x64

    float acc[TM][TN];
    for (int i = 0; i < TM; i++) {
        for (int j = 0; j < TN; j++) {
            acc[i][j] = 0.0f;
        }
    }

    int num_tiles = (N + BK - 1) / BK;

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
I want to zoom in on the loading pattern, we now have 16x16=256 threads in a block needing to load 64x8 values for A and 8x64 for B = 1024. Previously every active thread was simply loading a single element of both tiles like so, as the tiles were the size of the block:
```cuda
// Each thread loads one element
tile_A[threadIdx.y][threadIdx.x] = A[row * N + t * TILE_SIZE + threadIdx.x];
tile_B[threadIdx.y][threadIdx.x] = B[(t * TILE_SIZE + threadIdx.y) * K + col];
__syncthreads();
```
Since every thread now has multiple elements to load we have a choice to make inside our kernel on how to design this.
We could theoretically have every thread load 2 values from A and 2 values from B but the indexing will become quite cumbersome.

Alternatively, we can have only a subset of the threads participate (i.e. tx < BK), which keeps the mapping simple: tx directly indexes the BK columns and each participating thread loads a short vertical strip of TM elements. This preserves coalesced global loads and reduces the amount of index arithmetic and branching in the load path.

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

After loading the tile like before, we now load fragments into the registers first.


The magic is in the outer product. We load 4 values from A and 4 values from B (8 loads total), then perform 16 FMAs. Each a_frag[i] is used 4 times, each b_frag[j] is used 4 times. Compare this to the naive version: 2 loads → 1 FMA. Now we have 8 loads → 16 FMAs.

Outside of adding micro-tiling we needed to decouple the block reduction dimension and add cooperative loading. The good thing is this was the largest refactor and the skeleton of the kernel stays roughly the same from now.
<!-- 
The third kernel uses a larger block tile ($64 \times 64$) and further optimizes by having each thread compute multiple results ($4 \times 4 = 16$ outputs) using registers. This increases data reuse at two levels: the block level (Global $\rightarrow$ Shared) and the thread level (Shared $\rightarrow$ Registers). -->

Every thread now does $2N \times TM \times TN$ FLOPs while loading the same amount of bytes as the previous kernel: $8 \times \lceil N / T \rceil$. 

Global Memory Intensity (The Roofline AI)The intensity relative to global memory depends on the Block Tile size ($BM \times BN$).FLOPs per block iteration: $BM \times BN \times BK \times 2 = 64 \times 64 \times 8 \times 2 = 65,536$.Bytes per block iteration: $(BM \times BK + BN \times BK) \times 4 = (64 \times 8 + 64 \times 8) \times 4 = 4,096$ Calculation: $\frac{64 \times 64}{2(64 + 64)} = \frac{4096}{256} = 16.0$.

AI stays the same!

**Runtime: 194.28 ms percentile 92.5**

<h2>Optimization 3: Coalesced Global Memory Loading</h2>
Now we have a decent arithmetic intensity and keep data local it's time to look at the memory traffic. My initial implementation had each thread load its own strip of data:

```cuda
// Bad: each thread loops over consecutive addresses
for (int i = 0; i < TM; i++) {
    tile_A[ty * TM + i][k] = A[row * N + k_offset + k];
}
```
The problem is what happens at any given moment. Thread 0 is loading address 0, thread 1 is loading address 64, thread 2 is loading address 128. The memory controller sees scattered addresses and issues separate transactions for each. The key thing to remember is that global memory coalescing is decided at the warp level, not per thread. In CUDA, a load instruction is executed by 32 threads in lockstep, and the hardware coalesces the 32 addresses requested by the warp into as few memory transactions as possible.

That means a pattern can look “nice” within each thread (each thread walks consecutive elements), yet still be slow if neighboring lanes access far-apart addresses at the same instruction.
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

Now at any given moment, thread 0 loads address 0, thread 1 loads address 1, thread 2 loads address 2. The memory controller coalesces these into a single wide transaction.

As you can see the speedup is very modest here. There was a larger amount of difference when increasing the tile sizes (96.3 vs 97.0).

**Runtime: 193.43 ms percentile 92.6**

<h2>Optimization 4: Double Buffering</h2>

The next optimization overlaps memory loading with computation. Without double buffering:

```
Load tile 0 → [stall] → Compute tile 0 → Load tile 1 → [stall] → Compute tile 1
```

With double buffering, we use two sets of shared memory and load the next tile while computing on the current one:

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
This gave a modest speedup by hiding some memory latency behind computation. This did not hold when the tile size was increased!
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

The T4 has 64 KB shared memory and 255 registers per thread. With 128×128 tiles and 8×8 thread tiles, we're using about 16 KB shared memory and ~100 registers per thread, both well within limits.

More work per thread means better register reuse and fewer synchronization points. Also note increasing the tile sizes is not free! We now can schedule less warps per SM as every warp takes up more register space, this can lead to a lower occupancy and reduce the kernels performance. We'll see an example of this at the end where we run our kernel on newer GPU architectures. Turns out we can crank it up even further:
```
#define BM 192
#define BN 192
#define BK 16
#define TM 12
#define TN 12
```
**Runtime: 136.86 ms percentile 95.7**

<h2>Optimization 6: Remove double buffering with increasing tile size</h2>
Tended to not work after we increased tile sizes. Again, this could be an occupancy problem. We now use double the amount of shared memory to both store the load and compute tile and the larger the tile the more additional memory double buffering consumes.

**Runtime: 123.21 ms percentile 97.0**

<h2>Optimization 7: Strided Thread Layout</h2>

Now we have the reads coalesced we want to do the same with the writes.

With the contiguous layout, each thread computes a contiguous block of outputs (given TM=TN=12):

```
Thread (0,0) computes: C[0:12, 0:12]
Thread (1,0) computes: C[0:12, 12:24]
// Thread (0,0) handles outputs at (0,0), (0,1), (0,2), ..., (12,0), (12,1), ...
// Thread (1,0) handles outputs at (0,12), (0,13), (0,14), ..., (12,12), (12,13), ...

```

When writing results, thread 0 writes to column 0, thread 1 writes to column 12. These addresses are far apart, resulting in scattered writes.

The “strided thread layout” change is not a separate store routine — it’s a change in how tx maps to column indices. That mapping affects both (1) which columns each thread reads from shared B inside the k-loop, and (2) which columns each thread writes to global C at the end. The win comes from making the warp’s accesses contiguous.


The strided layout changes the mapping so threads handle outputs spaced tile size apart:

```cuda
#define BM 192
#define BN 192
#define BK 16
#define TM 12
#define TN 12

int row = threadIdx.y + BM * blockIdx.y;
int col = threadIdx.x + BN * blockIdx.x;

// Thread (0,0) handles outputs at (0,0), (0,12), (0,24), ..., (12,0), (12,12), ...
// Thread (1,0) handles outputs at (0,1), (0,13), (0,25), ..., (12,1), (12,13), ...
```
Now when writing row 0, thread 0 writes column 0, thread 1 writes column 1, thread 2 writes column 2. Adjacent threads write adjacent addresses.

**Runtime: 111.75 ms percentile 98.7**

<h2>Optimization 8: Compiler optimizations</h2>

Two final compiler-level tweaks helped push the kernel into the top 1%:

	•	__restrict__:
By marking input pointers as __restrict__, we tell the compiler that these pointers do not alias. This removes the need for conservative reloads and allows the compiler to keep values in registers across loop iterations. In a compute-heavy kernel like GEMM, this directly reduces redundant global memory traffic and enables more aggressive instruction scheduling.

	•	#pragma unroll:
The innermost loops have a small, compile-time–known trip count (e.g. BK, TM, TN). Forcing unrolling eliminates loop control overhead and, more importantly, exposes independent instructions to the compiler. This increases instruction-level parallelism, improves register reuse, and gives the scheduler more freedom to overlap arithmetic with memory operations.

**Runtime: 108.89 ms percentile 99.2**

<h2>Things That Didn't Help</h2>

Not everything I tried improved performance:

- **float4 vectorized loads**: Alignment issues with BK=8 caused crashes. Would need BK=16 or careful padding.
- **Double buffering with large tiles**: The added complexity wasn't worth it for the final configuration.
- **Bank conflict padding**: Consistently reduced runtime with ~10%

<h2>Final Configuration</h2>

| Parameter | Value |
|-----------|-------|
| TILE_SIZE | 16 |
| BN | 12 |
| TN | 192 |
| Threads per block | 256 (16×16) |
| Outputs per thread | 144 (12×12) |
| Outputs per block | 36,864 (192×192) |
| Shared memory | 24 KB |
<h2>Summary</h2>

| Optimization | Key Insight | Impact |
|--------------|-------------|--------|
| Thread coarsening | Reuse registers across multiple outputs | Major |
| Coalesced loads | Adjacent threads access adjacent addresses | 3× speedup |
| Double buffering | Overlap loads with compute | Minor |
| Larger tiles | More work per sync, better reuse | Significant |
| Strided layout | Coalesced writes | Top 1% |

The biggest lesson: memory access patterns dominate GPU performance. Both coalesced reads and coalesced writes matter. The strided thread layout was counterintuitive at first, but it's the key to getting both right simultaneously.

How far from cuBLAS?
Do test on T4.

<h2>Transfer to newer GPUs</h2>
As the T4 is ancient, let's try this exact kernel on the H100 and B200. As an illustration the specs compared to a T4:

We now get the 91.6th percentile on the H100 and 82.8 on the B200, which is unsurprising as the specs are very different.

Interestingly, the speedup is already 8x as we go to 14.53 ms and 12.48 ms on the newer GPUs. It’s one of the beautiful things about CUDA: its programming model abstracts how work is scheduled and executed, so as hardware evolves, the same kernel can automatically benefit from more SMs, wider memory paths, and better schedulers. Of course, this doesn’t mean the kernel is anywhere near optimal on H100 or B200 (especially given tons of new features), but it highlights how well CUDA’s core abstraction layer has held up across generations.

Decreasing BN to 8 here works actually, why:

BN is trading off two things that scale very differently across T4 vs H100/B200:
	1.	How much math you do per byte you load (reuse / arithmetic intensity)
	2.	How many warps you can keep resident (occupancy / latency hiding), which is dominated by register pressure (and sometimes spills)

And those two GPUs sit in very different regimes. 

Ideally, you would use the latest features of this GPU to make the difference even larger. TMA, latest tensor cores etc.

<h2>Resources</h2>
SBOEHM (classic)
See my cuda repo -> link to it.


This is it now