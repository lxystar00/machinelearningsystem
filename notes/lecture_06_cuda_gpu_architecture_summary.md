# Lecture 06 — GPU Architecture & CUDA Programming

**Source:** `06-CUDA-programming.pdf`  
**Course:** 15-442/15-642 Machine Learning Systems  
**Lecture topics:** CUDA programming abstractions, GPU architecture, matrix multiplication, parallel reduction

> This note summarizes the lecture itself. Extra explanations that were useful during study are explicitly marked as **Clarification**.

---

# 0. Progress

The deck contains **69 slides**.

We have covered essentially the entire lecture. The only part that should still receive one precise step-by-step pass is the final reduction optimization sequence:

- Slide 65: Coalesced Memory Access
- Slide 66: Interleaved Addressing — suboptimal memory accesses
- Slide 67: Sequential Addressing — fully coalesced memory access
- Slide 68: Code transformation to sequential addressing
- Slide 69: Recap

The rest of the lecture is summarized below.

---

# 1. SIMD: Why GPUs Have So Much Parallel Compute

## 1.1 Single Instruction, Multiple Data

**SIMD = Single Instruction, Multiple Data**

The basic idea:

```text
one instruction
      ↓
ALU  ALU  ALU  ALU ...
 ↓    ↓    ↓    ↓
x0   x1   x2   x3
```

The same operation is applied to multiple pieces of data in parallel.

Example:

```text
A = [1, 2, 3, 4]

same instruction: ×2

→ [2, 4, 6, 8]
```

### Key terms

- **Instruction**: a small hardware operation such as add, multiply, load, or store.
- **ALU**: Arithmetic Logic Unit; hardware that performs arithmetic/logic.
- **Fetch / Decode**: read an instruction and determine what operation it represents.

## 1.2 Why add more ALUs?

If the same instruction is shared across many ALUs, adding more ALUs can increase the number of data elements processed at once.

This mainly improves **throughput**:

```text
latency
= how long one operation takes

throughput
= how many operations finish per unit time
```

A GPU is primarily designed for high throughput on massively parallel workloads.

---

# 2. CPU vs GPU Design Philosophy

A simplified mental model:

```text
CPU:
few powerful cores
+ large caches
+ sophisticated control logic

Goal:
low latency for a small number of instruction streams
```

```text
GPU:
many arithmetic execution units
+ massive parallelism
+ comparatively less control per arithmetic lane

Goal:
high throughput
```

This is why workloads such as:

```python
for i in range(10_000_000):
    C[i] = A[i] + B[i]
```

are naturally GPU-friendly.

---

# 3. CUDA Programming Abstraction

CUDA exposes a hierarchy:

```text
Grid
 └── Blocks
      └── Threads
```

A kernel is launched as:

```cpp
kernel<<<grid, block>>>(arguments);
```

where:

```text
grid  = number / arrangement of blocks
block = number / arrangement of threads per block
```

---

# 4. Important CUDA Built-ins

For each CUDA thread:

```cpp
threadIdx
```

is its local index inside its block.

```cpp
blockIdx
```

is the block's index inside the grid.

```cpp
blockDim
```

is the size of a block.

```cpp
gridDim
```

is the size of the grid.

Each has:

```cpp
.x
.y
.z
```

for up to 3D organization.

---

# 5. Global Thread Index

For a 1D array:

```cpp
int i =
    blockIdx.x * blockDim.x
    + threadIdx.x;
```

Interpretation:

```text
global index
=
threads before my block
+
my position inside my block
```

For a 2D matrix:

```cpp
int col =
    blockIdx.x * blockDim.x
    + threadIdx.x;

int row =
    blockIdx.y * blockDim.y
    + threadIdx.y;
```

---

# 6. Bounds Checking

The number of launched CUDA threads does not have to exactly equal the amount of data.

Example:

```text
N = 1000
threads/block = 256

blocks = ceil(1000 / 256) = 4

launched threads = 1024
```

So the kernel needs:

```cpp
if (i < N) {
    ...
}
```

For 2D:

```cpp
if (row < rows && col < cols) {
    ...
}
```

This is normal CUDA programming.

---

# 7. Host vs Device Code

CUDA separates CPU and GPU execution.

```text
Host   = CPU
Device = GPU
```

A kernel:

```cpp
__global__ void kernel(...) {
    ...
}
```

runs on the GPU and is launched from host code.

A GPU helper function can be:

```cpp
__device__ float f(float x) {
    ...
}
```

Simplified:

| Function | Runs on | Typical caller |
|---|---|---|
| normal C++ function | CPU | CPU |
| `__device__` | GPU | GPU |
| `__global__` | GPU | host kernel launch |

---

# 8. Conditional Execution and Divergence

Suppose:

```cpp
if (x > 0) {
    x = 2 * x;
} else {
    x = ...;
}
```

Threads executing together may disagree:

```text
T T F T F F F F
```

GPU execution uses masks so that one path executes with only the relevant lanes active, then the other path executes.

Conceptually:

```text
true path:
✓ ✓ ✗ ✓ ✗ ✗ ✗ ✗

false path:
✗ ✗ ✓ ✗ ✓ ✓ ✓ ✓
```

Correctness is preserved, but execution resources can be underutilized.

This is **divergent execution**.

If all threads in an execution group follow the same path, execution is **coherent**.

---

# 9. Warps

On NVIDIA GPUs:

```text
1 warp = 32 CUDA threads
```

A block of 128 threads contains:

```text
128 / 32 = 4 warps
```

Example:

```text
Warp 0: threads 0–31
Warp 1: threads 32–63
Warp 2: threads 64–95
Warp 3: threads 96–127
```

Branch divergence matters primarily **within a warp**.

Good:

```text
Warp 0: T T T T ... T
Warp 1: F F F F ... F
```

Bad:

```text
Warp 0: T F T F F T ...
```

---

# 10. CUDA Memory Model

The lecture distinguishes:

```text
Host memory
Device memory
```

and, inside device code:

```text
per-thread private memory
per-block shared memory
device global memory
```

## 10.1 Scope

```text
private
→ one thread

shared
→ all threads in one block

global
→ all CUDA threads
```

## 10.2 Important terminology clarification

The slide says **per-thread private memory**.

Do not automatically interpret every thread-private variable as formal CUDA **local memory**.

A scalar such as:

```cpp
float x;
```

is ideally stored in a register.

---

# 11. Host ↔ Device Data Transfer

Typical allocation:

```cpp
float* deviceA;
cudaMalloc(&deviceA, bytes);
```

Copy host → device:

```cpp
cudaMemcpy(
    deviceA,
    A,
    bytes,
    cudaMemcpyHostToDevice
);
```

Copy device → host:

```cpp
cudaMemcpy(
    A,
    deviceA,
    bytes,
    cudaMemcpyDeviceToHost
);
```

Classic mental model:

```text
CPU RAM
   │
   │ cudaMemcpy
   ▼
GPU global memory
```

---

# 12. Why Shared Memory Exists

Shared memory enables threads within a block to cooperate.

Main use:

```text
Global memory
     ↓
load reusable data once
     ↓
Shared memory
     ↓
reuse by many threads
```

This reduces repeated expensive global-memory accesses.

---

# 13. 1D Convolution Example

The lecture uses:

\[
output[i]
=
\frac{
input[i]
+
input[i+1]
+
input[i+2]
}{3}
\]

Naive implementation:

```text
128 threads
×
3 global loads/thread
=
384 global loads
```

But neighboring outputs reuse most input values.

Example:

```text
output[0] needs 0,1,2
output[1] needs 1,2,3
output[2] needs 2,3,4
```

---

# 14. Convolution with Shared Memory

The block allocates:

```cpp
__shared__ float support[THREADS_PER_BLK + 2];
```

With:

```text
THREADS_PER_BLK = 128
```

the shared-memory region contains:

```text
128 main values
+
2 halo values
=
130 values
```

Threads cooperatively load these values.

Global loads become approximately:

```text
130
```

instead of:

```text
384
```

Then computation repeatedly reads shared memory.

This is the central pattern:

```text
global memory
     ↓
cooperative load
     ↓
shared memory
     ↓
synchronize
     ↓
reuse
```

---

# 15. `__syncthreads()`

```cpp
__syncthreads();
```

is a block-level barrier.

It means:

> all threads in the block must reach this barrier before any thread proceeds beyond it.

Typical shared-memory pattern:

```cpp
shared[threadIdx.x] = input[...];

__syncthreads();

// safe to consume data written by other threads
```

It does **not** synchronize arbitrary blocks across the grid.

---

# 16. Atomics

The lecture mentions atomic operations such as:

```cpp
atomicAdd(addr, amount);
```

They are useful when many threads update the same location.

Without atomic behavior:

```text
Thread A reads 5
Thread B reads 5

A writes 6
B writes 6

result = 6
```

but logically two increments should produce:

```text
7
```

---

# 17. GPU Block Scheduling

A major CUDA assumption:

> Thread blocks can execute in any order and should not depend on block execution order.

The GPU dynamically schedules blocks onto available hardware.

So:

```text
Block 0
```

does not necessarily execute before:

```text
Block 1
```

This allows one CUDA program to scale to GPUs with different numbers of SMs.

---

# 18. Streaming Multiprocessor (SM)

An NVIDIA GPU contains multiple **Streaming Multiprocessors (SMs)**.

An SM provides finite resources such as:

```text
warp/thread contexts
registers
shared memory
execution units
warp schedulers
```

A block is assigned to one SM.

Multiple blocks can be resident on the same SM if resources permit.

---

# 19. Resident Blocks and Resource Limits

Each block consumes resources.

Example from the lecture:

```text
128 threads / block
520 bytes shared memory / block
```

An SM can only admit as many blocks as its resource limits allow.

Possible limiting resources:

```text
threads / warps
shared memory
registers
maximum blocks per SM
```

The key principle:

```text
resident blocks per SM
=
limited by whichever hardware resource runs out first
```

When a block finishes, its resources are released and another waiting block can be admitted.

---

# 20. Resident Warps and Latency Hiding

An SM may keep many warps resident.

Example:

```text
Warp 0 → waiting for memory
Warp 1 → ready
Warp 2 → ready
Warp 3 → waiting
```

Instead of waiting for Warp 0:

```text
execute Warp 1
execute Warp 2
...
```

This is **latency hiding**.

Important:

```text
resident
≠
executing arithmetic at exactly the same instant
```

It means the warp has its state/resources available on the SM and can be scheduled.

---

# 21. Block Scheduling vs Warp Scheduling

Two levels:

```text
Grid
 ↓
block scheduler
 ↓
SM
 ↓
resident blocks
 ↓
warps
 ↓
warp scheduler
 ↓
execution
```

- **Block scheduler**: which SM receives a block?
- **Warp scheduler**: which ready warp on an SM executes next?

---

# 22. Tensor Cores

Modern NVIDIA SMs contain specialized **Tensor Cores**.

Conceptually they accelerate matrix multiply-accumulate:

\[
D = A B + C
\]

rather than only individual scalar arithmetic.

They are especially valuable for ML because many operations are dominated by matrix multiplication:

```text
Linear layers
Attention
MLPs
Convolution implementations
```

---

# 23. Lower Precision and Tensor Core Throughput

Formats such as:

```text
FP16
BF16
FP8
TF32
```

can enable much higher matrix throughput.

Two useful reasons:

## 23.1 Less memory traffic

Typical sizes:

```text
FP32 = 4 bytes
FP16 = 2 bytes
```

Moving the same number of FP16 values requires less bandwidth.

## 23.2 More arithmetic parallelism

Smaller-number arithmetic can be implemented with less hardware area, allowing specialized hardware to process more low-precision operations per cycle.

But lower precision loses numerical information.

So the benefit is available only when the workload can tolerate the reduced precision.

A common mixed-precision idea is:

```text
low-precision multiply
+
higher-precision accumulation
```

---

# 24. Case Study 1 — Naive Matrix Multiplication

For:

\[
C = AB
\]

one output element is:

\[
C[x][y]
=
\sum_{k=0}^{N-1}
A[x][k] B[k][y]
\]

Naive CUDA strategy:

```text
1 thread
→ 1 element of C
```

Kernel structure:

```cpp
float result = 0;

for (int k = 0; k < N; ++k) {
    result += A[x][k] * B[k][y];
}

C[x][y] = result;
```

---

# 25. Naive Matmul Memory Cost

Per output element:

```text
N loads from A
+
N loads from B
=
2N global loads
```

There are:

\[
N^2
\]

outputs.

Therefore:

\[
\boxed{2N^3}
\]

global-memory accesses in the lecture's simplified analysis.

The problem:

> the same A/B values are repeatedly fetched by different threads.

---

# 26. Optimization 1 — Thread-Level Register Tiling

Instead of:

```text
1 thread → 1 output
```

use:

```text
1 thread → V × V output tile
```

A thread loads:

```text
V values from A
+
V values from B
```

then reuses them to update:

\[
V^2
\]

output accumulators.

The output tile is kept in thread-private variables/registers.

Simplified global-memory cost:

\[
\boxed{\frac{2N^3}{V}}
\]

So reuse increases by roughly a factor of \(V\).

---

# 27. Optimization 2 — Block-Level Shared-Memory Tiling

Next level:

```text
one block
→ computes L × L output tile

one thread
→ computes V × V part of that tile
```

A block loads tiles of A and B into shared memory.

Then all threads reuse them.

Data hierarchy:

```text
GLOBAL MEMORY
      ↓
shared-memory tiles
      ↓
thread registers
      ↓
multiply-add operations
```

This gives two reuse levels:

```text
shared memory
→ reuse across threads in the block

registers
→ reuse within one thread
```

---

# 28. `S`, `L`, and `V` in Matmul Tiling

The lecture uses:

```text
L
= block-level output tile dimension

V
= thread-level output tile dimension

S
= chunk size along the inner K dimension
```

One block repeatedly performs:

```text
A tile: L × S
×
B tile: S × L
↓
partial update to
C tile: L × L
```

across all `K` chunks.

---

# 29. Shared-Memory Tiling Memory Cost

Per block:

\[
2LN
\]

global loads.

Number of blocks:

\[
\frac{N^2}{L^2}
\]

Total:

\[
\boxed{\frac{2N^3}{L}}
\]

global-memory accesses.

The lecture also gives:

\[
\boxed{\frac{2N^3}{V}}
\]

shared-memory accesses.

The goal is not to eliminate memory access entirely.

The goal is to replace expensive repeated global-memory access with faster on-chip reuse.

---

# 30. Cooperative Fetching

A tile may contain more values than there are threads.

So all block threads divide the tile-loading work.

Flatten a 2D thread ID:

```cpp
int tid =
    threadIdx.y * blockDim.x
    + threadIdx.x;
```

Then distribute flat tile elements:

```cpp
idx = j * nthreads + tid;
```

Convert back to 2D:

```cpp
row = idx / L;
col = idx % L;
```

This is **cooperative fetching**.

---

# 31. Case Study 2 — Parallel Reduction

Reduction maps many values to fewer values, often one:

```text
[3,1,7,0,4,1,6,3]
        ↓
        25
```

Examples:

```text
sum
max
min
```

Reductions are important inside operations such as normalization and softmax.

A tree reduction looks like:

```text
8 values
→ 4
→ 2
→ 1
```

with dependency depth approximately:

\[
O(\log N)
\]

instead of a serial \(O(N)\) dependency chain.

---

# 32. Why Reduction Needs Multiple Blocks

For a very large array:

```text
Block 0 → partial result
Block 1 → partial result
Block 2 → partial result
...
```

Using many blocks keeps multiple SMs busy.

The challenge:

> How do we combine the partial results?

---

# 33. No Normal Grid-Wide `__syncthreads()`

`__syncthreads()` only synchronizes one block.

Blocks may execute in arbitrary order, so a general block-wide barrier inside an ordinary kernel could deadlock if waiting blocks occupy all available resident slots.

The lecture solution is:

```text
Kernel 1
→ produce partial sums

Kernel 2
→ reduce partial sums
```

Kernel decomposition provides synchronization between dependent stages.

---

# 34. Reduction Version 1 — Interleaved Addressing

Initial pattern:

```cpp
for (unsigned int s = 1;
     s < blockDim.x;
     s *= 2) {

    if (tid % (2*s) == 0) {
        sdata[tid] += sdata[tid + s];
    }

    __syncthreads();
}
```

Active threads become:

```text
s=1:
T F T F T F T F ...

s=2:
T F F F T F F F ...

s=4:
T F F F F F F F ...
```

This causes highly divergent warps.

---

# 35. Why Two Synchronizations Are Needed

First:

```cpp
sdata[tid] = g_idata[i];

__syncthreads();
```

ensures all initial shared-memory values are ready.

Then each reduction stage needs:

```cpp
__syncthreads();
```

because the next stage depends on values produced by the current stage.

---

# 36. Reduction Version 2 — Non-Divergent Active Threads

Replace:

```cpp
tid % (2*s) == 0
```

with:

```cpp
int index = 2 * s * threadIdx.x;

if (index < blockDim.x) {
    sdata[index] += sdata[index + s];
}
```

Now active thread IDs are contiguous:

```text
s=1:
T T T T T T T T F F ...

s=2:
T T T T F F F F ...

s=4:
T T F F ...
```

This avoids the scattered active-thread pattern of Version 1.

---

# 37. Coalesced Memory Access

This is the important final optimization topic of the source deck.

**Coalesced access** means neighboring GPU threads access neighboring memory addresses.

Good pattern:

```text
thread 0 → A[0]
thread 1 → A[1]
thread 2 → A[2]
thread 3 → A[3]
...
```

This lets the hardware efficiently combine requests into memory transactions.

Poor pattern:

```text
thread 0 → A[0]
thread 1 → A[large stride]
thread 2 → A[another stride]
...
```

This wastes memory-transaction capacity.

The slide summarizes:

> multiple GPU threads access consecutive memory addresses → maximize GPU memory usage.

---

# 38. Sequential Addressing in Reduction

The final reduction form reverses the loop:

```cpp
for (
    unsigned int s = blockDim.x / 2;
    s > 0;
    s /= 2
) {
    if (threadIdx.x < s) {
        sdata[threadIdx.x]
            += sdata[threadIdx.x + s];
    }

    __syncthreads();
}
```

Example with 8 values:

```text
s = 4

T0: sdata[0] += sdata[4]
T1: sdata[1] += sdata[5]
T2: sdata[2] += sdata[6]
T3: sdata[3] += sdata[7]
```

then:

```text
s = 2

T0: sdata[0] += sdata[2]
T1: sdata[1] += sdata[3]
```

then:

```text
s = 1

T0: sdata[0] += sdata[1]
```

This keeps active threads contiguous and produces a more favorable memory-access pattern.

---

# 39. Coalescing vs Shared-Memory Bank Conflicts

These are related performance ideas, but they are **not the same concept**.

## Coalescing

Primarily think:

```text
global memory
```

Question:

> Are neighboring warp threads accessing neighboring addresses so their requests can be served efficiently?

## Shared-memory bank conflicts

Think:

```text
shared memory
```

Question:

> Are threads accessing shared-memory addresses that map to conflicting banks?

The final slides 65–68 of this lecture specifically emphasize **coalesced memory access and sequential addressing**.

The final recap also lists **shared memory bank conflict** as an optimization topic, but the deck does not develop it in the same step-by-step detail as the coalescing sequence.

---

# 40. Final Lecture Recap

The source's final recap lists:

## CUDA programming and GPU architecture

You should understand:

```text
Grid
→ Block
→ Thread

GPU
→ SM
→ Warp
→ Threads
```

## GPU optimization techniques

The lecture highlights:

```text
coherent warps
coalesced memory access
shared-memory bank conflict
warp-level optimizations
Tensor Cores
```

---

# 41. Most Important Mental Model

A performant GPU kernel is largely about keeping expensive hardware busy while minimizing expensive data movement.

```text
GPU global memory
        ↓
coalesced / efficient loads
        ↓
shared memory
        ↓
reuse across threads
        ↓
registers
        ↓
reuse inside each thread
        ↓
ALUs / Tensor Cores
```

Meanwhile:

```text
many resident warps
        ↓
warp scheduler
        ↓
hide memory/instruction latency
```

And control flow should ideally maintain:

```text
threads in one warp
→ coherent execution
```

---

# 42. The Four Questions to Ask When Reading a CUDA Kernel

When you see a CUDA kernel, ask:

## 1. Parallel mapping

```text
What does one thread compute?
What does one block compute?
```

## 2. Memory movement

```text
What comes from global memory?
What is placed in shared memory?
What stays in registers?
```

## 3. Reuse

```text
How many times is each loaded value reused?
```

## 4. Hardware utilization

```text
Are warps divergent?
Are memory accesses coalesced?
Are register/shared-memory requirements limiting residency?
Can Tensor Cores be used?
```

These questions capture most of the lecture's performance reasoning.

---

# 43. Very Short Cheat Sheet

```text
SIMD
= same instruction, different data
```

```text
CUDA hierarchy
= Grid → Block → Thread
```

```text
NVIDIA execution
= GPU → SM → Warp → 32 threads
```

```text
global thread id
= blockIdx.x * blockDim.x + threadIdx.x
```

```text
private
= one thread

shared
= one block

global
= whole device
```

```text
__syncthreads()
= block-level barrier
```

```text
warp divergence
= threads in one warp take different control-flow paths
```

```text
latency hiding
= schedule another ready warp while one warp waits
```

```text
matmul optimization
= global memory → shared tile → registers → compute
```

```text
reduction
= many values → fewer values using a tree
```

```text
coalescing
= neighboring threads access neighboring memory addresses
```

```text
Tensor Core
≈ specialized matrix multiply-accumulate hardware
```

---

# 44. What to Learn Next

After this lecture, the natural next topics are:

1. **Arithmetic intensity / roofline thinking**
2. **Memory bandwidth vs compute-bound kernels**
3. **Triton**
4. **Kernel fusion**
5. **FlashAttention**
6. **Modern GEMM kernels**
7. **Profiling with Nsight**
8. **PyTorch / `torch.compile` relationship to CUDA kernels**

These build directly on the concepts in this deck.
