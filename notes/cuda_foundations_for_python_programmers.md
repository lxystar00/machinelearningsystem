# CUDA Foundations for Python Programmers

This note summarizes the prerequisite material needed before reading a beginner CUDA programming lecture. It assumes you already know Python, so it focuses only on the C++ and CUDA concepts that are materially different from Python and that appear constantly in CUDA code.

---

# 1. C++ Variables, Types, Arrays, and Loops

If you know Python, most basic C++ control flow is familiar. The main difference is that C++ usually requires explicit types.

## 1.1 Variables and types

Python:

```python
x = 5
y = 3.2
```

C++:

```cpp
int x = 5;
float y = 3.2f;
double z = 3.1415926535;
```

Common numeric types:

```text
int      -> integer
float    -> 32-bit floating-point number
double   -> usually 64-bit floating-point number
```

In ML/CUDA code, `float` appears very often because tensors are commonly stored as floating-point values.

A C++ statement usually ends with:

```cpp
;
```

Example:

```cpp
int x = 5;
```

means:

> Create an integer variable named `x` and store the value 5 inside it.

---

## 1.2 Arrays

Python:

```python
A = [10, 20, 30, 40]
```

C++:

```cpp
float A[4] = {10, 20, 30, 40};
```

Indexing starts at 0:

```cpp
A[0]  // 10
A[1]  // 20
A[2]  // 30
A[3]  // 40
```

Conceptually:

```text
index     0      1      2      3
        ┌────┬────┬────┬────┐
value   │ 10 │ 20 │ 30 │ 40 │
        └────┴────┴────┴────┘
```

---

## 1.3 `for` loops

Python:

```python
for i in range(4):
    C[i] = A[i] + B[i]
```

C++:

```cpp
for (int i = 0; i < 4; i++) {
    C[i] = A[i] + B[i];
}
```

The three pieces are:

```cpp
int i = 0;   // initialize
i < 4;       // continue condition
i++          // increment
```

`i++` means approximately:

```cpp
i = i + 1;
```

This loop:

```cpp
for (int i = 0; i < 4; i++) {
    C[i] = A[i] + B[i];
}
```

is equivalent to:

```cpp
C[0] = A[0] + B[0];
C[1] = A[1] + B[1];
C[2] = A[2] + B[2];
C[3] = A[3] + B[3];
```

This becomes very important in CUDA because CUDA often replaces this serial loop with many GPU threads.

---

# 2. Pointers

Pointers are one of the most important C++ concepts for CUDA.

CUDA APIs constantly use syntax such as:

```cpp
float* A;
cudaMalloc(&A, size);
```

To understand that, you need four ideas:

```text
value
address
pointer
dereference
```

---

## 2.1 Values and addresses

Suppose:

```cpp
int x = 5;
```

Conceptually:

```text
memory address      value
0x1000              5
```

Then:

```cpp
x
```

means:

> the value stored in `x`

while:

```cpp
&x
```

means:

> the memory address where `x` is stored

So:

```text
x   -> 5
&x  -> address of x, e.g. 0x1000
```

The exact address is not important.

---

## 2.2 What is a pointer?

A pointer is simply a variable whose value is a memory address.

Example:

```cpp
int x = 5;
int* p = &x;
```

Here:

```cpp
int* p
```

means:

> `p` is a pointer to an integer.

And:

```cpp
p = &x;
```

means:

> store the address of `x` inside `p`.

Conceptually:

```text
p
┌────────┐
│ 0x1000 │
└───┬────┘
    │
    ▼
address 0x1000
┌─────┐
│  5  │
└─────┘
   x
```

So:

```text
x   = 5
&x  = 0x1000
p   = 0x1000
```

---

## 2.3 Dereferencing: `*p`

If `p` contains an address, then:

```cpp
*p
```

means:

> go to the address stored inside `p` and read the value there.

Example:

```cpp
int x = 5;
int* p = &x;
```

Then:

```text
p   -> address of x
*p  -> value stored at that address
```

So:

```cpp
*p
```

evaluates to:

```text
5
```

---

## 2.4 `*` has two related meanings

Declaration:

```cpp
int* p;
```

means:

> declare `p` as a pointer to an integer.

Usage:

```cpp
*p
```

means:

> dereference `p`, i.e. read the value stored at the address in `p`.

---

## 2.5 Modifying through a pointer

Example:

```cpp
int x = 10;
int* p = &x;

*p = 20;
```

Because `p` points to `x`, this changes `x`.

Afterward:

```text
x = 20
```

So:

```cpp
*p = 20;
```

means:

> go to the memory location that `p` points to and store 20 there.

---

## 2.6 Python analogy

Python hides addresses, but references behave somewhat similarly.

```python
x = [10]
y = x

y[0] = 20
print(x)
```

prints:

```python
[20]
```

because both `x` and `y` refer to the same object.

C++ pointers make this kind of indirection explicit.

---

# 3. Arrays and Pointer Arithmetic

Suppose:

```cpp
float A[4] = {10, 20, 30, 40};
```

Memory might look conceptually like:

```text
address      value
0x1000       10
0x1004       20
0x1008       30
0x100C       40
```

A typical `float` is 4 bytes.

In many C/C++ expressions, the array name:

```cpp
A
```

acts like a pointer to the first element.

Approximately:

```cpp
A == &A[0]
```

So:

```cpp
float* p = A;
```

means:

```text
p -> A[0]
```

---

## 3.1 Pointer arithmetic

If:

```cpp
float* p = A;
```

then:

```cpp
p + 1
```

does not mean "add one byte."

Because `p` is a `float*`, C++ knows each element is a float.

So:

```text
p       -> A[0]
p + 1   -> A[1]
p + 2   -> A[2]
p + 3   -> A[3]
```

Therefore:

```cpp
*(p + 2)
```

returns:

```text
30
```

And:

```cpp
A[2]
```

is conceptually equivalent to:

```cpp
*(A + 2)
```

This is important because CUDA arrays are commonly passed around as pointers.

---

# 4. `sizeof`, Dynamic Memory, Stack, and Heap

Python handles memory allocation automatically.

C++ often requires explicit memory management.

---

## 4.1 `sizeof`

Suppose:

```cpp
int N = 4;
```

and you want enough memory for 4 floats.

A typical float is 4 bytes.

So:

```text
4 floats × 4 bytes = 16 bytes
```

C++ calculates type sizes using:

```cpp
sizeof(float)
```

Example:

```cpp
size_t size = N * sizeof(float);
```

If `N = 4`, then `size` is typically:

```text
16 bytes
```

`size_t` is an integer type commonly used to represent memory sizes.

Typical sizes:

```text
sizeof(float)  ≈ 4 bytes
sizeof(int)    ≈ 4 bytes
sizeof(double) ≈ 8 bytes
```

Use `sizeof(...)` instead of hardcoding sizes.

---

## 4.2 Stack vs heap

A simplified model:

```text
stack -> small/local variables, automatically managed
heap  -> dynamically allocated memory
```

Example local variable:

```cpp
void foo() {
    int x = 5;
}
```

`x` commonly lives on the stack and disappears automatically when `foo()` returns.

For large or dynamically sized arrays, heap allocation is often used.

---

## 4.3 C++ `new`

Example:

```cpp
float* A = new float[N];
```

This means:

> allocate enough heap memory for `N` floats and return the address of that memory.

`A` stores that address.

Conceptually:

```text
A
│
▼
heap memory
┌────┬────┬────┬────┐
│A[0]│A[1]│A[2]│ ...│
└────┴────┴────┴────┘
```

You can then use:

```cpp
A[0] = 10;
A[1] = 20;
```

When done:

```cpp
delete[] A;
```

---

## 4.4 `malloc`

Another allocation API is:

```cpp
float* A = (float*)malloc(N * sizeof(float));
```

`malloc` receives a number of bytes.

So:

```cpp
malloc(N * sizeof(float))
```

means:

> allocate enough raw memory to store `N` floats.

When using `malloc`, free memory with:

```cpp
free(A);
```

Pairing rules:

```text
new[]     <-> delete[]
malloc()  <-> free()
```

Do not mix them.

For CUDA, you mainly need to recognize the pointer and allocation pattern rather than master all C++ allocation styles.

---

# 5. CPU Memory vs GPU Memory

CUDA uses two important words:

```text
host   -> CPU
device -> GPU
```

So:

```text
host memory   -> CPU/system RAM
device memory -> GPU VRAM
```

A classic CUDA mental model is:

```text
CPU / Host                    GPU / Device
System RAM                    GPU VRAM
```

They are separate memory spaces.

---

## 5.1 Common naming convention

You may see:

```cpp
float* h_A;
float* d_A;
```

where:

```text
h_A -> host pointer
d_A -> device pointer
```

This is only a naming convention, but it is very common.

Conceptually:

```text
h_A -> array in CPU memory
d_A -> array in GPU memory
```

---

# 6. `cudaMalloc`

Suppose:

```cpp
float* d_A;
```

This creates a pointer variable, but it does not yet point to valid GPU memory.

Then:

```cpp
cudaMalloc(&d_A, size);
```

allocates GPU memory.

Conceptually:

```text
before:
d_A = unknown

cudaMalloc allocates GPU memory

after:
d_A = address of allocated GPU memory
```

---

## 6.1 Why `&d_A`?

This is important.

`d_A` itself is a variable that stores a GPU address.

`cudaMalloc` needs to modify `d_A` so that it contains the newly allocated address.

Therefore it needs:

```cpp
&d_A
```

which means:

> the address of the pointer variable `d_A`

Think of it like this:

```text
d_A
┌─────────────┐
│ GPU address │
└─────────────┘
```

`cudaMalloc` fills that box with the address of GPU memory.

---

## 6.2 Allocation is not copying

This:

```cpp
cudaMalloc(&d_A, size);
```

only creates space.

It does not copy actual values from the CPU.

After allocation:

```text
GPU memory:
[?, ?, ?, ?]
```

The memory exists, but the contents are not yet your input data.

---

# 7. `cudaMemcpy`

To copy data from CPU to GPU:

```cpp
cudaMemcpy(
    d_A,
    h_A,
    size,
    cudaMemcpyHostToDevice
);
```

Argument order:

```cpp
cudaMemcpy(
    destination,
    source,
    number_of_bytes,
    direction
);
```

So:

```text
destination first
source second
```

Example:

```text
CPU RAM                    GPU VRAM

h_A [1,2,3,4]  --------->  d_A [1,2,3,4]
```

using:

```cpp
cudaMemcpy(
    d_A,
    h_A,
    size,
    cudaMemcpyHostToDevice
);
```

---

## 7.1 Copying results back

After GPU computation:

```cpp
cudaMemcpy(
    h_C,
    d_C,
    size,
    cudaMemcpyDeviceToHost
);
```

means:

```text
GPU -> CPU
```

So:

```text
GPU VRAM                  CPU RAM

d_C [11,22,33,44] -----> h_C [11,22,33,44]
```

---

# 8. Typical CUDA Memory Workflow

A classic CUDA program often follows this pattern:

```text
1. Create CPU arrays
2. Allocate GPU arrays
3. Copy inputs CPU -> GPU
4. Launch GPU kernel
5. Copy output GPU -> CPU
6. Free memory
```

Example skeleton:

```cpp
int N = 4;
size_t size = N * sizeof(float);

float h_A[4] = {1, 2, 3, 4};
float h_B[4] = {10, 20, 30, 40};
float h_C[4];

float* d_A;
float* d_B;
float* d_C;

cudaMalloc(&d_A, size);
cudaMalloc(&d_B, size);
cudaMalloc(&d_C, size);

cudaMemcpy(
    d_A,
    h_A,
    size,
    cudaMemcpyHostToDevice
);

cudaMemcpy(
    d_B,
    h_B,
    size,
    cudaMemcpyHostToDevice
);

// GPU kernel runs here

cudaMemcpy(
    h_C,
    d_C,
    size,
    cudaMemcpyDeviceToHost
);

cudaFree(d_A);
cudaFree(d_B);
cudaFree(d_C);
```

---

# 9. Your First CUDA Kernel

A CUDA function that runs on the GPU is commonly called a **kernel**.

Example:

```cpp
__global__ void add(
    float* A,
    float* B,
    float* C
) {
    int i = threadIdx.x;

    C[i] = A[i] + B[i];
}
```

---

## 9.1 `__global__`

A normal C++ function:

```cpp
void add() {
}
```

runs on the CPU.

A CUDA kernel:

```cpp
__global__ void add() {
}
```

runs on the GPU.

A useful Python analogy is:

```python
@gpu
def add(...):
    ...
```

---

## 9.2 Every thread runs the same kernel

If 4 GPU threads are launched, all 4 execute the same function:

```text
Thread 0 executes add()
Thread 1 executes add()
Thread 2 executes add()
Thread 3 executes add()
```

The code is identical for every thread.

The difference comes from each thread having a different thread index.

---

# 10. `threadIdx.x`

CUDA automatically gives every thread an ID within its block.

Inside a kernel:

```cpp
threadIdx.x
```

means:

> my thread number inside this block

If a block has 4 threads:

```text
thread 0 -> threadIdx.x = 0
thread 1 -> threadIdx.x = 1
thread 2 -> threadIdx.x = 2
thread 3 -> threadIdx.x = 3
```

So:

```cpp
int i = threadIdx.x;
```

lets each thread operate on a different array element.

Example:

```cpp
C[i] = A[i] + B[i];
```

becomes conceptually:

```text
thread 0 -> C[0] = A[0] + B[0]
thread 1 -> C[1] = A[1] + B[1]
thread 2 -> C[2] = A[2] + B[2]
thread 3 -> C[3] = A[3] + B[3]
```

---

# 11. Kernel Launch Syntax

CUDA uses:

```cpp
kernel<<<number_of_blocks, threads_per_block>>>(arguments);
```

Example:

```cpp
add<<<1, 4>>>(d_A, d_B, d_C);
```

means:

```text
launch 1 block
with 4 threads in that block
```

Total launched threads:

```text
1 × 4 = 4
```

The syntax:

```cpp
<<< ... >>>
```

is CUDA-specific, not normal C++.

---

# 12. From CPU Loop to GPU Threads

CPU version:

```cpp
for (int i = 0; i < N; i++) {
    C[i] = A[i] + B[i];
}
```

CUDA idea:

```text
thread 0 -> element 0
thread 1 -> element 1
thread 2 -> element 2
...
```

Instead of one CPU loop iterating over all elements, many GPU threads work on elements in parallel.

This is one of the most important conceptual transitions in CUDA.

---

# 13. Blocks and Grids

A single CUDA block has a limited number of threads.

For large arrays, CUDA uses many blocks.

Hierarchy:

```text
Grid
  |
  +-- Block 0
  |     +-- Thread 0
  |     +-- Thread 1
  |     +-- ...
  |
  +-- Block 1
  |     +-- Thread 0
  |     +-- Thread 1
  |     +-- ...
  |
  +-- ...
```

The entire set of blocks is called a **grid**.

---

# 14. `threadIdx.x` Is Local to a Block

Suppose each block contains 4 threads.

Then:

```text
Block 0:
threadIdx.x = 0, 1, 2, 3

Block 1:
threadIdx.x = 0, 1, 2, 3

Block 2:
threadIdx.x = 0, 1, 2, 3
```

So `threadIdx.x` repeats in every block.

Therefore:

```cpp
int i = threadIdx.x;
```

is not enough when using multiple blocks.

We need a global thread index.

---

# 15. `blockIdx.x`

CUDA gives each block an ID.

```cpp
blockIdx.x
```

means:

> which block am I in?

Example:

```text
Block 0 -> blockIdx.x = 0
Block 1 -> blockIdx.x = 1
Block 2 -> blockIdx.x = 2
```

---

# 16. `blockDim.x`

```cpp
blockDim.x
```

means:

> how many threads are in this block along the x dimension?

Example:

```cpp
kernel<<<3, 4>>>();
```

gives:

```text
gridDim.x  = 3
blockDim.x = 4
```

---

# 17. Global Thread Index

The standard 1D global thread index is:

```cpp
int i =
    blockIdx.x * blockDim.x
    + threadIdx.x;
```

Why?

Because:

```text
number of threads before my block
+
my position inside this block
```

Example with 4 threads per block:

```text
Block 0:
local IDs   0 1 2 3
global IDs  0 1 2 3

Block 1:
local IDs   0 1 2 3
global IDs  4 5 6 7

Block 2:
local IDs   0 1 2 3
global IDs  8 9 10 11
```

Formula:

```text
global index
=
block index × block size
+
local thread index
```

Example:

```text
blockIdx.x  = 2
blockDim.x  = 4
threadIdx.x = 3

i = 2 × 4 + 3 = 11
```

So that thread handles:

```text
element 11
```

---

# 18. Bounds Checking

Suppose:

```text
N = 1000
threads_per_block = 256
```

You need:

```text
ceil(1000 / 256) = 4 blocks
```

That launches:

```text
4 × 256 = 1024 threads
```

But only elements:

```text
0 ... 999
```

exist.

Therefore real kernels usually contain:

```cpp
if (i < N) {
    C[i] = A[i] + B[i];
}
```

Threads with:

```text
i >= N
```

simply do nothing.

This is normal.

---

# 19. Ceiling Division

To compute the number of blocks:

```cpp
int numBlocks =
    (N + threadsPerBlock - 1)
    / threadsPerBlock;
```

Example:

```text
N = 1000
threadsPerBlock = 256
```

Then:

```text
(1000 + 256 - 1) / 256
=
1255 / 256
=
4
```

with integer division.

This is a standard CUDA idiom.

---

# 20. A Realistic 1D Vector Addition Kernel

```cpp
__global__ void vectorAdd(
    float* A,
    float* B,
    float* C,
    int N
) {
    int i =
        blockIdx.x * blockDim.x
        + threadIdx.x;

    if (i < N) {
        C[i] = A[i] + B[i];
    }
}
```

Launch:

```cpp
int threadsPerBlock = 256;

int numBlocks =
    (N + threadsPerBlock - 1)
    / threadsPerBlock;

vectorAdd<<<numBlocks, threadsPerBlock>>>(
    d_A,
    d_B,
    d_C,
    N
);
```

---

# 21. Important 1D CUDA Built-ins

```text
threadIdx.x
    -> my thread position inside the block

blockIdx.x
    -> which block am I in?

blockDim.x
    -> number of threads in each block

gridDim.x
    -> number of blocks in the grid
```

Key formula:

```cpp
int i =
    blockIdx.x * blockDim.x
    + threadIdx.x;
```

---

# 22. 2D CUDA Indexing

Matrices naturally use 2D indexing.

A matrix element is identified by:

```text
row
column
```

CUDA supports 2D thread layouts.

---

## 22.1 `threadIdx.x` and `threadIdx.y`

Conceptually:

```text
threadIdx.x -> column-like coordinate
threadIdx.y -> row-like coordinate
```

A 2D block might look like:

```text
             threadIdx.x
          0      1      2      3
       ┌──────┬──────┬──────┬──────┐
y = 0  │  T   │  T   │  T   │  T   │
       ├──────┼──────┼──────┼──────┤
y = 1  │  T   │  T   │  T   │  T   │
       ├──────┼──────┼──────┼──────┤
y = 2  │  T   │  T   │  T   │  T   │
       └──────┴──────┴──────┴──────┘
```

---

# 23. `dim3`

CUDA provides:

```cpp
dim3
```

for 1D/2D/3D dimensions.

Example:

```cpp
dim3 block(4, 3);
```

means:

```text
blockDim.x = 4
blockDim.y = 3
blockDim.z = 1
```

Total threads:

```text
4 × 3 = 12
```

You can think of `dim3` roughly like a Python tuple storing dimensions.

---

# 24. 2D Blocks and Grids

Blocks can also be arranged in 2D.

CUDA provides:

```cpp
blockIdx.x
blockIdx.y
```

and:

```cpp
threadIdx.x
threadIdx.y
```

So the 2D global position is:

```cpp
int col =
    blockIdx.x * blockDim.x
    + threadIdx.x;

int row =
    blockIdx.y * blockDim.y
    + threadIdx.y;
```

This is simply the 1D global-index formula applied separately in x and y.

---

# 25. Example 2D Index Calculation

Suppose:

```text
blockDim.x = 4
blockDim.y = 3

blockIdx.x = 2
blockIdx.y = 1

threadIdx.x = 1
threadIdx.y = 2
```

Then:

```text
col = 2 × 4 + 1 = 9
row = 1 × 3 + 2 = 5
```

So this thread handles:

```text
matrix[5][9]
```

---

# 26. Matrices Are Usually Stored as Flat Memory

A matrix:

```text
1 2 3
4 5 6
```

looks 2D logically, but row-major memory is flat:

```text
1 2 3 4 5 6
```

So a logical matrix coordinate:

```text
(row, col)
```

must be converted into a flat index.

For row-major storage:

```cpp
int index = row * cols + col;
```

---

# 27. Why `row * cols + col`?

Suppose:

```text
cols = 3
row  = 1
col  = 2
```

One full previous row contains:

```text
3 elements
```

Then move 2 positions into the current row:

```text
index = 1 × 3 + 2 = 5
```

So:

```text
A[1][2]
```

corresponds to flat memory:

```cpp
A[5]
```

---

# 28. 2D Matrix Addition Kernel

```cpp
__global__ void matrixAdd(
    float* A,
    float* B,
    float* C,
    int rows,
    int cols
) {
    int col =
        blockIdx.x * blockDim.x
        + threadIdx.x;

    int row =
        blockIdx.y * blockDim.y
        + threadIdx.y;

    if (row < rows && col < cols) {
        int index = row * cols + col;

        C[index] =
            A[index] + B[index];
    }
}
```

Each thread computes one matrix element.

---

# 29. Launching a 2D Kernel

Example:

```cpp
dim3 block(16, 16);
```

means:

```text
16 × 16 = 256 threads per block
```

Then:

```cpp
dim3 grid(
    (cols + block.x - 1) / block.x,
    (rows + block.y - 1) / block.y
);
```

Launch:

```cpp
matrixAdd<<<grid, block>>>(
    d_A,
    d_B,
    d_C,
    rows,
    cols
);
```

For a 1000×1000 matrix:

```text
ceil(1000 / 16) = 63
```

so the grid is approximately:

```text
63 × 63 blocks
```

Some threads at the edges fall outside the matrix, so we use:

```cpp
if (row < rows && col < cols)
```

---

# 30. Core Mental Model

The most important CUDA picture so far is:

```text
Python / NumPy thinking:
    C = A + B

CUDA thinking:
    many threads each handle one element
```

For 1D:

```text
thread i -> C[i]
```

For 2D:

```text
thread (row, col) -> C[row][col]
```

CUDA execution hierarchy:

```text
Grid
  |
  +-- Block
        |
        +-- Thread
```

CUDA indexing:

```text
threadIdx  -> local position inside a block
blockIdx   -> block position inside the grid
blockDim   -> block dimensions
gridDim    -> grid dimensions
```

1D global index:

```cpp
int i =
    blockIdx.x * blockDim.x
    + threadIdx.x;
```

2D global coordinates:

```cpp
int col =
    blockIdx.x * blockDim.x
    + threadIdx.x;

int row =
    blockIdx.y * blockDim.y
    + threadIdx.y;
```

2D to flat memory:

```cpp
int index =
    row * cols + col;
```

---

# 31. CPU vs GPU Mental Model

CPU code:

```cpp
for (int i = 0; i < N; i++) {
    C[i] = A[i] + B[i];
}
```

Conceptually:

```text
CPU:
i = 0
then i = 1
then i = 2
then ...
```

GPU code:

```text
thread 0 -> element 0
thread 1 -> element 1
thread 2 -> element 2
...
```

The GPU exposes massive parallelism rather than one explicit serial loop.

---

# 32. What You Should Know Before Starting the CUDA Slides

You now have enough foundation to read a beginner CUDA lecture if the following concepts are comfortable:

## C++ foundation

```text
int / float / double
arrays
for loops
functions
pointers
&x
*p
pointer arithmetic
sizeof
dynamic allocation
```

## CUDA memory foundation

```text
host = CPU
device = GPU

h_A -> CPU pointer
d_A -> GPU pointer

cudaMalloc
cudaMemcpy
cudaFree
```

## CUDA execution foundation

```text
__global__
kernel launch <<< >>>
threadIdx
blockIdx
blockDim
gridDim
grid
block
thread
```

## Indexing foundation

```text
1D:
blockIdx.x * blockDim.x + threadIdx.x

2D:
row / col global indexing

flat matrix index:
row * cols + col
```

---

# 33. Where the CUDA Slides Go Next

The next important topics are no longer generic C++ prerequisites. They are the actual CUDA performance model:

```text
1. Warps
2. Streaming Multiprocessors (SMs)
3. SIMT execution
4. Warp divergence
5. Global memory
6. Shared memory
7. Registers
8. Synchronization
9. Coalesced memory access
10. Shared-memory tiling
11. Matrix multiplication
12. Parallel reduction
13. Occupancy
14. Tensor Cores
```

At that point, the main challenge is no longer syntax.

The challenge becomes:

> understanding how thousands of logical CUDA threads map onto real GPU hardware and how memory access patterns determine performance.

That is the right point to begin reading the CUDA programming slides directly.

---

# 34. Ultra-Short Cheat Sheet

```cpp
// GPU pointer
float* d_A;

// allocate GPU memory
cudaMalloc(&d_A, N * sizeof(float));

// CPU -> GPU
cudaMemcpy(
    d_A,
    h_A,
    N * sizeof(float),
    cudaMemcpyHostToDevice
);

// CUDA kernel
__global__ void kernel(float* A, int N) {

    int i =
        blockIdx.x * blockDim.x
        + threadIdx.x;

    if (i < N) {
        // process A[i]
    }
}

// launch
int threads = 256;
int blocks = (N + threads - 1) / threads;

kernel<<<blocks, threads>>>(d_A, N);

// GPU -> CPU
cudaMemcpy(
    h_A,
    d_A,
    N * sizeof(float),
    cudaMemcpyDeviceToHost
);

// free GPU memory
cudaFree(d_A);
```

For a matrix:

```cpp
int col =
    blockIdx.x * blockDim.x
    + threadIdx.x;

int row =
    blockIdx.y * blockDim.y
    + threadIdx.y;

int index =
    row * cols + col;
```

---

# 35. Final Mental Picture

```text
                  CPU / HOST

          h_A -> CPU memory
          h_B -> CPU memory

                 |
                 | cudaMemcpy
                 v

                  GPU / DEVICE

          d_A -> GPU memory
          d_B -> GPU memory
          d_C -> GPU memory

                 |
                 | kernel launch
                 v

           Grid of CUDA blocks
                 |
          +------+------+
          |             |
       Block 0        Block 1
          |             |
       threads        threads

Each thread:

global index
=
blockIdx * blockDim
+
threadIdx

Then:

C[i] = A[i] + B[i]
```

The next conceptual jump is:

```text
CUDA says "many threads"

but GPU hardware actually executes those threads in groups called warps,
on hardware units called Streaming Multiprocessors (SMs).
```

That is where the CUDA performance model begins.
