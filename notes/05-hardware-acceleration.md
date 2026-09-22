# Slide 5: Hardware Acceleration

## Where this fits

This is the "Hardware Kernel Acceleration" layer of the ML systems stack: once you know *what* operator to run (e.g. a specific matmul), this is about making that single operator run fast on real hardware (CPU/GPU).

## General acceleration techniques

### Vectorization (SIMD)

Modern CPUs can process multiple numbers with a single instruction. `load_float4`/`add_float4`/`store_float4` load, add, and store 4 floats at once — 64 vector instructions instead of 256 scalar ones for a 256-element add. Requires memory to be aligned (e.g. to 128 bits = 4 floats) so the hardware can load a chunk as one unit.

### Data layout and strides

A matrix is logically 2D but memory is physically 1D — need a formula to map `A[i][j]` to a 1D offset:
- **Row-major**: `offset = i * num_columns + j` (same-row elements adjacent in memory).
- **Column-major**: `offset = j * num_rows + i` (same-column elements adjacent).
- **General strided format**: `offset = i * stride0 + j * stride1` — row/column-major are special cases of this.

**Why strides matter:** many operations become zero-copy (no data movement) by just changing the stride bookkeeping:
- Slice: change starting offset + shape.
- Transpose: swap stride0 and stride1.
- Broadcast: set a stride to 0 (every index along that axis reads the same location).

**Cost:** after such transformations, memory access is no longer contiguous, which breaks vectorization (can't SIMD-load scattered values) and often forces an explicit "compaction" copy back into a contiguous layout before further ops can run efficiently.

### Parallelization

Split loop iterations across CPU threads (e.g. `#pragma omp parallel for`) so multiple cores work simultaneously.

## Case study: matrix multiplication

### The vanilla version

```
for i in 0..n:
  for j in 0..n:
    C[i][j] = 0
    for k in 0..n:
      C[i][j] += A[i][k] * B[j][k]      # computes C = A dot B-transposed
```

O(n^3) multiply-adds (three nested loops of size n). Note it's deliberately `B[j][k]` (not `B[k][j]`) so both A and B are read row-wise — contiguous, vectorization-friendly access.

### The memory hierarchy — the core constraint

Registers (~0.5ns) → L1 cache (~7ns, 14x slower than registers) → L2 cache → DRAM (~200ns, 200x slower than L1). Each tier is faster but much smaller than the one below.

**Core principle of hardware-aware kernel design:** DRAM is huge but extremely slow relative to registers/cache. The whole game is to reuse data already pulled into fast memory as much as possible before it's discarded, to avoid repeated slow round-trips to DRAM.

### Why the vanilla loop is slow

For each of the n² (i,j) output pairs, the inner k-loop reloads a *fresh* A[i][k] and B[j][k] from DRAM every iteration — no reuse across (i,j) pairs, even though the same rows/columns get touched repeatedly for different outputs.

```
Load cost:     2 * dramspeed * n^3     (A and B each read n^3 times total from DRAM)
Register cost: 3                       (only a, b, c held at once)
```

The n³ DRAM traffic (versus only n² unique elements in each of A and B) is the bottleneck — massive redundant reloading because the loop structure gives loaded values zero chance to be reused.

### Fix 1: Register tiling

Compute a `v1 x v2` block of outputs at once using a `v1 x v3` chunk of A and `v2 x v3` chunk of B held in registers, instead of one scalar output at a time:

```
for i in 0..n/v1:
  for j in 0..n/v2:
    c[v1][v2] = 0
    for k in 0..n/v3:
      a[v1][v3] = A[i][k]        # loaded once, reused v2 times (doesn't depend on j)
      b[v2][v3] = B[j][k]        # loaded once, reused v1 times (doesn't depend on i)
      c += dot(a, b.T)
    C[i][j] = c
```

Because A doesn't depend on j, it gets reused v2 times before being discarded (once per column of the output tile); B doesn't depend on i, so it's reused v1 times.

```
Load cost:     dramspeed * (n^3/v2 + n^3/v1)   <- shrinks as v1, v2 grow
Register cost: v1*v3 + v2*v3 + v1*v2            <- grows as tile size grows
```

**Fundamental tradeoff:** bigger tiles = less DRAM traffic but more register space needed, and registers are a small, fixed, scarce resource — this bounds how large v1/v2/v3 can be.

### Fix 2: Cache-line (L1) tiling

Registers are too small to get much reuse. L1 cache is bigger (though still much faster than DRAM), so apply the same idea one level up: load a `b1`-row block of A and `b2`-row block of B into L1 cache, and reuse that L1-resident data across many register-tiled computations (register tiling runs as a sub-procedure *inside* this):

```
for i in 0..n/b1:
  a[b1][n] = A[i]              # loaded from DRAM into L1 once total across all i
  for j in 0..n/b2:
    b[b2][n] = B[j]            # reloaded from DRAM into L1 for every i iteration
    C[i][j] = dot(a, b.T)      # internally register-tiled
```

- A's DRAM->L1 cost: n^2 (each element loaded exactly once, ever).
- B's DRAM->L1 cost: n^3/b1 (all of B gets reloaded once per outer i iteration; bigger b1 means fewer reloads).

**Constraint:** `b1*n + b2*n < L1 cache size` (both blocks must fit together in L1), and `b1 % v1 == 0`, `b2 % v2 == 0` so register tiling still divides evenly inside.

Cache tiling helps *beyond* register tiling specifically because L1 cache is much bigger than the register file — it lets DRAM-loaded data be reused across a much wider window before eviction than registers alone could ever support.

### Putting it together (full hierarchy)

Combine both levels: DRAM -> L1 cache (outer tiling) -> registers (inner tiling) -> compute.

```
load cost = l1speed * (n^3/v2 + n^3/v1)     <- L1-to-register traffic
          + dramspeed * (n^2 + n^3/b1)      <- DRAM-to-L1 traffic
```

Every tier gets its own tiling, each amortizing the (expensive) cost of the tier below it by maximizing reuse before eviction.

### The unifying insight: memory load reuse

Whenever a piece of loaded data doesn't depend on some loop index, tiling that index lets the data be reused across the whole tile instead of being reloaded from scratch. In the register-tiling example: A (depends on i,k but not j) is reused v2 times because j got tiled; B (depends on j,k but not i) is reused v1 times because i got tiled.

**Generalizes beyond matmul**, e.g. convolution: `Output[b][co][y][x] = sum over k,ry,rx of Input[b][k][y+ry][x+rx] * Weight[co][k][ry][rx]`. Input doesn't depend on `co`, so tiling the output-channel dimension lets a loaded input patch be reused across multiple output channels; Weight doesn't depend on `b,y,x`, so tiling those lets a loaded weight be reused across many spatial positions/batch elements.

This "find what a loaded value doesn't depend on, then tile around it to maximize reuse" is exactly the search space that automated kernel generators (e.g. TVM, from slide 2/3) explore automatically, rather than a human hand-picking tile sizes (v1, v2, v3, b1, b2, ...) for every operator/hardware combination.
