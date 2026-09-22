# Q&A Log — Slide 5 (Hardware Acceleration)

## Q1. In the vanilla matmul loop, why specifically is the DRAM traffic O(n^3) instead of something smaller, like O(n^2)?

**My answer:** There are 3 nested loops.

**Feedback/refinement:** Partially right — needs the mechanism, not just the loop count. Having 3 nested loops is why the total loop body runs n^3 times, but that alone doesn't force n^3 *DRAM* loads. The real reason is that there's **zero reuse** across iterations: for each of the n^2 (i,j) pairs, the inner k-loop reloads a *fresh* A[i][k] and B[j][k] from DRAM every time, even though many of those values were already loaded moments ago for a different (i,j) pair. n^2 output elements × n DRAM loads each = n^3 total DRAM loads. It's not the loop nesting itself that's the problem — it's that the loop structure gives each loaded value zero opportunity to be reused before being thrown away.

---

## Q2. In register tiling, why does making v1 and v2 bigger reduce DRAM traffic — walk through why `a` gets reused v2 times specifically.

**My answer:** Larger v1 and v2 means more data reused and less data traffic.

**Feedback/refinement:** Right, made concrete: for a fixed i, as j sweeps from 0 to n/v2, the *same* v1×v3 block of A sitting in registers gets reused at every one of those v2 iterations of j (A doesn't depend on j at all!) before it's finally discarded and the next i block gets loaded. So A gets loaded once and used v2 times → fewer total A-loads → less DRAM traffic. Symmetrically, B doesn't depend on i, so it gets reused v1 times.

---

## Q3. Why do we need a *second* level of tiling (cache-line tiling) on top of register tiling — what problem does it solve that register tiling alone doesn't?

**My answer:** Register capacity is limited, so we can use L1 cache — it's bigger, and the data load to register is much faster than in DRAM.

**Feedback/refinement:** Right idea, one addition: register tiling alone is already good, but registers are so small that you can only amortize load cost over a tiny amount of reuse (v1, v2 can't be large — you'd run out of registers). L1 cache is much bigger than the register file (though still far faster than DRAM), so blocking at the cache level lets you reuse DRAM-loaded data across a much larger window before eviction than the register file alone could ever support — that's why cache tiling further shrinks DRAM traffic (down to n^2 + n^3/b1) beyond what register tiling alone achieves.

---

## Q4. What's the real-world constraint that stops you from just picking enormous tile sizes (v1, v2, v3, b1, b2) to minimize DRAM/L1 traffic as much as possible?

**My answer:** Register capacity.

**Feedback/refinement:** ✓ Right — register capacity is exactly the constraint for register tiling (and correspondingly, L1 cache *size* is the analogous constraint for cache tiling — `b1*n + b2*n < L1 cache size`). Every tier of the hierarchy has this same tension: bigger tile → more reuse → less traffic from the tier below, but bounded by how much that tier can physically hold.
