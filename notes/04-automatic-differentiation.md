# Slide 4: Automatic Differentiation

## Why this matters

Training needs the gradient of the loss with respect to every parameter (SGD: `w := w - lr * grad(L)(w)`). This lecture is about how a system computes that gradient automatically and efficiently, for arbitrary computation graphs.

## Three candidate approaches

### 1. Numerical differentiation

Approximate a derivative directly from its definition — nudge the input a tiny amount, see how much the output changes:

```
df/dtheta_i ≈ (f(theta + step*e_i) - f(theta)) / step
```

A more accurate centered version (error shrinks faster, ~step²):

```
df/dtheta_i ≈ (f(theta + step*e_i) - f(theta - step*e_i)) / (2*step)
```

**Problems:** numerical imprecision from floating point + finite step size, and — critically — you need to redo the whole process once *per parameter* to get the full gradient. With millions of parameters this is far too slow for training.

**Actual use case:** correctness-checking an AD implementation. Pick a random direction δ, estimate the directional derivative numerically, and compare against what your AD code computes:

```
delta · gradient(f)(theta) ≈ (f(theta + step*delta) - f(theta - step*delta)) / (2*step)
```

If they disagree, the AD implementation (or a specific operator's backward rule) has a bug. This is how gradient implementations get unit-tested (e.g. in HW1).

### 2. Symbolic differentiation

Apply calculus rules (sum, product, chain rule) directly to the formula to produce a new formula for the derivative.

**Problem — wasted/duplicated computation.** Example: f(theta) = theta_1 × theta_2 × ... × theta_n. The partial derivative w.r.t. theta_k is "the product of all other thetas." Computed naively and separately for each k, this costs roughly n×n multiplies overall, even though there's massive shared structure between these n formulas (they all share large overlapping sub-products). Symbolic differentiation on its own doesn't reuse that shared intermediate work.

**The missing idea:** track computation as a *graph* of intermediate steps and reuse intermediate results across all the gradients you need. That's automatic differentiation (AD).

## Computation graph recap

Example: y = ln(x1) + x1*x2 - sin(x2), with x1=2, x2=5. Decompose into single-operation steps (each an intermediate value v_i):

```
v1 = x1            = 2
v2 = x2            = 5
v3 = ln(v1)        = 0.693
v4 = v1 * v2       = 10
v5 = sin(v2)       = -0.959
v6 = v3 + v4       = 10.693
v7 = v6 - v5       = 11.652   <- this is y
```

This is the **forward evaluation trace**.

## Forward mode AD

Track, alongside each v_i, its derivative with respect to *one chosen input* (say x1) — call it v_i-dot. Propagate forward, in the same order as the original computation, applying the chain rule at each step:

```
v1-dot = 1                                    (v1 IS x1)
v2-dot = 0                                    (v2 = x2, doesn't depend on x1)
v3-dot = v1-dot / v1              = 0.5
v4-dot = v1-dot*v2 + v2-dot*v1    = 5
v5-dot = v2-dot*cos(v2)           = 0
v6-dot = v3-dot + v4-dot          = 5.5
v7-dot = v6-dot - v5-dot          = 5.5        <- this is dy/dx1
```

**Limitation:** for f: R^n -> R^k, you need a full separate forward pass *per input* to get the gradient w.r.t. every input — n passes total. In ML, the loss is a single scalar (k=1) but there are typically millions of parameters (huge n). Needing one pass per parameter is infeasible — same fundamental problem as numerical differentiation.

## Reverse mode AD (this is what "backpropagation" means)

Flip what's tracked: instead of "how does this value change as one input changes," track "how does the final scalar output change as this value changes." Call this the **adjoint**, v_i-bar = dy/dv_i. Start at the output (v_output-bar = 1) and propagate **backward**, in reverse topological order:

```
v7-bar = 1
v6-bar = v7-bar * 1          = 1          (v7 = v6 - v5, d(v7)/d(v6) = 1)
v5-bar = v7-bar * (-1)       = -1         (d(v7)/d(v5) = -1)
v4-bar = v6-bar * 1          = 1
v3-bar = v6-bar * 1          = 1
v2-bar = v5-bar*cos(v2) + v4-bar*v1   = 1.716
v1-bar = v4-bar*v2 + v3-bar*(1/v1)    = 5.5
```

**One backward pass through the graph gives the gradient with respect to every input at once** — both v1-bar and v2-bar came out of the same single pass. This matches exactly what ML training needs (one loss, many parameters), which is why every DL framework uses reverse mode as its core AD mechanism.

## Multiple downstream consumers -> sum partial adjoints

v1 (x1) is used in two places: computing v3 (via ln) and computing v4 (via multiply). Both downstream uses genuinely affect the final output, so v1's adjoint must be the **sum** of the contribution through each path — this is just the multivariable chain rule, applied node by node.

Formally: define a partial adjoint for each edge i->j as `v_i->j = v_j-bar * (dv_j/dv_i)`, and then `v_i-bar = sum over all j downstream of i of v_i->j`.

## The general reverse-mode algorithm

```
def gradient(out):
    node_to_grad = {out: [1]}            # dict: node -> list of partial adjoints received
    for i in reverse_topo_order(out):
        v_i_bar = sum(node_to_grad[i])   # sum all partial adjoints that reached node i
        for k in inputs(i):
            v_k_to_i = v_i_bar * (dv_i/dv_k)     # local derivative rule for this op
            append v_k_to_i to node_to_grad[k]   # propagate to k's list for later summing
    return adjoint of the original input(s)
```

Walk the graph once in reverse order; at each node sum whatever partial adjoints have arrived, then push new partial-adjoint contributions to each of its inputs.

## Two implementation strategies

- **Classic backprop** (early frameworks, e.g. Caffe, cuda-convnet): run backward operations directly on the *same* forward graph, computing/accumulating adjoint values in place as plain numbers. No new graph is built.
- **Reverse-mode AD by extending the graph** (modern frameworks — PyTorch, TensorFlow, etc.): treat gradient computation as *building new graph nodes* (e.g. `v4-bar` computed via an `id` node, `v2-bar`/`v1-bar` via `+`/`×` nodes) appended onto the original graph, using the same node/op abstraction as the forward pass.

**Why the second approach won:**
1. Since the backward pass is now an ordinary computation graph, the same graph-level optimizations (fusion, etc.) that apply to the forward pass apply to it too.
2. **Gradient of gradient becomes trivial**: because the adjoint values are graph nodes (not just numbers), the whole backward graph can itself be fed back into reverse-mode AD a second time, exactly like differentiating the forward graph the first time. Classic in-place backprop has no graph left to re-differentiate — its backward pass is a dead end, only numbers come out. (Part of HW1.)

## Reverse mode AD on tensors

Real DNNs operate on matrices/tensors, not scalars, but the exact same algorithm applies — you just need the "local derivative rule" (adjoint propagation rule) for each tensor operator.

**Worked example — matrix multiply.** For Z = X * W (X is m×k, W is k×n, Z is m×n), feeding into scalar loss y, given the adjoint of Z (Z-bar, same shape as Z):

- Each Z[i][j] = sum over k' of X[i][k'] * W[k'][j].
- X[i][k] only affects y through row i of Z, so by the chain rule: X-bar[i][k] = sum over j of (Z-bar[i][j] * dZ[i][j]/dX[i][k]).
- dZ[i][j]/dX[i][k] = W[k][j] (only the k'=k term of the sum survives).
- So X-bar[i][k] = sum over j of (Z-bar[i][j] * W[k][j]) — which is exactly matrix multiplication of Z-bar by W transposed.

**Result: X-bar = Z-bar times W-transposed.**

Shape check (useful trick for remembering/verifying such rules): X-bar must be m×k (same shape as X); Z-bar is m×n; W-transposed is n×k; (m×n) times (n×k) = m×k. ✓.

Intuition: the forward pass mixed X and W together via matmul to produce Z; the backward pass "undoes" that mixing using the *other* matrix, transposed — transposing swaps which axis gets summed over, turning the forward contraction into the correct backward contraction.

## Beyond tensors: general data types

The algorithm doesn't require values to be scalars or tensors — it works for any data type, as long as you define (a) an adjoint in the *same data shape* as the forward value, and (b) a propagation rule for each operator. Example: for a dict lookup `b = d["cat"]`, the adjoint of `d` is simply `{"cat": b-bar}` — only the accessed key receives a nonzero adjoint, other keys get none (or implicitly zero). This is why AD frameworks can be extended to structures like tuples, not just plain tensors.
