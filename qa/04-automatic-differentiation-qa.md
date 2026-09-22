# Q&A Log — Slide 4 (Automatic Differentiation)

## Q1. Why does reverse mode AD fit machine learning specifically better than forward mode AD?

**My answer:** Reverse mode only needs one pass in total. In contrast, forward mode needs many passes depending on the number of inputs.

**Feedback/refinement:** ✓ Exactly right. Reverse mode: one backward pass gets you the gradient w.r.t. every input simultaneously (since there's one scalar loss). Forward mode would need one pass per input — infeasible when there are millions of parameters.

---

## Q2. In the worked example, v1 (x1) feeds into both v3=ln(v1) and v4=v1×v2. Why do we need to *sum* two partial adjoints for v1-bar rather than just picking one path?

**My answer:** Because it has 2 downstreams, both of them contribute to the output.

**Feedback/refinement:** ✓ Right. More precisely: v1's value flows into both v3 and v4, so it has two "downstream consumers." Each consumer contributes its own effect on the final output, and since both effects are real and simultaneous, you sum them (multivariable chain rule) rather than picking one.

---

## Q3. What's the practical difference between "classic backprop" (run backward ops on the same forward graph) and "reverse mode AD by extending the graph" (build new graph nodes for adjoints)? Why does the second one make "gradient of gradient" straightforward?

**My answer:** The classic one doesn't build a new graph, it computes in place. The reverse mode AD by extending the graph builds a new graph, so it can also be optimized like merging conv and batchnorm. It's easy to compute adjoint.

**Feedback/refinement:** Mostly right — classic backprop computes gradients in-place (numbers only, no new graph), while reverse-mode-by-extending-the-graph builds actual new graph nodes for the adjoints, so the backward computation can be fused/optimized just like the forward pass. Sharpened: the gradient-of-gradient part specifically works because those adjoint nodes (v̄1, v̄2, etc.) *are* graph nodes now — not just numbers — so the whole backward graph can be treated as an ordinary computation graph and reverse-mode AD run on it *again*, exactly the same way it ran on the forward graph the first time. With classic backprop, there's no graph left to differentiate a second time — the backward pass just produced numbers, a dead end.

---

## Q4. What is numerical differentiation actually useful for in practice, given that it's too slow/inaccurate for training?

**My answer:** It helps to know whether the AD's results are correct or not.

**Feedback/refinement:** ✓ Right. Specifically: it's a correctness check/unit test for an AD implementation — compare the AD-computed gradient against a numerically-estimated one; if they disagree, the AD code (or a new operator's backward rule) has a bug.
