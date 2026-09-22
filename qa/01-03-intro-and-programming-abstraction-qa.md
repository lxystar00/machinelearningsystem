# Q&A Log — Slides 1–3 (Course Intro, MLSys Overview, DL Programming Abstraction)

## Q1. The instructor's "first deep learning project" story (44k lines of CUDA, 6 months, failed) is contrasted with "100 lines of Python, a few hours." What's the actual point of that contrast — what does it say about what an ML systems framework *is*, conceptually?

**My answer:** ML system shortens the code and fastens the development of DL. It could automate a lot of things, so people only focus on the top level rather than every aspect.

**Feedback/refinement:** Correct. More precisely: the framework's abstraction (automatic differentiation + a library of tensor operators) lets a researcher express a model in ~100 lines of Python. The framework then handles gradient computation, hardware dispatch, and (as later lectures cover) parallelization/optimization — turning "6 months of hand-written CUDA" into "a few hours."

---

## Q2. The 5-layer stack is: Automatic Differentiation → Graph-Level Optimization → Parallelization → Kernel Generation (Code Optimization) → Memory Optimization. What problem does each layer solve?

**My answer:**
- Automatic differentiation: compute how much each parameter contributes to the loss; once we have it, we can lower the loss and update the params.
- Graph level optimization: decide how to merge modules and arrange kernels based on current resources; reduce total time and maximize resource use.
- Parallelization: not sure.
- Kernel generation / memory optimization: not sure.

**Feedback/refinement:**
- Automatic differentiation ✓ — build the backward graph automatically from the forward graph so every parameter's gradient w.r.t. loss is computed without the user deriving it by hand.
- Graph-level optimization — close, but reframe: it's not about "current resources" at runtime so much as applying mathematically-equivalent rewrites to the static graph (e.g. fuse conv+batchnorm, fuse multiple convs, eliminate redundant nodes) to cut kernel-launch overhead and memory traffic — happens before execution, on the graph structure itself.
- Parallelization — splitting the computation across multiple devices (GPUs/nodes) because one device doesn't have enough compute/memory. Two main flavors: data parallelism and model parallelism.
- Kernel Generation (code optimization) — for each operator (conv, matmul, etc.) you need an actual low-level program that runs fast on specific hardware. Vendor libraries (cuDNN, cuBLAS) hand-write this; modern systems (TVM) automatically search over loop transformations/thread bindings/tensorization to generate fast code.
- Memory optimization — GPU memory is the bottleneck: training must keep every intermediate activation alive from forward through backward pass. Techniques like recomputation/checkpointing reduce peak memory.

---

## Q3. The ResNet fusion example showed the "optimized" graph was 30% faster on a V100 but 10% *slower* on a K80. Why does this specific result motivate *automated* (search + verify) graph optimization rather than more hand-written rules?

**My answer:** Different GPUs have different conditions, like different memory, the number of kernels...

**Feedback/refinement:** Different GPUs (V100 vs K80) have different compute-to-memory-bandwidth ratios, cache sizes, etc., so the same fusion that helps on one hurts on another — expert heuristics tuned with one hardware/model in mind don't generalize. Since optimizations must work across {operators} × {graph structures} × {hardware backends}, manually writing rules for every combination is combinatorially infeasible. Automation solves this by generating many candidate rewrites from mathematical equivalence, *verifying* correctness, then using search/profiling to pick the best one for the specific model+hardware pairing — instead of one-size-fits-all rules.

---

## Q4. What's the core difference between data parallelism and model parallelism — what gets split in each, and what has to be communicated between GPUs in each case?

**My answer:** Data parallelism: split the data and share one model, compute the total loss and update the params together; they communicate through some services provided by NVIDIA. Model parallelism: split the model and share all data, so some of the model runs first, others run next.

**Feedback/refinement:** ✓ Solid. To sharpen the communication difference: data parallelism communicates **gradients** (an all-reduce across GPUs, e.g. via NCCL) after each GPU independently computes gradients on its data shard using a full model replica. Model parallelism communicates **intermediate activations** across the layer boundary where the model is split — and because GPU 2 can't start until GPU 1 finishes its part, this is inherently more sequential/dependent than data parallelism.

---

## Q5. Why exactly are RNNs hard to parallelize during training, and why does attention fix that?

**My answer:** The input is the output or internals of the last network, so they should run one by one. Attention solves this because all the inputs can see each other.

**Feedback/refinement:** ✓ Good, slightly refine. In an RNN, state_t depends on state_{t-1}, so both forward and backward passes have O(sequence length) *sequential* steps — you literally cannot compute step 5 before step 4 finishes. Attention removes this dependency: every position computes its output as a weighted combination over *all* positions' values in one parallel operation (matrix multiplies), so the number of sequential steps stays constant regardless of sequence length — only the (parallelizable) compute grows.

---

## Q6. What's the difference between "define-then-run" (TF1 style) and "define-and-run" (PyTorch style), and why does the slide say define-then-run still has an advantage despite being less flexible?

**My answer:** TF1 style generates the graph first, then optimizes, then runs. PyTorch style generates and runs at the same time. (I thought the advantage was because it could optimize to the best.)

**Feedback/refinement:** ✓ Good, one addition. The reason define-then-run has an edge: having the *entire* graph available before execution lets the framework do **global** optimizations (fusion, memory planning, scheduling) across the whole computation, because it can see every future op. Define-and-run (eager, like PyTorch) executes op-by-op, so it doesn't have that full picture — though modern PyTorch closes this gap with JIT tracing/compilation (`torch.compile`), a hybrid approach.
