# Slides 1–3: Course Intro, MLSys Overview, DL Programming Abstraction

## Why ML Systems matters

Three pillars fuel modern ML applications: **Methods** (research/algorithms), **Data**, and **Compute**. Most of the core algorithms (backprop, SVM, ConvNet, GBM) predate 2000; what changed since is the arrival of big data (2000s) and compute scaling (GPUs, TensorCore, 2006+).

The instructor's story illustrates the point concretely: a first DL project in 2010 took **44k lines of hand-written CUDA and 6 months**, and still failed. A modern framework does the same job in **~100 lines of Python and a few hours**. The gap between these two is exactly what an ML systems framework provides: an abstraction layer (automatic differentiation + a tensor-operator library) that lets a researcher express a model without hand-deriving gradients or hand-writing hardware kernels — the system handles gradient computation, hardware dispatch, parallelization, and optimization underneath.

**MLSys as a research field**: a holistic approach spanning Model, Data, and Compute together (not just better models, not just better systems) to solve real deployment problems (e.g. "make pedestrian detection X% accurate at Y-ms latency" is solved by combining more data + specialized hardware + hardware-aware model design + end-to-end systems — not any single lever alone).

## The 5-layer ML systems stack

Given an ML model, a system needs to do (roughly in this order):

1. **Automatic Differentiation** — automatically construct the backward computation graph from the forward graph, so every trainable parameter's gradient w.r.t. the loss is computed without manual derivation.
2. **Graph-Level Optimization** — apply mathematically-equivalent rewrites to the static computation graph (e.g. fuse Conv+BatchNorm into a single Conv with adjusted weights, fuse multiple convs, eliminate redundant ops) to cut kernel-launch overhead and memory traffic. This happens on the graph structure itself, before execution.
3. **Parallelization / Distributed Training** — split the computation across multiple devices because a single GPU doesn't have enough compute or memory (data parallelism, model parallelism — see below).
4. **Kernel Generation / Code Optimization** — for each operator (conv, matmul, ...), generate an actual fast low-level program for the target hardware. Vendor libraries (cuDNN, cuBLAS) hand-write these; but hand-writing can't keep pace with new operators/hardware and misses hardware-specific tuning — hence automated search-based code generation (e.g. TVM), searching over loop transformations, thread bindings, tensorization, cache locality, etc.
5. **Memory Optimization** — GPU/accelerator memory is often the bottleneck: training must keep every intermediate activation alive from the forward pass through the backward pass. Techniques like recomputation (gradient checkpointing) reduce peak memory so bigger models/batches fit.

## Why rule-based graph optimization doesn't scale

TensorFlow's rule-based optimizer (~200 rules, ~53K LOC) illustrates three failure modes:
- **Robustness**: expert heuristics tuned for one model/hardware don't transfer. E.g., turning on XLA made some real workloads 20%–2x *slower*.
- **Scalability**: new operators/graph structures require writing more rules by hand (TensorFlow uses ~4K LOC just to optimize convolution).
- **Performance**: hand rules miss subtle, model/hardware-specific optimizations.

**Concrete example (ResNet block)**: a sequence of graph rewrites (enlarge convs → fuse convs → fuse conv+add → fuse conv+relu) produced a final graph that was **30% faster on a V100 but 10% slower on a K80**. Different GPUs have different compute/memory-bandwidth ratios and cache behavior, so the same rewrite that helps one hardware backend can hurt another. Since optimizations must work across {ML operators} × {graph architectures} × {hardware backends}, manually enumerating rules for every combination is infeasible.

**The fix — automated graph optimization**: generate many *candidate* graph rewrites from the mathematical properties of ML ops, *verify* each is mathematically equivalent to the original, then use a cost model / profiling-based search to pick the best rewrite for the specific target model + hardware, instead of relying on one-size-fits-all expert rules.

## Parallelizing training

SGD update: `w := w - γ∇L(w) = w - (γ/n) Σⱼ ∇Lⱼ(w)` — the per-example gradient sum is naturally parallelizable.

- **Data parallelism**: partition the training data into batches; each GPU holds a *full copy* of the model and computes gradients on its own data shard; gradients are then **aggregated across GPUs** (e.g. all-reduce via NCCL) and applied identically to every replica. Communication = gradients/parameters.
- **Model parallelism**: split the *model* into subgraphs, each assigned to a different device (same or different data). Each device computes only its part of forward/backward; **intermediate activations are transferred between devices** at the split boundary. Because a downstream device can't start until the upstream device finishes its part, this is inherently more sequentially dependent than data parallelism.

## Deep learning models overview (context for the systems problems)

- **CNNs**: sequences of convolution + pooling/normalization/activation; used for vision tasks (classification, detection, segmentation, synthesis).
- **RNNs**: have an internal state updated as a sequence is processed (one-to-one, one-to-many, many-to-one, many-to-many). Represented in computation graphs (which must be DAGs) via **unrolling** — instantiate a fixed max depth of RNN cells connected in sequence, turning the self-loop into a chain.
  - **Inefficiency**: state_t depends on state_{t-1}, so both forward and backward passes have O(sequence length) *sequential*, unparallelizable steps — you cannot compute step t before step t-1 finishes. This limits training on long sequences.
- **Attention / Transformers**: treat each position's representation as a **query** that looks up a weighted combination of **values** from all other positions, in one parallel operation (matmuls). This removes the sequential dependency chain — the number of unparallelizable steps stays constant regardless of sequence length; only the (parallelizable) compute grows. This is the key reason Transformers train faster than RNNs on long sequences.
- **GNNs**: combine graph propagation (aggregate neighbor representations — sum, LSTM, etc.) with standard DNN operations, for relational/graph-structured data.
- **Mixture-of-Experts (MoE)**: a gating/router network selects a subset of "expert" sub-networks (e.g. FFNs) to process each token, so each expert specializes on a subset of cases. Switch Transformers = Transformers with FFN layers replaced by MoE layers.

## Programming abstractions for deep learning

**Computation graph abstraction**: nodes = operations, edges = data dependencies between them. This is the common substrate underneath every framework, regardless of front-end API style.

**Two construction styles**:
- **Define-then-run (TF1-style)**: first *declare* the entire computation graph (forward ops, loss, gradients via `tf.gradients`, update rule via `tf.assign`), then execute it repeatedly inside a `Session.run()` loop with different data fed in via `feed_dict`. Nothing actually computes until `sess.run()` is called.
- **Define-and-run (PyTorch-style, eager)**: the graph is constructed *on the fly*, interleaved with actual computation (e.g. `z = x - y` immediately computes and also builds the graph node), and `loss.backward()` walks that dynamically-built graph to compute gradients.

**Why the distinction matters**: define-and-run gives more flexibility to the programmer (e.g., easy Python control flow, per-step graph structure). Define-then-run's advantage is that having the **entire graph visible up front** lets the framework perform **global optimizations** (fusion, memory planning, scheduling) across the whole computation before any execution — something an eager, op-by-op executor can't do as easily. This tradeoff motivates modern hybrid approaches (JIT tracing/compilation, e.g. `torch.compile`) that try to get eager-mode flexibility with graph-mode optimization.

## Elements of Machine Learning (recap, applies to DL too)

Every ML method (including deep learning) is built from three ingredients:
1. **Model (hypothesis) class** — the parameterized function mapping inputs to predictions (e.g. logistic regression, or in DL, a compositional multi-layer model).
2. **Loss function** — measures how well a given parameter setting performs.
3. **Training (optimization) method** — a procedure (e.g. SGD) to find parameters minimizing the loss.

Deep learning's distinguishing idea: a **compositional multi-layer model** trained **end-to-end** (all layers' parameters learned jointly) — the loss and optimization machinery stay the same as classical ML.
