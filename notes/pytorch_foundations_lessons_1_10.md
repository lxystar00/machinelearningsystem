# PyTorch Foundations --- Review Notes

> A practical PyTorch foundation for someone who already knows Python
> and is learning ML systems, CUDA, and model implementation.

## How to Read PyTorch Code

For every tensor, build the habit of asking:

1.  **What is its shape?**
2.  **What does each dimension mean?**
3.  **What is its dtype?**
4.  **What device is it on?**
5.  **Does it require gradients?**

A useful ML-systems mental model is:

``` text
Tensor
├── shape
├── stride
├── dtype
├── device
└── requires_grad
```

For example:

``` python
path_xy.shape == [B, T, 2]
```

means:

-   `B`: batch size
-   `T`: number of timesteps
-   `2`: `(x, y)` coordinates

------------------------------------------------------------------------

# Lesson 1 --- Tensor Creation and Basic Properties

A `torch.Tensor` is similar to a NumPy array, but it can run on GPUs and
participate in automatic differentiation.

``` python
import torch
```

## Creating tensors

### `torch.tensor`

``` python
x = torch.tensor([1, 2, 3])
```

Creates a tensor from existing Python data.

### `torch.zeros`

``` python
x = torch.zeros(2, 3)
```

``` text
[[0., 0., 0.],
 [0., 0., 0.]]
```

### `torch.ones`

``` python
x = torch.ones(2, 3)
```

### `torch.arange`

``` python
x = torch.arange(0, 5)
# tensor([0, 1, 2, 3, 4])
```

Like Python `range`, the end is excluded.

### `torch.linspace`

``` python
x = torch.linspace(0, 1, steps=5)
```

Produces evenly spaced values between two endpoints.

### `torch.rand`

``` python
x = torch.rand(2, 3)
```

Uniform random values, typically in `[0, 1)`.

### `torch.randn`

``` python
x = torch.randn(2, 3)
```

Random values sampled from a standard normal distribution.

### `zeros_like` / `ones_like`

``` python
y = torch.zeros_like(x)
z = torch.ones_like(x)
```

Create tensors with the same shape/device/dtype family as another
tensor.

## Important tensor properties

``` python
x.shape
x.ndim
x.numel()
x.dtype
x.device
x.requires_grad
```

-   `.shape`: dimensions
-   `.ndim`: number of dimensions
-   `.numel()`: total number of elements
-   `.dtype`: element data type
-   `.device`: CPU/GPU location
-   `.requires_grad`: whether autograd tracks computations for gradient
    calculation

Common dtypes:

``` text
torch.float32
torch.float16
torch.bfloat16
torch.int64
torch.bool
```

------------------------------------------------------------------------

# Lesson 2 --- Shape Operations

Shape manipulation is one of the most important PyTorch skills.

## `view()`

``` python
x = torch.arange(12)

y = x.view(3, 4)
```

Changes the tensor's logical shape while preserving the number of
elements.

``` text
[12] → [3, 4]
```

The number of elements must stay the same.

### Using `-1`

PyTorch can infer one dimension:

``` python
x.view(3, -1)
```

For 12 elements:

``` text
[12] → [3, 4]
```

## `reshape()`

``` python
y = x.reshape(3, 4)
```

Similar to `view()`, but more flexible with non-contiguous tensors. It
may return a view when possible or make a copy when necessary.

## `unsqueeze()`

Adds a dimension of size 1.

``` python
x.shape
# [3]

x.unsqueeze(0).shape
# [1, 3]

x.unsqueeze(1).shape
# [3, 1]
```

A common example:

``` text
[C, H, W]
↓ unsqueeze(0)
[1, C, H, W]
```

This adds a batch dimension.

## `squeeze()`

Removes dimensions of size 1.

``` python
x.shape == [B, 1, D]

x.squeeze(1).shape
# [B, D]
```

Be careful with:

``` python
x.squeeze()
```

It removes **all** size-1 dimensions. If `B == 1`, this can accidentally
remove the batch dimension.

Prefer specifying the dimension when possible.

## `flatten()`

``` python
x.shape == [B, C, H, W]

y = x.flatten(start_dim=1)
```

Result:

``` text
[B, C, H, W]
→
[B, C*H*W]
```

## `expand()`

Logically expands dimensions of size 1 without physically copying all
data.

``` python
x.shape == [B, 1, D]

y = x.expand(-1, T, -1)
```

Result:

``` text
[B, 1, D]
→
[B, T, D]
```

`-1` means: keep the original size for this dimension.

## `repeat()`

Physically repeats tensor data.

``` python
x.repeat(...)
```

Conceptually:

-   `expand`: logical expansion, usually no full data copy
-   `repeat`: actual repetition, uses additional memory

------------------------------------------------------------------------

# Lesson 3 --- Indexing, Masks, Gather, and Scatter

## Basic indexing

``` python
x = torch.tensor([
    [10, 11, 12],
    [20, 21, 22]
])

x[0]
# [10, 11, 12]

x[1, 2]
# 22
```

Integer indexing removes the selected dimension.

## Slicing

``` python
x[:, 1]
x[1, :]
x[:, 0:2]
```

Python slices use `[start:end)`: the end is excluded.

For:

``` text
x.shape = [B, T, D]
```

examples:

``` python
x[:, 0, :]     # [B, D]
x[:, :10, :]   # [B, 10, D]
x[:, :, :2]    # [B, T, 2]
```

## Ellipsis `...`

``` python
x[..., :2]
```

means: preserve all leading dimensions and take the first two values of
the last dimension.

``` text
[B, T, 10] → [B, T, 2]
```

## Boolean masks

``` python
mask = x > 3
selected = x[mask]
```

A boolean mask selects values where the condition is `True`.

## `torch.where`

``` python
result = torch.where(condition, A, B)
```

For every position:

``` text
if condition is True  → choose A
if condition is False → choose B
```

## `clamp`

``` python
idx = idx.clamp(0, T - 1)
```

Keeps values inside a valid range.

## Advanced indexing

``` python
rows = torch.tensor([0, 1, 2])
cols = torch.tensor([3, 0, 2])

x[rows, cols]
```

This selects paired coordinates:

``` text
(0, 3)
(1, 0)
(2, 2)
```

### Per-batch trajectory indexing

Suppose:

``` text
path_xy.shape = [B, T, 2]
end_idx.shape = [B]
```

Then:

``` python
G_xy = path_xy[torch.arange(B), end_idx]
```

produces:

``` text
[B, 2]
```

Each batch element selects its own timestep.

## `index_select`

``` python
torch.index_select(x, dim, index)
```

Selects the same index list along a dimension for all remaining
coordinates.

## `gather`

`gather` supports different indices for different rows/batches.

``` python
x = torch.tensor([
    [10, 11, 12, 13],
    [20, 21, 22, 23]
])

idx = torch.tensor([
    [3, 1],
    [0, 2]
])

torch.gather(x, dim=1, index=idx)
```

Result:

``` text
[[13, 11],
 [20, 22]]
```

Mental model:

``` text
out[i, j] = x[i, idx[i, j]]
```

### Trajectory example

``` python
end_idx = (t_star - 1).clamp(0, T - 1)

G_xy = path_xy.gather(
    1,
    end_idx.view(B, 1, 1).expand(B, 1, 2)
).squeeze(1)
```

Shape flow:

``` text
end_idx                     [B]
view(B, 1, 1)               [B, 1, 1]
expand(B, 1, 2)             [B, 1, 2]
gather(dim=1)               [B, 1, 2]
squeeze(1)                  [B, 2]
```

For selecting one timestep per batch, advanced indexing is often
simpler:

``` python
G_xy = path_xy[torch.arange(B), end_idx]
```

## `scatter`

Conceptually:

``` text
gather  = read/collect values from selected positions
scatter = write/place values into selected positions
```

------------------------------------------------------------------------

# Lesson 4 --- Concatenation and Splitting

## `torch.cat`

Concatenates tensors along an **existing** dimension.

If:

``` text
a.shape = [2, 2]
b.shape = [2, 2]
```

then:

``` python
torch.cat([a, b], dim=0)
```

has shape:

``` text
[4, 2]
```

and:

``` python
torch.cat([a, b], dim=1)
```

has shape:

``` text
[2, 4]
```

Trajectory example:

``` text
A = [B, T1, D]
B = [B, T2, D]

torch.cat([A, B], dim=1)
→ [B, T1+T2, D]
```

Feature fusion:

``` text
image_features = [B, 256]
text_features  = [B, 128]

torch.cat([...], dim=-1)
→ [B, 384]
```

## `torch.stack`

Creates a **new** dimension.

If:

``` text
a.shape = [3]
b.shape = [3]
```

then:

``` python
torch.stack([a, b], dim=0)
```

gives:

``` text
[2, 3]
```

while:

``` python
torch.cat([a, b])
```

gives:

``` text
[6]
```

Important trajectory example:

``` text
path_x = [B, T]
path_y = [B, T]
```

``` python
path_xy = torch.stack([path_x, path_y], dim=-1)
```

Result:

``` text
[B, T, 2]
```

### Decision rule

``` text
Want an existing dimension to become longer → cat
Want a new semantic dimension             → stack
```

## `split`

Specify chunk sizes:

``` python
torch.split(x, 3)
```

or explicit sizes:

``` python
image_feat, text_feat = torch.split(
    x,
    [256, 128],
    dim=-1
)
```

## `chunk`

Specify the number of chunks:

``` python
parts = torch.chunk(x, 3, dim=...)
```

Mental distinction:

``` text
split → how large should the chunks be?
chunk → how many chunks do I want?
```

## `unbind`

Removes a dimension and returns its slices.

``` python
x, y = torch.unbind(path_xy, dim=-1)
```

If:

``` text
path_xy = [B, T, 2]
```

then:

``` text
x = [B, T]
y = [B, T]
```

------------------------------------------------------------------------

# Lesson 5 --- Math and Reduction Operations

A **reduction** aggregates values across one or more dimensions.

The key rule:

> `dim=k` usually means operate across/reduce dimension `k`, not
> preserve it.

Suppose:

``` text
x.shape = [B, T, D]
```

## `sum`

``` python
x.sum(dim=1)
```

sums over time:

``` text
[B, T, D]
→
[B, D]
```

## `mean`

``` python
x.mean(dim=1)
```

averages over time:

``` text
[B, T, D]
→
[B, D]
```

This can implement mean pooling.

## `keepdim=True`

Normally:

``` python
x.mean(dim=1)
# [B, D]
```

With:

``` python
x.mean(dim=1, keepdim=True)
# [B, 1, D]
```

The reduced dimension remains with size 1, which is especially useful
for broadcasting.

## `max`

``` python
values, indices = x.max(dim=1)
```

Returns both:

-   maximum values
-   positions of those maxima

## `argmax`

``` python
pred = logits.argmax(dim=-1)
```

If:

``` text
logits.shape = [B, C]
```

then:

``` text
pred.shape = [B]
```

Each value is the predicted class index.

`min` and `argmin` work analogously.

## Vector norm

For modern code, a clear option is:

``` python
distance = torch.linalg.vector_norm(delta_xy, dim=-1)
```

For:

``` text
delta_xy.shape = [B, T, 2]
```

the final dimension contains:

``` text
[dx, dy]
```

and the operation computes:

``` text
sqrt(dx² + dy²)
```

Result:

``` text
[B, T]
```

`torch.norm(...)` is also commonly encountered in existing code.

## `clamp`

``` python
x.clamp(min=0, max=10)
```

Restricts values to a range.

## `sigmoid`

``` python
prob = torch.sigmoid(logit)
```

Maps each value independently into `(0, 1)`.

Common for binary and multi-label tasks.

## `softmax`

``` python
prob = torch.softmax(logits, dim=-1)
```

Normalizes values along the chosen dimension so that they sum to 1.

For:

``` text
logits.shape = [B, C]
```

``` python
torch.softmax(logits, dim=-1)
```

produces class probabilities for each sample while preserving shape:

``` text
[B, C] → [B, C]
```

### Sigmoid vs. Softmax

``` text
Sigmoid
→ each output is independent
→ useful for multi-label prediction

Softmax
→ outputs compete along a dimension
→ useful for mutually exclusive classes
```

For training with `nn.CrossEntropyLoss`, pass **raw logits** rather than
manually applying softmax first.

## `topk`

``` python
values, indices = logits.topk(5, dim=-1)
```

For:

``` text
logits = [B, 1000]
```

results have shape:

``` text
values  = [B, 5]
indices = [B, 5]
```

## Trajectory metric example

``` python
dist = torch.linalg.vector_norm(
    pred_xy - gt_xy,
    dim=-1
)
```

If both trajectories are:

``` text
[B, T, 2]
```

then:

``` text
dist = [B, T]
```

Then:

``` python
ade_per_sample = dist.mean(dim=1)
```

gives:

``` text
[B]
```

and:

``` python
batch_ade = ade_per_sample.mean()
```

gives a scalar.

------------------------------------------------------------------------

# Lesson 6 --- Broadcasting

Broadcasting lets PyTorch perform operations on compatible tensors with
different shapes.

## The broadcasting rule

Compare dimensions **from right to left**.

Two dimensions are compatible when:

1.  They are equal, or
2.  One of them is `1`, or
3.  One tensor has no corresponding dimension.

## Example: `[B, T, D] + [D]`

``` text
x:       [B, T, D]
bias:          [D]
```

This works.

The same `[D]` bias is applied to every batch element and every
timestep.

## A common failure: `[B, T, D] + [B, D]`

Right-align:

``` text
[B, T, D]
   [B, D]
```

Comparison:

``` text
D vs D → OK
T vs B → not compatible in general
```

PyTorch does not know that the first `B` semantically means "batch."

Fix it:

``` python
y = y.unsqueeze(1)
```

Now:

``` text
[B, T, D]
[B, 1, D]
```

and:

``` text
D vs D → OK
T vs 1 → broadcast
B vs B → OK
```

## Trajectory example

``` text
path_xy.shape = [B, T, 2]
G_xy.shape    = [B, 2]
```

To subtract each trajectory's goal from every timestep:

``` python
delta = path_xy - G_xy.unsqueeze(1)
```

or:

``` python
delta = path_xy - G_xy[:, None, :]
```

because:

``` text
G_xy[:, None, :]

[B, 2]
→
[B, 1, 2]
```

Then:

``` text
[B, T, 2]
-
[B, 1, 2]
=
[B, T, 2]
```

## `keepdim` + broadcasting

Instead of:

``` python
center = x.mean(dim=1)
center = center.unsqueeze(1)
```

you can write:

``` python
center = x.mean(dim=1, keepdim=True)
x_centered = x - center
```

Shape flow:

``` text
x                         [B, T, D]
mean(dim=1, keepdim=True) [B, 1, D]
result                    [B, T, D]
```

## `[B, T, 1] * [B, T, D]`

This works:

``` text
weights  [B, T, 1]
features [B, T, D]
```

The scalar weight for each timestep broadcasts across all `D` features.

## Broadcasting checklist

``` text
[B, T, D] + [D]       ✓
[B, T, D] + [1, D]    ✓
[B, T, D] + [B, 1, D] ✓
[B, T, D] + [B, D]    ✗ in general
[B, T, D] * [B, T, 1] ✓
```

------------------------------------------------------------------------

# Lesson 7 --- Transpose, Permute, Strides, and Contiguous Memory

## `transpose`

Swaps two dimensions.

``` python
x.shape == [B, T, D]

y = x.transpose(1, 2)
```

Result:

``` text
[B, T, D]
→
[B, D, T]
```

## `.T`

For a 2-D tensor:

``` python
x.T
```

acts like:

``` python
x.transpose(0, 1)
```

For higher-dimensional ML code, explicit `transpose` or `permute` is
usually clearer.

## `permute`

Reorders all dimensions.

``` python
x.shape == [B, T, D]

y = x.permute(0, 2, 1)
```

Result:

``` text
[B, D, T]
```

Image example:

``` text
[C, H, W]
→ permute(1, 2, 0)
[H, W, C]
```

Batch image example:

``` text
[B, C, H, W]
→ permute(0, 2, 3, 1)
[B, H, W, C]
```

## Physical memory and stride

Suppose:

``` python
x = torch.tensor([
    [1, 2, 3],
    [4, 5, 6]
])
```

Logical shape:

``` text
[2, 3]
```

Underlying contiguous memory can be viewed as:

``` text
1 2 3 4 5 6
```

Its stride is:

``` text
(3, 1)
```

A stride tells you:

> How many underlying elements must be skipped to move one position
> along a dimension?

For `x[i, j]`, the storage offset is conceptually:

``` text
i * 3 + j * 1
```

## Transpose often changes metadata, not storage

``` python
y = x.transpose(0, 1)
```

Logically:

``` text
1 4
2 5
3 6
```

But PyTorch can still use the same underlying storage:

``` text
1 2 3 4 5 6
```

by changing shape and stride.

The new stride is typically:

``` text
(1, 3)
```

This is why transpose/permute operations can be cheap.

## Non-contiguous tensors

After transpose:

``` python
y.is_contiguous()
```

may be:

``` text
False
```

because logical iteration order no longer matches a standard contiguous
layout.

## `.contiguous()`

``` python
z = y.contiguous()
```

creates a contiguous representation when necessary.

A common pattern is:

``` python
x = x.permute(...).contiguous()
```

## Why this matters for `view`

`view()` requires a compatible storage layout.

After operations such as transpose:

``` python
y = x.transpose(0, 1)
```

this may fail:

``` python
y.view(...)
```

A traditional pattern is:

``` python
y.contiguous().view(...)
```

`reshape()` is more flexible because it can make a copy when required.

## Connection to CUDA

Memory layout affects performance.

GPUs generally benefit when nearby threads access nearby memory
locations, enabling efficient/coalesced memory accesses.

Therefore:

``` text
shape
stride
memory layout
contiguity
```

can influence:

``` text
memory access efficiency
→ memory bandwidth utilization
→ kernel performance
```

## Transformer example

Starting from:

``` text
x = [B, T, D]
```

where:

``` text
D = num_heads * head_dim
```

you may see:

``` python
x = x.view(B, T, num_heads, head_dim)
x = x.transpose(1, 2)
```

Shape flow:

``` text
[B, T, D]
→
[B, T, H, Dh]
→
[B, H, T, Dh]
```

This organizes tokens separately for each attention head.

------------------------------------------------------------------------

# Lesson 8 --- Autograd

PyTorch autograd automatically computes gradients.

The training pipeline is:

``` text
forward
→ computational graph
→ loss
→ backward()
→ gradients
→ optimizer.step()
→ updated parameters
```

## `requires_grad=True`

``` python
w = torch.tensor(
    2.0,
    requires_grad=True
)
```

means PyTorch should track operations involving `w` so gradients can
later be computed.

Example:

``` python
x = torch.tensor(3.0)
w = torch.tensor(2.0, requires_grad=True)

y = w * x
loss = (y - 10) ** 2

loss.backward()

print(w.grad)
# tensor(-24.)
```

Mathematically:

``` text
y = wx
L = (y - 10)^2
```

and:

``` text
dL/dw = 2(y - 10)x
       = 2(6 - 10)(3)
       = -24
```

## Computational graph

For:

``` python
y = w * x
loss = (y - 10) ** 2
```

think:

``` text
w ───┐
     multiply → y → subtract → square → loss
x ───┘
```

PyTorch records enough information about these operations to perform
reverse-mode automatic differentiation.

## `grad_fn`

Computed tensors may have:

``` python
y.grad_fn
loss.grad_fn
```

showing the operation that produced them.

## `backward`

``` python
loss.backward()
```

starts reverse-mode differentiation from the loss.

Gradients for leaf parameters are accumulated in:

``` python
parameter.grad
```

## Basic training loop

``` python
optimizer.zero_grad()

pred = model(x)

loss = criterion(pred, target)

loss.backward()

optimizer.step()
```

Meaning:

``` text
zero_grad()
→ clear previous gradients

forward
→ compute prediction and loss

backward()
→ compute parameter gradients

optimizer.step()
→ update parameters
```

## Why `zero_grad()` is necessary

PyTorch gradients accumulate by default.

If one backward pass produces:

``` text
w.grad = 3
```

and another produces:

``` text
4
```

without clearing the gradient, the result becomes:

``` text
w.grad = 7
```

This behavior is useful for deliberate gradient accumulation, but
ordinary training usually clears gradients every optimization step.

## `torch.no_grad`

For inference:

``` python
with torch.no_grad():
    output = model(x)
```

Operations inside the block are not recorded for backward, reducing
autograd-related memory and overhead.

## `torch.set_grad_enabled`

``` python
with torch.set_grad_enabled(fusion_grad):
    output = model(x)
```

If:

``` text
fusion_grad = True
```

gradient tracking is enabled.

If:

``` text
fusion_grad = False
```

gradient tracking is disabled for the block.

This is useful when the same code can run in training or inference-like
modes.

## `detach`

``` python
y_detached = y.detach()
```

returns a tensor detached from the current computational graph.

Conceptually:

``` text
x → y ─X→ detached branch
```

Gradients through the detached branch do not flow back through the
earlier graph.

A useful distinction:

``` text
no_grad()
→ do not record operations in this code region

detach()
→ disconnect this tensor from its existing graph
```

## Freezing parameters

``` python
for p in encoder.parameters():
    p.requires_grad_(False)
```

prevents those parameters from requiring gradients.

Common during fine-tuning.

## `model.train()` vs `model.eval()` is NOT autograd control

These control module behavior, especially modules such as Dropout and
BatchNorm.

### Dropout

During training:

``` text
model.train()
→ Dropout randomly drops activations
```

During evaluation:

``` text
model.eval()
→ Dropout is disabled
```

### BatchNorm

During training, BatchNorm normally:

``` text
uses current mini-batch statistics
+
updates running statistics
```

During evaluation, it normally:

``` text
uses stored running statistics
```

Therefore:

``` text
model.train() / model.eval()
→ control module behavior

grad enabled / torch.no_grad()
→ control autograd
```

They are independent switches.

Typical inference:

``` python
model.eval()

with torch.no_grad():
    output = model(x)
```

## Why training uses more memory

Training may need memory for:

``` text
parameters
+ gradients
+ optimizer states
+ saved activations
+ temporary buffers
```

Intermediate activations are often required later during backward.

------------------------------------------------------------------------

# Lesson 9 --- Device, GPU, and Dtype

Two fundamental tensor properties are:

``` text
device → where the tensor lives/computes
dtype  → how each element is represented
```

## Device

``` python
x = torch.tensor([1., 2., 3.])

print(x.device)
# cpu
```

Move to an NVIDIA GPU:

``` python
x = x.to("cuda")
```

or:

``` python
x = x.cuda()
```

Move back:

``` python
x = x.cpu()
```

## CPU and GPU tensors cannot generally be mixed directly

This is invalid:

``` python
x = torch.tensor([1., 2.])          # CPU
y = torch.tensor([3., 4.]).cuda()   # GPU

z = x + y
```

The operands must be on compatible devices.

## Common device pattern

``` python
device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

model = model.to(device)
x = x.to(device)
target = target.to(device)
```

Remember that:

``` python
x.to(device)
```

does not normally mutate `x` in place. Use:

``` python
x = x.to(device)
```

## Dtypes

Common types:

``` text
torch.float32   FP32
torch.float16   FP16
torch.bfloat16  BF16
torch.int64     integer / long
torch.bool      boolean
```

## FP32

``` text
32 bits = 4 bytes per element
```

One billion FP32 parameters require roughly:

``` text
1B × 4 bytes ≈ 4 GB
```

for parameter storage alone.

## FP16

``` text
16 bits = 2 bytes per element
```

Benefits can include:

-   less memory
-   less memory traffic
-   higher throughput on supported GPU hardware such as Tensor Cores

But FP16 has a smaller numerical range than FP32 and can suffer from
overflow/underflow.

## BF16

Also:

``` text
16 bits = 2 bytes
```

Compared with FP16, BF16 has a much larger dynamic range, close to FP32,
but less significand precision.

A useful mental model:

``` text
FP32 → stable, higher memory cost
FP16 → compact and fast, smaller range
BF16 → compact, large range, widely useful in modern training
```

## Dtype conversions

``` python
x = x.float()                  # float32
x = x.half()                   # float16
x = x.to(torch.bfloat16)       # bfloat16
x = x.long()                   # int64
```

`.to()` can change both device and dtype:

``` python
x = x.to(
    device="cuda",
    dtype=torch.float16
)
```

## Why class labels are often `long`

For class-index targets:

``` text
logits.shape = [B, C]
target.shape = [B]
```

targets might be:

``` text
[2, 0, 5, 1]
```

These are class IDs, so `nn.CrossEntropyLoss` commonly expects them as
integer (`torch.long`) indices.

## Boolean tensors

``` python
mask = scores > 0.5
```

produces:

``` text
dtype = torch.bool
```

and can be used for indexing or `torch.where`.

## Automatic Mixed Precision (AMP)

Mixed precision does not simply mean converting everything permanently
to FP16.

Example:

``` python
with torch.autocast(
    device_type="cuda",
    dtype=torch.float16
):
    pred = model(x)
    loss = criterion(pred, target)
```

Autocast lets suitable operations run in lower precision while
preserving higher precision where appropriate.

## Gradient scaling

FP16 gradients can become too small to represent reliably.

A common AMP training pattern is:

``` python
scaler = torch.amp.GradScaler("cuda")

with torch.autocast(
    device_type="cuda",
    dtype=torch.float16
):
    pred = model(x)
    loss = criterion(pred, target)

scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

Loss scaling reduces the risk of FP16 gradient underflow.

BF16 usually does not require loss scaling in the same way because of
its larger exponent range.

## CPU ↔ GPU transfer is not free

``` text
CPU RAM
   │
   │ PCIe / interconnect
   ▼
GPU VRAM
```

Repeatedly moving data between CPU and GPU can become a bottleneck.

Try to keep a computation on the GPU once its data has been transferred
there.

## Why tiny GPU operations may not be faster

GPU execution has overhead, including kernel launches.

GPUs are especially effective for large parallel workloads such as:

``` text
matrix multiplication
convolution
attention
```

rather than tiny isolated operations.

## Tensor → NumPy

A common pattern:

``` python
array = x.detach().cpu().numpy()
```

Meaning:

``` text
detach()
→ disconnect from autograd graph

cpu()
→ move GPU tensor to CPU if needed

numpy()
→ convert to NumPy representation
```

------------------------------------------------------------------------

# Lesson 10 --- `nn.Module` and Common Neural Network Layers

## `nn.Module`

Most PyTorch models inherit from:

``` python
nn.Module
```

Example:

``` python
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.fc = nn.Linear(10, 5)

    def forward(self, x):
        return self.fc(x)
```

`nn.Module` helps PyTorch manage:

-   parameters
-   submodules
-   device movement
-   train/eval state
-   state dictionaries
-   model saving/loading workflows

## `__init__` vs `forward`

`__init__` usually defines what components the model owns:

``` python
self.fc = nn.Linear(10, 5)
```

`forward` defines how data flows through those components:

``` python
def forward(self, x):
    return self.fc(x)
```

Use:

``` python
output = model(x)
```

rather than manually calling `model.forward(x)`, because
`nn.Module.__call__` performs PyTorch module machinery before invoking
`forward`.

## `nn.Linear`

``` python
layer = nn.Linear(
    in_features=10,
    out_features=5
)
```

Shape transformation:

``` text
[..., 10]
→
[..., 5]
```

A Linear layer implements:

``` text
y = x @ W.T + b
```

For:

``` python
nn.Linear(10, 5)
```

the parameter shapes are:

``` text
weight = [5, 10]
bias   = [5]
```

For a batch:

``` text
x       [B, 10]
W.T     [10, 5]
----------------
x @ W.T [B, 5]

bias    [5]
broadcast
----------------
output  [B, 5]
```

Linear operates on the **last dimension**, so:

``` text
[B, T, 128]
↓ Linear(128, 256)
[B, T, 256]
```

This is extremely common in Transformers.

## `nn.ReLU`

``` python
relu = nn.ReLU()
```

Implements:

``` text
ReLU(x) = max(0, x)
```

Example:

``` text
[-3, -1, 0, 2, 5]
→
[ 0,  0, 0, 2, 5]
```

It has no learnable parameters and does not change shape.

## Why nonlinear activations are needed

A stack of purely linear transformations can collapse into another
linear transformation.

Therefore:

``` text
Linear
→ ReLU
→ Linear
```

has greater expressive power than simply stacking Linear layers without
nonlinearities.

## Basic MLP

``` python
class MLP(nn.Module):
    def __init__(self):
        super().__init__()

        self.fc1 = nn.Linear(128, 256)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(256, 10)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x
```

Shape flow:

``` text
[B, 128]
→ [B, 256]
→ [B, 256]
→ [B, 10]
```

## `nn.Sequential`

For a simple sequential network:

``` python
self.net = nn.Sequential(
    nn.Linear(128, 256),
    nn.ReLU(),
    nn.Dropout(0.2),
    nn.Linear(256, 10)
)
```

Then:

``` python
def forward(self, x):
    return self.net(x)
```

## `nn.Dropout`

``` python
nn.Dropout(p=0.2)
```

During training it randomly drops activations as a regularization
mechanism.

During evaluation it is disabled.

Its behavior therefore depends on:

``` python
model.train()
model.eval()
```

## `nn.Embedding`

``` python
embedding = nn.Embedding(
    num_embeddings=10000,
    embedding_dim=256
)
```

Internally, the embedding table has shape:

``` text
[10000, 256]
```

Input token IDs:

``` text
[B, T]
```

become:

``` text
[B, T, 256]
```

Embedding is conceptually a learnable lookup table.

For a token ID `537`, the operation is conceptually similar to:

``` python
embedding.weight[537]
```

## `nn.Conv2d`

``` python
conv = nn.Conv2d(
    in_channels=3,
    out_channels=64,
    kernel_size=3
)
```

Input convention:

``` text
[B, C, H, W]
```

Output:

``` text
[B, 64, H_out, W_out]
```

The weight shape is:

``` text
[out_channels, in_channels, kernel_height, kernel_width]
```

For the example:

``` text
[64, 3, 3, 3]
```

and:

``` text
bias.shape = [64]
```

### Padding and stride

With:

``` text
kernel_size = 3
stride = 1
padding = 0
```

a spatial dimension of `224` becomes:

``` text
224 - 3 + 1 = 222
```

Using:

``` python
padding=1
```

with a 3×3, stride-1 convolution commonly preserves spatial size.

Using:

``` python
stride=2
```

often approximately halves spatial dimensions.

Typical CNN progression:

``` text
channels ↑
spatial resolution ↓
```

## `nn.BatchNorm2d`

Often used with CNN activations:

``` python
nn.BatchNorm2d(64)
```

A traditional block may look like:

``` text
Conv2d
→ BatchNorm2d
→ ReLU
```

Its train/eval behavior differs because it uses batch statistics during
training and stored running statistics during evaluation under the
standard configuration.

## `nn.LayerNorm`

Very common in Transformers:

``` python
norm = nn.LayerNorm(768)
```

For:

``` text
x = [B, T, 768]
```

the shape remains:

``` text
[B, T, 768]
```

LayerNorm commonly normalizes over the final feature dimension(s).

## `nn.Parameter`

To create your own learnable tensor:

``` python
class Model(nn.Module):
    def __init__(self):
        super().__init__()

        self.scale = nn.Parameter(
            torch.tensor(1.0)
        )

    def forward(self, x):
        return x * self.scale
```

`nn.Parameter` is automatically registered as a model parameter.

Contrast:

``` python
self.x = torch.tensor(1.0)
```

This is just a regular tensor attribute and is not automatically treated
as a trainable model parameter.

## `model.parameters()`

For:

``` python
model = nn.Linear(10, 5)
```

the model contains:

``` text
weight [5, 10]
bias   [5]
```

An optimizer can manage these:

``` python
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=1e-3
)
```

Then:

``` text
loss.backward()
→ weight.grad, bias.grad

optimizer.step()
→ updates weight and bias
```

------------------------------------------------------------------------

# Putting Everything Together

Consider:

``` python
class Classifier(nn.Module):
    def __init__(self):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 10)
        )

    def forward(self, x):
        return self.net(x)
```

For:

``` text
x = [32, 128]
```

shape flow:

``` text
[32, 128]
     │
     ▼
Linear(128, 256)
     │
     ▼
[32, 256]
     │
     ▼
ReLU
     │
     ▼
[32, 256]
     │
     ▼
Dropout
     │
     ▼
[32, 256]
     │
     ▼
Linear(256, 10)
     │
     ▼
[32, 10]
```

The final tensor contains 10 logits for each of 32 samples.

A training loop might be:

``` python
device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

model = Classifier().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()

model.train()

for x, target in dataloader:
    x = x.to(device)
    target = target.to(device)

    optimizer.zero_grad()

    logits = model(x)
    loss = criterion(logits, target)

    loss.backward()
    optimizer.step()
```

The conceptual pipeline is:

``` text
CPU data
   │
   │ transfer
   ▼
GPU tensors
   │
   ▼
model forward
   │
   ├── Linear: matmul + bias broadcasting
   ├── ReLU: elementwise operation
   └── Dropout: train-mode behavior
   │
   ▼
logits
   │
   ▼
loss
   │
   ▼
autograd backward
   │
   ▼
parameter gradients
   │
   ▼
optimizer.step()
   │
   ▼
updated parameters
```

------------------------------------------------------------------------

# Quick Reference Cheat Sheet

## Tensor creation

``` python
torch.tensor(data)
torch.zeros(...)
torch.ones(...)
torch.arange(...)
torch.linspace(...)
torch.rand(...)
torch.randn(...)
torch.zeros_like(x)
torch.ones_like(x)
```

## Tensor information

``` python
x.shape
x.ndim
x.numel()
x.dtype
x.device
x.requires_grad
x.stride()
x.is_contiguous()
```

## Shape operations

``` python
x.view(...)
x.reshape(...)
x.unsqueeze(dim)
x.squeeze(dim)
x.flatten(...)
x.expand(...)
x.repeat(...)
```

## Indexing

``` python
x[i]
x[i, j]
x[:, :10]
x[..., :2]
x[mask]

torch.where(...)
torch.index_select(...)
torch.gather(...)
torch.scatter(...)
```

## Combine / split

``` python
torch.cat(...)
torch.stack(...)
torch.split(...)
torch.chunk(...)
torch.unbind(...)
```

## Math / reductions

``` python
x.sum(dim=...)
x.mean(dim=...)
x.max(dim=...)
x.min(dim=...)
x.argmax(dim=...)
x.argmin(dim=...)
torch.linalg.vector_norm(x, dim=...)
x.clamp(...)
torch.sigmoid(x)
torch.softmax(x, dim=...)
torch.topk(x, k, dim=...)
```

## Dimension order / memory

``` python
x.transpose(...)
x.permute(...)
x.T
x.contiguous()
x.stride()
```

## Autograd

``` python
requires_grad=True
loss.backward()
x.grad
x.grad_fn
x.detach()

with torch.no_grad():
    ...

with torch.set_grad_enabled(condition):
    ...
```

## Device / dtype

``` python
x.to(device)
x.cuda()
x.cpu()

x.float()
x.half()
x.long()
x.to(torch.bfloat16)
```

## Model building

``` python
nn.Module
nn.Parameter
nn.Linear
nn.ReLU
nn.Dropout
nn.Sequential
nn.Embedding
nn.Conv2d
nn.BatchNorm2d
nn.LayerNorm
```

## Model modes

``` python
model.train()
model.eval()
```

Remember:

``` text
train/eval
→ module behavior

grad/no_grad
→ autograd behavior
```

------------------------------------------------------------------------

# Final Mental Model

When reading unfamiliar PyTorch code, trace it in this order:

``` text
1. What is the input shape?
        ↓
2. What does each dimension mean?
        ↓
3. What operation is applied?
        ↓
4. Which dimension does that operation affect?
        ↓
5. What is the output shape?
        ↓
6. Is broadcasting happening?
        ↓
7. Did the memory layout/stride change?
        ↓
8. What dtype and device are used?
        ↓
9. Is autograd tracking this operation?
        ↓
10. Which parameters will receive gradients?
```

If you can answer these questions, most PyTorch code becomes much easier
to reason about.

## Core shape patterns to recognize instantly

``` text
[B, T, D]
Batch × Time/Tokens × Features

[B, C, H, W]
Batch × Channels × Height × Width

[B, T, 2]
Batch × Time × XY coordinates

[B, C]
Batch × Classes
```

And always remember:

``` text
PyTorch does not understand semantic names such as
"batch", "time", "feature", or "coordinate."

It sees dimensions, sizes, strides, dtypes, devices, and operations.

Your job when reading ML code is to attach the semantic meaning to those dimensions.
```
