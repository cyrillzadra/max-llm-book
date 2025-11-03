# Step 03: Layer normalization

<div class="note">
    Learn to implement layer normalization for stabilizing neural network training.
</div>

## Implementing layer normalization

In this step you'll create the `LayerNorm` class. This normalizes activations across the feature dimension to stabilize training. The process: compute mean and variance across features, normalize by subtracting mean and dividing by standard deviation, then scale and shift using learned weight and bias parameters.

Unlike batch normalization, [layer normalization](https://arxiv.org/abs/1607.06450) works independently for each example. This makes it ideal for transformers because it doesn't depend on batch size or require tracking running statistics for inference.

## Understanding layer normalization

Layer normalization normalizes across the feature dimension (the last dimension) independently for each example. It learns two parameters per feature: weight (gamma) for scaling and bias (beta) for shifting.

**The normalization formula**:

```math
output = weight * (x - mean) / sqrt(variance + epsilon) + bias
```

GPT-2 applies layer normalization before the attention and MLP blocks in each transformer layer. The epsilon value (typically 1e-5) prevents division by zero.

<div class="note">
<div class="title">MAX operations</div>

You'll use the following MAX operations to complete this task:

**Tensor initialization**:
- [`Tensor.ones()`](https://docs.modular.com/max/api/python/experimental/tensor#max.experimental.tensor.Tensor.ones): Creates tensor filled with 1.0 values
- [`Tensor.zeros()`](https://docs.modular.com/max/api/python/experimental/tensor#max.experimental.tensor.Tensor.zeros): Creates tensor filled with 0.0 values

**Layer normalization**:
- [`F.layer_norm()`](https://docs.modular.com/max/api/python/experimental/functional#max.experimental.functional.layer_norm): Applies layer normalization with parameters: `input`, `gamma` (weight), `beta` (bias), and `epsilon`

</div>

## Implementing the class

You'll implement the `LayerNorm` class in several steps:

1. **Import required modules**: Import `functional as F` and `Tensor` from MAX libraries.

2. **Initialize weight parameter**: Use `Tensor.ones([dim])` to create the weight parameter (gamma). Initialized to ones so initial normalization is identity.

3. **Initialize bias parameter**: Use `Tensor.zeros([dim])` to create the bias parameter (beta). Initialized to zeros so initial normalization has no shift.

4. **Apply layer normalization**: Use `F.layer_norm(x, gamma=self.weight, beta=self.bias, epsilon=self.eps)` in the forward pass.

**Implementation** (`step_03.py`):

```python
{{#include ../../steps/step_03.py}}
```

### Validation

Run `pixi run s03` to verify your implementation.

**Reference**: `solutions/solution_03.py`

**Next**: In [Step 04](./step_04.md), you'll implement the feed-forward network (MLP) with GELU activation used in each transformer block.
