# Step 12: Stacking transformer blocks

<div class="note">
    Learn to stack 12 transformer blocks with embeddings and final normalization to create the complete GPT-2 model.
</div>

## Implementing model stacking

In this step you'll create the `GPT2Model` class that stacks 12 transformer blocks with embeddings to form the complete GPT-2 architecture.

The flow is: token IDs → token embeddings (lookup table), add position embeddings, pass through 12 transformer blocks in sequence, apply final layer normalization, output contextualized representations for each token.

Each block processes the output of the previous block, progressively refining representations from simple embeddings to rich contextual understanding. Stacking multiple layers creates a hierarchy where early blocks learn surface patterns, middle blocks capture syntax, and later blocks encode semantics.

## Understanding the stacking

**Sequential Composition**:
- [`Sequential(*modules)`](https://docs.modular.com/max/api/python/nn/module_v3#max.nn.module_v3.Sequential) chains modules
- Applies each module in order: `output = module_n(...module_2(module_1(input)))`
- For GPT-2: `Sequential(block1, block2, ..., block12)`
- Convenient for stacking identical structures

**Embedding Combination**:
- Token embeddings: `wte(input_ids)` maps each token ID to 768-dim vector
- Position embeddings: `wpe(positions)` maps each position to 768-dim vector
- Combined: `tok_embeds + pos_embeds` (element-wise addition)
- Both contribute to initial representation

**Position Encoding**:
- Create positions: `Tensor.arange(seq_length, dtype=input_ids.dtype, device=input_ids.device)`
- Must match input dtype and device for compatibility
- Positions are [0, 1, 2, ..., seq_length-1]
- Same positions used for all examples in batch (broadcast)

**Final Layer Normalization**:
- Applied after all transformer blocks
- Stabilizes the output distribution
- Called `ln_f` (layer norm final) in HuggingFace
- Essential for consistent output scale

**Model Hyperparameters**:
- `config.n_layer = 12`: Number of transformer blocks
- `config.n_embd = 768`: Embedding/hidden dimension
- `config.vocab_size = 50257`: Vocabulary size
- `config.n_positions = 1024`: Maximum sequence length

<div class="note">
<div class="title">MAX operations</div>

You'll use the following MAX operations to complete this task:

**Module composition**:
- [`Sequential(*modules)`](https://docs.modular.com/max/api/python/nn/module_v3#max.nn.module_v3.Sequential): Chains transformer blocks in sequence

**Embeddings**:
- [`Embedding(num_embeddings, dim)`](https://docs.modular.com/max/api/python/nn/module_v3#max.nn.module_v3.Embedding): Token and position embeddings

**Position generation**:
- [`Tensor.arange(seq_length, dtype, device)`](https://docs.modular.com/max/api/python/experimental/tensor#max.experimental.tensor.Tensor.arange): Creates position indices

</div>

## Implementation tasks

1. **Import Required Modules** (Lines 13-18):
   - Import `Tensor` from `max.experimental.tensor`
   - Import `Embedding, Module, Sequential` from `max.nn.module_v3`
   - Import `GPT2Config` from `solutions.solution_01`
   - Import `LayerNorm` from `solutions.solution_10`
   - Import `GPT2Block` from `solutions.solution_11`

2. **Create Embeddings** (Lines 34-39):
   - Token embeddings: `Embedding(config.vocab_size, dim=config.n_embd)`
   - Position embeddings: `Embedding(config.n_positions, dim=config.n_embd)`
   - Store as `self.wte` and `self.wpe`

3. **Stack Transformer Blocks** (Lines 42-44):
   - Use `Sequential(*(GPT2Block(config) for _ in range(config.n_layer)))`
   - Generator creates 12 identical blocks
   - Sequential chains them together

4. **Create Final Layer Norm** (Lines 47-48):
   - `LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)`
   - Store as `self.ln_f`

5. **Implement Forward Pass** (Lines 61-87):
   - Extract shape: `batch_size, seq_length = input_ids.shape`
   - Token embeddings: `tok_embeds = self.wte(input_ids)`
   - Position indices: `Tensor.arange(seq_length, dtype=input_ids.dtype, device=input_ids.device)`
   - Position embeddings: `pos_embeds = self.wpe(position_indices)`
   - Combine: `x = tok_embeds + pos_embeds`
   - Apply blocks: `x = self.h(x)`
   - Final norm: `x = self.ln_f(x)`
   - Return `x`

**Implementation** (`step_12.py`):

```python
{{#include ../../steps/step_12.py}}
```

### Validation

Run `pixi run s12` to verify your implementation.

**Reference**: `solutions/solution_12.py`

**Next**: In [Step 13](./step_13.md), you'll add the language modeling head that projects hidden states to vocabulary logits for text generation.
