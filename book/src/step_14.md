# Step 14: Text generation

<div class="note">
    Learn to implement autoregressive text generation with sampling and temperature control.
</div>

## Implementing text generation

You've built the embeddings, attention mechanisms, transformer blocks, and language model head. Now you'll implement the generation loop that actually produces text.

The model generates text one token at a time, using each prediction as input for the next. Start with prompt `[15496, 995]` ("Hello world"), predict next token `[15496, 995, 318]` ("Hello world is"), predict next token `[15496, 995, 318, 257]` ("Hello world is a"), and repeat until reaching the desired length.

You'll implement temperature control (adjusting randomness) and sampling (choosing from the probability distribution) to control generation quality and creativity. Temperature scales logits before softmax. Lower values make high-probability tokens more likely (focused), while higher values flatten the distribution (diverse).

## Understanding generation techniques

The generation loop repeats: predict next token, append to sequence, repeat until reaching `max_new_tokens`. Each iteration requires a full forward pass through the model. The input sequence grows from `[batch, seq_length]` to `[batch, seq_length + 1]` and so on.

The model outputs `[batch, seq_length, vocab_size]` logits. To get the next token prediction, extract the last position: `logits[0, -1, :]` giving shape `[vocab_size]`. These are raw logits (unnormalized scores), not probabilities.

Temperature scaling controls generation randomness using the formula `scaled_logits = logits / temperature`. Lower values (like 0.7) sharpen the distribution making high-probability tokens more likely. Higher values (like 1.2) flatten the distribution making generation more diverse. Temperature 1.0 uses the original distribution.

You can generate tokens two ways: greedy decoding selects the highest-probability token using [`F.argmax(logits)`](https://docs.modular.com/max/api/python/experimental/functional#max.experimental.functional.argmax). This is fast and deterministic but often produces repetitive text. Sampling randomly selects tokens according to their probability distribution, producing more diverse and creative text. Most practical generation uses sampling with temperature control.

For sampling, convert logits to probabilities with `F.softmax(logits)`, transfer to CPU, convert to NumPy with `np.from_dlpack(probs)`, then sample with `np.random.choice(len(probs), p=probs)`. NumPy is used because MAX doesn't have built-in sampling yet.

After generating a token, append it to the sequence using [`F.concat([seq, new_token], axis=1)`](https://docs.modular.com/max/api/python/experimental/functional#max.experimental.functional.concat). The new token must be reshaped to 2D `[1, 1]` before concatenation.

<div class="note">
<div class="title">MAX operations</div>

You'll use the following MAX operations to complete this task:

**Probability operations**:
- [`F.softmax(logits)`](https://docs.modular.com/max/api/python/experimental/functional#max.experimental.functional.softmax): Converts logits to probabilities
- [`F.argmax(logits)`](https://docs.modular.com/max/api/python/experimental/functional#max.experimental.functional.argmax): Selects highest-probability token (greedy)

**Sequence building**:
- [`F.concat([seq, new_token], axis=1)`](https://docs.modular.com/max/api/python/experimental/functional#max.experimental.functional.concat): Appends token to sequence
- [`Tensor.constant(value, dtype, device)`](https://docs.modular.com/max/api/python/experimental/tensor#max.experimental.tensor.Tensor.constant): Creates scalar tensors

**NumPy interop**:
- `probs.to(CPU())`: Transfers tensor to CPU
- `np.from_dlpack(probs)`: Converts MAX tensor to NumPy for sampling

</div>

## Implementation tasks

1. **Import Required Modules** (Lines 13-17):
   - Import `numpy as np`
   - Import `CPU` from `max.driver`
   - Import `DType` from `max.dtype`
   - Import `functional as F` from `max.experimental`
   - Import `Tensor` from `max.experimental.tensor`

2. **Get Model Logits** (Lines 32-37):
   - Call `logits = model(input_ids)`
   - Extract last position: `next_token_logits = logits[0, -1, :]`

3. **Apply Temperature and Sample** (Lines 42-54):
   - Create temperature tensor: `Tensor.constant(temperature, dtype=..., device=...)`
   - Scale logits: `next_token_logits / temp_tensor`
   - Get probabilities: `F.softmax(next_token_logits)`
   - Convert to NumPy: `np.from_dlpack(probs.to(CPU()))`
   - Sample: `np.random.choice(len(probs_np), p=probs_np)`
   - Convert back: `Tensor.constant(next_token_id, dtype=DType.int64, device=...)`

4. **Implement Greedy Decoding** (Lines 57-58):
   - If not sampling: `next_token_tensor = F.argmax(next_token_logits)`

5. **Implement Generation Loop** (Lines 77-94):
   - Initialize: `generated_tokens = input_ids`
   - Loop `max_new_tokens` times
   - Generate next token: `generate_next_token(model, generated_tokens, ...)`
   - Reshape: `next_token.reshape([1, -1])`
   - Concatenate: `F.concat([generated_tokens, next_token_2d], axis=1)`

**Implementation** (`step_14.py`):

```python
{{#include ../../steps/step_14.py}}
```

### Validation

Run `pixi run s14` to verify your implementation.

**Reference**: `solutions/solution_14.py`

## What you've built

You've completed all 14 steps and built a complete GPT-2 model from scratch using MAX. You now have a working implementation of:

**Core components**:
- Model configuration and architecture definition
- Causal masking for autoregressive generation
- Layer normalization for training stability
- Feed-forward networks with GELU activation
- Token and position embeddings
- Multi-head self-attention
- Residual connections and transformer blocks
- Language model head for next-token prediction
- Text generation with temperature and sampling

Your model loads OpenAI's pretrained GPT-2 weights and generates text. You understand how every component works, from the low-level tensor operations to the high-level architecture decisions. This knowledge transfers directly to other transformer models like BERT, GPT-3, and beyond.
