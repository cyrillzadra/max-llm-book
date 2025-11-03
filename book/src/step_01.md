# Step 01: Model configuration

<div class="note">
    Learn to define the GPT-2 model architecture parameters using configuration classes.
</div>

## Defining the model architecture

Before you can implement GPT-2, you need to define its architecture - the dimensions, layer counts, and structural parameters that determine how the model processes information.

In this step, you'll create `GPT2Config`, a class that holds all the architectural decisions for GPT-2. This class describes things like: embedding dimensions, number of transformer layers, and number of attention heads. These parameters define the shape and capacity of your model.

OpenAI trained the original GPT-2 model with specific parameters that you can see in the [config.json file](https://huggingface.co/openai-community/gpt2/blob/main/config.json) on Hugging Face. By using the exact same values, we can later load OpenAI's pretrained weights.

## Understanding the parameters

The GPT-2 configuration consists of seven key parameters. Each one controls a different aspect of the model's architecture:

- `vocab_size`: Size of the token vocabulary - the number of unique tokens the model can process (default: 50,257)
- `n_positions`: Maximum sequence length, also called the context window (default: 1,024)
- `n_embd`: Embedding dimension - the size of the hidden states that flow through the model (default: 768)
- `n_layer`: Number of transformer blocks stacked vertically (default: 12)
- `n_head`: Number of attention heads per layer, enabling parallel attention to different parts of the input (default: 12)
- `n_inner`: Dimension of the MLP intermediate layer, typically 4x the embedding dimension (default: 3,072)
- `layer_norm_epsilon`: Small constant for numerical stability in layer normalization (default: 1e-5)

These values define the _small_ GPT-2 model. OpenAI released four sizes (small, medium, large, XL), each with different configurations that scale up these parameters.

## Implementing the configuration

Now let's implement this yourself. You'll create the `GPT2Config` class using Python's [`@dataclass`](https://docs.python.org/3/library/dataclasses.html) decorator. Dataclasses reduce boilerplate.

Instead of writing `__init__` and defining each parameter manually, you just declare the fields with type hints and default values.

First, you'll need to import the dataclass decorator from the dataclasses module. Then you'll add the `@dataclass` decorator to the `GPT2Config` class definition.

The actual parameter values come from Hugging Face. You can get them in two ways:

- **Option 1**: Run `pixi run huggingface` to access these parameters programmatically from the Hugging Face `transformers` library.
- **Option 2**: Read the values directly from the [GPT-2 model card](https://huggingface.co/openai-community/gpt2/blob/main/config.json).

Once you have the values, replace each `None` in the `GPT2Config` class properties with the correct numbers from the configuration.

**Implementation** (`step_01.py`):

```python
{{#include ../../steps/step_01.py}}
```

### Validation

Run `pixi run s01` to verify your implementation matches the expected configuration.

**Reference**: `steps/step_01.py`

**Next**: In [Step 02](./step_02.md), you'll implement causal masking to prevent tokens from attending to future positions in autoregressive generation.
