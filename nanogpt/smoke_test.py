import numpy as np

from train_gpt2 import GPT2Config, GPT2

# Create a sample input
batch_size = 2
seq_length = 16
vocab_size = 50304
config = GPT2Config(block_size=1024, vocab_size=vocab_size, n_embd=768, n_layer=2, n_head=12, dropout=0.1)

# Initialize random input tokens
rng = jax.random.PRNGKey(0)
inputs = jax.random.randint(rng, (batch_size, seq_length), 0, vocab_size)

model = GPT2(config)

# Initialize the model parameters using only the random number generator (rng)
params = model.init(rng)['params']  # Remove 'inputs' from the arguments

# Run forward pass and check the output shape
logits, _ = model.apply({'params': params}, inputs, rngs={'dropout': rng})  # Pass inputs to the apply method

assert logits.shape == (batch_size, seq_length, vocab_size), f"Expected logits shape (B, T, V), got {logits.shape}"
print("Smoke test passed. Output shape is correct.")