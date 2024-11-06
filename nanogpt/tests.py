import jax
import optax 
import jax.numpy as jnp

from train_gpt2 import GPT2Config, GPT2, SelfAttention

# Create a sample input
batch_size = 2
seq_length = 16
vocab_size = 50304
config = GPT2Config(block_size=1024, vocab_size=vocab_size, n_embd=768, n_layer=2, n_head=12, dropout=0.1)

# Initialize random input tokens
rng = jax.random.PRNGKey(0)
inputs = jax.random.randint(rng, (batch_size, seq_length), 0, vocab_size)

# Initialize the model parameters
model = GPT2(config)
params = model.init(rng)['params']

# 1. Forward pass and check the output shape
logits, _ = model.apply({'params': params}, inputs, rngs={'dropout': rng})

assert logits.shape == (batch_size, seq_length, vocab_size), f"Expected logits shape (B, T, C), got {logits.shape}"
print("Forward pass shape test passed. Output shape is correct.")

# 2. Gradient flow test
def compute_loss(params):
    logits, _ = model.apply({'params': params}, inputs, rngs={'dropout': rng})
    target = jax.random.randint(rng, (batch_size, seq_length), minval=0, maxval=vocab_size)
    loss = optax.softmax_cross_entropy_with_integer_labels(logits, target).mean()
    return loss

loss, grads = jax.value_and_grad(compute_loss)(params)
assert not jnp.isnan(loss), "Loss should not be NaN."
assert all([jnp.all(jnp.isfinite(g)) for g in jax.tree_util.tree_leaves(grads)]), "All gradients should be finite."
print("Gradient flow test passed.")

# 3. Masking test
# Initialize test parameters for the attention layer
attention_layer = SelfAttention(config=config)
test_values = jax.random.normal(rng, (batch_size, seq_length, config.n_embd))
attention_mask = jnp.tril(jnp.ones((seq_length, seq_length)))
params_attn = attention_layer.init(rng, test_values, mask=attention_mask)['params']

# Apply attention with masking
attn_output = attention_layer.apply({'params': params_attn}, test_values, mask=attention_mask)
assert attn_output.shape == (batch_size, seq_length, config.n_embd), (
    f"Expected output shape (B, T, C), got {attn_output.shape}"
)
print("Masking test passed.")

# 4. Logits test
assert not jnp.isnan(logits).any(), "Logits should not contain NaN values."
assert not jnp.isinf(logits).any(), "Logits should not contain Inf values."
print("Logits value range test passed.")

# 5. Print token and position embdings
token_embeddings = params['wte']['embedding']
position_embeddings = params['wpe']['embedding']
print("Token Embeddings Sample:", token_embeddings[:5])
print("Position Embeddings Sample:", position_embeddings[:5])