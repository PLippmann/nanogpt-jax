import jax
import optax 
import jax.numpy as jnp

from model import GPT2Config, GPT2, SelfAttentionFlax, SelfAttention


# Create a sample input
batch_size = 2
seq_length = 16
vocab_size = 50304
config = GPT2Config(block_size=1024, vocab_size=vocab_size, n_embd=768, n_layer=12, n_head=12, dropout=0.1)

# Initialize random input tokens
rng = jax.random.PRNGKey(0)
inputs = jax.random.randint(rng, (batch_size, seq_length), 0, vocab_size)

# Initialize the model parameters
model = GPT2(config)
params = model.init(rng)['params']

def test_forward_pass():
    # Forward pass and check the output shape
    logits, _ = model.apply({'params': params}, inputs, rngs={'dropout': rng})
    assert logits.shape == (batch_size, seq_length, vocab_size), f"Expected logits shape (B, T, C), got {logits.shape}"
    print("Forward pass shape test passed. Output shape is correct.")

def test_gradient_flow():
    # Gradient flow test
    def compute_loss(params):
        logits, _ = model.apply({'params': params}, inputs, rngs={'dropout': rng})
        target = jax.random.randint(rng, (batch_size, seq_length), minval=0, maxval=vocab_size)
        loss = optax.softmax_cross_entropy_with_integer_labels(logits, target).mean()
        return loss

    loss, grads = jax.value_and_grad(compute_loss)(params)
    assert not jnp.isnan(loss), "Loss should not be NaN."
    assert all([jnp.all(jnp.isfinite(g)) for g in jax.tree_util.tree_leaves(grads)]), "All gradients should be finite."
    print("Gradient flow test passed.")

def test_masking():
    # Initialize test parameters for the attention layer
    attention_layer = SelfAttention(config=config)
    test_values = jax.random.normal(rng, (batch_size, seq_length, config.n_embd))
    attention_mask = jnp.tril(jnp.ones((seq_length, seq_length)))
    params_attn = attention_layer.init(rng, test_values, mask=attention_mask)['params']

    # Apply attention with masking
    attn_output = attention_layer.apply({'params': params_attn}, test_values, mask=attention_mask, rngs={'dropout': rng})
    assert attn_output.shape == (batch_size, seq_length, config.n_embd), (
        f"Expected output shape (B, T, C), got {attn_output.shape}"
    )
    print("Masking test passed.")

def test_logits():
    # Logits test
    logits, _ = model.apply({'params': params}, inputs, rngs={'dropout': rng})
    assert not jnp.isnan(logits).any(), "Logits should not contain NaN values."
    assert not jnp.isinf(logits).any(), "Logits should not contain Inf values."
    assert logits.max() < 1e2, "Logits values are unexpectedly high, indicating potential instability."
    print("Logits value and range tests passed.")

def test_embeddings():
    # Print token and position embdings
    token_embeddings = params['wte']['embedding']
    position_embeddings = params['wpe']['embedding']
    print("Token Embeddings Sample:", token_embeddings[:5])
    print("Position Embeddings Sample:", position_embeddings[:5])

    # Check embedding shapes and ranges
    assert token_embeddings.shape == (config.vocab_size, config.n_embd), "Token embeddings shape mismatch."
    assert position_embeddings.shape == (config.block_size, config.n_embd), "Position embeddings shape mismatch."
    assert jnp.all(jnp.isfinite(token_embeddings)), "Token embeddings contain non-finite values."
    assert jnp.all(jnp.isfinite(position_embeddings)), "Position embeddings contain non-finite values."
    print("Embedding checks passed.")

def test_deterministic_mode():
    # Test model in deterministic mode (inference)
    logits_deterministic, _ = model.apply({'params': params}, inputs, rngs={'dropout': rng}, deterministic=True)
    assert logits_deterministic.shape == (batch_size, seq_length, vocab_size), "Output shape in deterministic mode is incorrect."
    print("Deterministic mode shape test passed.")

    # Test model in non-deterministic mode (training)
    logits_non_deterministic, _ = model.apply({'params': params}, inputs, rngs={'dropout': rng}, deterministic=False)
    assert logits_non_deterministic.shape == (batch_size, seq_length, vocab_size), "Output shape in non-deterministic mode is incorrect."
    print("Non-deterministic mode shape test passed.")

def test_parameter_count():
    # Helper func to calculate param count
    def calculate_expected_params(vocab_size, d_model, ff_dim, n_layers, seq_len):
        # Embedding layers
        embed_params = (vocab_size * d_model) + (seq_len * d_model)  # Token + Position embeddings
        
        # Per block parameters (Self-Attention + MLP + LayerNorms)
        attn_params = (4 * d_model * d_model + 4 * d_model)  # 3 * Dense layers (query, key, value) + 1 for output
        mlp_params = (ff_dim * d_model + ff_dim) + (d_model * ff_dim + d_model)  # MLP layers
        block_params = attn_params + mlp_params + 2 * (2 * d_model)  # Attention + MLP + LayerNorms
        
        # Total for all layers
        total_params = embed_params + (n_layers * block_params)
        
        return total_params

    # Retrieve actual number of parameters from the model
    actual_params = sum(p.size for p in jax.tree_util.tree_leaves(params))

    # Calculate expected number of parameters
    expected_params = calculate_expected_params(vocab_size=config.vocab_size,
                                                d_model=config.n_embd,
                                                ff_dim=config.n_embd * 4,
                                                n_layers=config.n_layer,
                                                seq_len=config.block_size)

    # Compare the actual vs expected
    print(f"Actual # of params: {actual_params}, Expected: {expected_params}")
    # Assert parameter counts match to the nearest million
    assert round(actual_params, -6) == round(expected_params, -6), "Parameter count mismatch."

def test_generate_function():
    rng = jax.random.PRNGKey(0)
    initial_tokens = jnp.zeros((1, 1), dtype=jnp.uint16)
    model = GPT2(config)
    params = model.init(rng)['params']

    # Generate some tokens
    generated_tokens = model.apply({'params': params}, initial_tokens, rngs={'dropout': rng}, method=model.generate, max_new_tokens=10, temperature=1.0, top_k=10)
    
    # Check the generated output
    print("Generated Tokens:", generated_tokens)
    assert generated_tokens.shape[1] == 11, f"Expected sequence length of 11, got {generated_tokens.shape[1]}"
    assert jnp.all(jnp.logical_and(0 <= generated_tokens, generated_tokens < config.vocab_size)), "Generated tokens are out of vocabulary range"
    print("Generate function test passed.")

# Run all tests
test_forward_pass()
test_gradient_flow()
test_masking()
test_logits()
test_embeddings()
test_deterministic_mode()
test_parameter_count()
test_generate_function()