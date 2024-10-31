#pip install jax optax tiktoken
import jax
import jax.numpy as jnp
import jax.lax as lax
from jax import random, grad, jit, vmap
import optax
import numpy as np
from typing import NamedTuple, Optional
import tiktoken


class TransformerConfig(NamedTuple):
    """Configuration for Transformer model.
       
       vocab_size: Size of the vocabulary
       block_size: Maximum sequence length
       n_embd: Embedding dimension
       n_head: Number of attention heads
       n_layer: Number of transformer layers
       dropout: Dropout probability
       batch_size: Training batch size
       learning_rate: Learning rate for optimization
       max_iters: Maximum number of training iterations
       eval_interval: Number of steps between evaluations
       eval_iters: Number of evaluation iterations
       init_scale: Scale for weight initialization
    """
    vocab_size: int = 50304  # GPT-2 vocabulary size
    block_size: int = 256
    n_embd: int = 384
    n_head: int = 6
    n_layer: int = 6
    dropout: float = 0.2
    batch_size: int = 32
    learning_rate: float = 3e-4
    max_iters: int = 2000
    eval_interval: int = 500
    eval_iters: int = 200
    init_scale: float = 0.02

    def __post_init__(self):
        assert self.n_embd % self.n_head == 0, "n_embd must be divisible by n_head"
        assert self.dropout >= 0 and self.dropout <= 1, "dropout must be between 0 and 1 inclusive"

class TransformerParams(NamedTuple):
    token_embedding: jnp.ndarray
    position_embedding: jnp.ndarray
    layer_norms: list
    attention_weights: list
    attention_projections: list
    mlp_weights: list
    final_layer_norm: jnp.ndarray
    lm_head: jnp.ndarray

@jit
def get_sequence(data, start_idx):
    """Extract a sequence of tokens and the corresponding targets from the data."""
    x_seq = lax.dynamic_slice(data, (start_idx,), (config.block_size,))
    y_seq = lax.dynamic_slice(data, (start_idx + 1,), (config.block_size,))
    
    return x_seq, y_seq

@jit
def get_batch(data, rng_key):
    """Generate a batch of data for training or validation."""
    data_size = data.shape[0]
    max_start_idx = data_size - config.block_size
    ix = jax.random.randint(rng_key, (config.batch_size,), 0, max_start_idx)
    x, y = jax.vmap(lambda idx: get_sequence(data, idx))(ix)
    
    return x, y

def init_params(rng):
    """Initialize model parameters."""
    rngs = jax.random.split(rng, 8)
    
    token_embedding = jax.random.normal(rngs[0], (config.vocab_size, config.n_embd)) * 0.02
    position_embedding = jax.random.normal(rngs[1], (config.block_size, config.n_embd)) * 0.02
    
    layer_norms, attention_weights, attention_projections, mlp_weights = [], [], [], []
    for _ in range(config.n_layer):
        # Initialize layer norm parameters with scale and bias
        layer_norms.append(jnp.ones((config.n_embd,)))
        attention_weights.append(jax.random.normal(rngs[2], (config.n_embd, 3 * config.n_embd)) * 0.02)
        attention_projections.append(jax.random.normal(rngs[3], (config.n_embd, config.n_embd)) * 0.02)
        mlp_weights.append({
            'c_fc': jax.random.normal(rngs[4], (config.n_embd, 4 * config.n_embd)) * 0.02,
            'c_proj': jax.random.normal(rngs[5], (4 * config.n_embd, config.n_embd)) * 0.02
        })

    # Initialize final layer norm with scale and bias
    final_layer_norm = jnp.ones((config.n_embd,))
    lm_head = jax.random.normal(rngs[6], (config.n_embd, config.vocab_size)) * 0.02
    
    return TransformerParams(
        token_embedding=token_embedding,
        position_embedding=position_embedding,
        layer_norms=layer_norms,
        attention_weights=attention_weights,
        attention_projections=attention_projections,
        mlp_weights=mlp_weights,
        final_layer_norm=final_layer_norm,
        lm_head=lm_head
    )

@jit
def layer_norm(x, weight):
    mean = jnp.mean(x, axis=-1, keepdims=True)
    variance = jnp.var(x, axis=-1, keepdims=True)
    
    return weight * (x - mean) / jnp.sqrt(variance + 1e-5)

@jit
def attention(q, k, v, mask=None):
    """Compute attention scores and perform weighted aggregation."""
    head_dim = q.shape[-1]
    attn = jnp.einsum('bhid,bhjd->bhij', q, k) / jnp.sqrt(head_dim)
    
    if mask is not None:
        attn = jnp.where(mask == 0, float('-inf'), attn)
    
    attn = jax.nn.softmax(attn, axis=-1)
    out = jnp.einsum('bhij,bhjd->bhid', attn, v)
    
    return out

def forward(params, x, key, training=False):
    """Forward pass through the transformer model."""
    b, t = x.shape
    head_size = config.n_embd // config.n_head
    token_emb = params.token_embedding[x]
    pos = jnp.arange(t)
    pos_emb = params.position_embedding[pos]
    x = token_emb + pos_emb
    mask = jnp.tril(jnp.ones((t, t)))
    
    for i in range(config.n_layer):
        # Layer normalization with scale and bias
        ln1 = layer_norm(x, params.layer_norms[i])
        qkv = jnp.dot(ln1, params.attention_weights[i])
        q, k, v = jnp.split(qkv, 3, axis=-1)
        q = q.reshape(b, t, config.n_head, head_size).transpose(0, 2, 1, 3)
        k = k.reshape(b, t, config.n_head, head_size).transpose(0, 2, 1, 3)
        v = v.reshape(b, t, config.n_head, head_size).transpose(0, 2, 1, 3)
        
        att = attention(q, k, v, mask)
        att = att.transpose(0, 2, 1, 3).reshape(b, t, config.n_embd)
        att = jnp.dot(att, params.attention_projections[i])
        x = x + att
        
        ln2 = layer_norm(x, params.layer_norms[i])
        mlp = jnp.dot(ln2, params.mlp_weights[i]['c_fc'])
        mlp = jax.nn.gelu(mlp)
        mlp = jnp.dot(mlp, params.mlp_weights[i]['c_proj'])
        x = x + mlp
        
        if training:
            dropout_key, key = jax.random.split(key)
            x = jnp.where(jax.random.uniform(dropout_key, x.shape) > config.dropout, x, 0)
    
    # Final layer normalization
    x = layer_norm(x, params.final_layer_norm)
    logits = jnp.dot(x, params.lm_head)
    
    return logits

@jit
def loss_fn(params, batch, key):
    """Compute loss over a batch of data."""
    x, y = batch
    logits = forward(params, x, key)
    loss = optax.softmax_cross_entropy_with_integer_labels(
        logits.reshape(-1, logits.shape[-1]),
        y.reshape(-1)
    )
    
    return jnp.mean(loss)

@jit
def train_step(params, opt_state, batch, key):
    """Perform a single training step."""
    loss, grads = jax.value_and_grad(loss_fn)(params, batch, key)
    updates, opt_state = optimizer.update(grads, opt_state)
    params = optax.apply_updates(params, updates)
    
    return params, opt_state, loss

@jit
def generate_step(params, x, key):
    """Single forward step with temperature-adjusted sampling."""
    logits = forward(params, x[None, :], key, training=False)[0]
    next_token_logits = logits[-1, :]  # Get logits for the last position
    # Apply temperature sampling
    temperature = 0.8  # Adjust this value to control randomness (lower = more focused)
    next_token_logits = next_token_logits / temperature
    # Sample from the distribution
    next_token = jax.random.categorical(key, next_token_logits)
    
    return next_token

def generate_text(params, prompt, max_new_tokens=100, temperature=0.8):
    """Generate text from a prompt."""
    # Encode the prompt
    input_ids = jnp.array(enc.encode(prompt))
    
    # Initialize generation
    generated = list(input_ids)
    key = jax.random.PRNGKey(0)
    
    # Generate tokens
    for _ in range(max_new_tokens):
        # Get the window of tokens that fits our model's context size
        x = jnp.array(generated[-config.block_size:] if len(generated) > config.block_size else generated)
        
        # Get next token
        key, subkey = jax.random.split(key)
        next_token = generate_step(params, x, subkey)
        
        # Append to generated sequence
        generated.append(next_token)
    
    # Decode the generated tokens
    return enc.decode(generated)

if __name__ == "__main__":
    # Initialize configuration
    config = TransformerConfig()

    # RNG setup
    rng = jax.random.PRNGKey(0)
    rng, init_rng = jax.random.split(rng)

    # Download data (wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt)
    with open('input.txt', 'r', encoding='utf-8') as f:
        text = f.read()
        print(text[:20])

    # Tokenizer setup
    enc = tiktoken.get_encoding("gpt2")
    assert enc.decode(enc.encode("hello world")) == "hello world"

    # Train and test splits
    data = np.array(enc.encode(text))
    train_data = jnp.array(data[:int(0.9 * len(data))])
    val_data = jnp.array(data[int(0.9 * len(data)):])
    print(data.shape, train_data.shape, val_data.shape, data[:10])
    
    # Initialize model and optimizer
    params = init_params(init_rng)
    optimizer = optax.adam(config.learning_rate)
    opt_state = optimizer.init(params)

    # Main training loop
    for iter in range(config.max_iters):
        rng, split_key = jax.random.split(rng)
        batch = get_batch(train_data, split_key)
        params, opt_state, loss = train_step(params, opt_state, batch, split_key)
        
        if iter % config.eval_interval == 0:
            losses = []
            for _ in range(config.eval_iters):
                rng, split_key = jax.random.split(rng)
                batch = get_batch(val_data, split_key)
                losses.append(loss_fn(params, batch, split_key))
            print(f"step {iter}: train loss {loss:.4f}, val loss {jnp.mean(jnp.array(losses)):.4f}")

    # Example usage:
    prompt = "Behold,"
    print("\nGenerating text from prompt:", prompt)
    print("-" * 40)
    generated_text = generate_text(params, prompt, max_new_tokens=200)
    print(generated_text)