import jax
import jax.numpy as jnp
import jax.lax as lax
from jax import random, grad, jit, vmap
import optax
import numpy as np
from typing import NamedTuple, Optional
import tiktoken

# hyperparameters
batch_size = 32
block_size = 256
max_iters = 5000
eval_interval = 500
learning_rate = 3e-4
eval_iters = 200
n_embd = 384
n_head = 6
n_layer = 6
dropout = 0.2

# RNG setup
rng = jax.random.PRNGKey(0)
rng, init_rng = jax.random.split(rng)

# Download data
#!wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
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
    x_seq = lax.dynamic_slice(data, (start_idx,), (block_size,))
    y_seq = lax.dynamic_slice(data, (start_idx + 1,), (block_size,))
    return x_seq, y_seq

@jit
def get_batch(data, rng_key):
    data_size = data.shape[0]
    max_start_idx = data_size - block_size
    ix = jax.random.randint(rng_key, (batch_size,), 0, max_start_idx)
    x, y = jax.vmap(lambda idx: get_sequence(data, idx))(ix)
    return x, y

def init_params(rng, vocab_size):
    rngs = jax.random.split(rng, 8)
    
    token_embedding = jax.random.normal(rngs[0], (vocab_size, n_embd)) * 0.02
    position_embedding = jax.random.normal(rngs[1], (block_size, n_embd)) * 0.02
    
    layer_norms, attention_weights, attention_projections, mlp_weights = [], [], [], []
    for _ in range(n_layer):
        layer_norms.append(jnp.ones((n_embd,)))
        attention_weights.append(jax.random.normal(rngs[2], (n_embd, 3 * n_embd)) * 0.02)
        attention_projections.append(jax.random.normal(rngs[3], (n_embd, n_embd)) * 0.02)
        mlp_weights.append({
            'c_fc': jax.random.normal(rngs[4], (n_embd, 4 * n_embd)) * 0.02,
            'c_proj': jax.random.normal(rngs[5], (4 * n_embd, n_embd)) * 0.02
        })
    
    final_layer_norm = jnp.ones((n_embd,))
    lm_head = jax.random.normal(rngs[6], (n_embd, vocab_size)) * 0.02
    
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
    head_dim = q.shape[-1]
    attn = jnp.einsum('bhid,bhjd->bhij', q, k) / jnp.sqrt(head_dim)
    
    if mask is not None:
        attn = jnp.where(mask == 0, float('-inf'), attn)
    
    attn = jax.nn.softmax(attn, axis=-1)
    out = jnp.einsum('bhij,bhjd->bhid', attn, v)
    
    return out

def forward(params, x, key, training=False):
    b, t = x.shape
    head_size = n_embd // n_head
    token_emb = params.token_embedding[x]
    pos = jnp.arange(t)
    pos_emb = params.position_embedding[pos]
    x = token_emb + pos_emb
    mask = jnp.tril(jnp.ones((t, t)))
    
    for i in range(n_layer):
        ln1 = layer_norm(x, params.layer_norms[i])
        qkv = jnp.dot(ln1, params.attention_weights[i])
        q, k, v = jnp.split(qkv, 3, axis=-1)
        q = q.reshape(b, t, n_head, head_size).transpose(0, 2, 1, 3)
        k = k.reshape(b, t, n_head, head_size).transpose(0, 2, 1, 3)
        v = v.reshape(b, t, n_head, head_size).transpose(0, 2, 1, 3)
        
        att = attention(q, k, v, mask)
        att = att.transpose(0, 2, 1, 3).reshape(b, t, n_embd)
        att = jnp.dot(att, params.attention_projections[i])
        x = x + att
        
        ln2 = layer_norm(x, params.layer_norms[i])
        mlp = jnp.dot(ln2, params.mlp_weights[i]['c_fc'])
        mlp = jax.nn.gelu(mlp)
        mlp = jnp.dot(mlp, params.mlp_weights[i]['c_proj'])
        x = x + mlp
        
        if training:
            dropout_key, key = jax.random.split(key)
            x = jnp.where(jax.random.uniform(dropout_key, x.shape) > dropout, x, 0)
    
    x = layer_norm(x, params.final_layer_norm)
    logits = jnp.dot(x, params.lm_head)
    
    return logits

@jit
def loss_fn(params, batch, key):
    x, y = batch
    logits = forward(params, x, key)
    loss = optax.softmax_cross_entropy_with_integer_labels(
        logits.reshape(-1, logits.shape[-1]),
        y.reshape(-1)
    )
    return jnp.mean(loss)

@jit
def train_step(params, opt_state, batch, key):
    loss, grads = jax.value_and_grad(loss_fn)(params, batch, key)
    updates, opt_state = optimizer.update(grads, opt_state)
    params = optax.apply_updates(params, updates)
    return params, opt_state, loss

# Initialize model
vocab_size = int(max(train_data)) + 1
params = init_params(init_rng, vocab_size)

# Initialize optimizer
optimizer = optax.adam(learning_rate)
opt_state = optimizer.init(params)

# Main training loop
for iter in range(max_iters):
    rng, split_key = jax.random.split(rng)
    batch = get_batch(train_data, split_key)
    params, opt_state, loss = train_step(params, opt_state, batch, split_key)
    
    if iter % eval_interval == 0:
        losses = []
        for _ in range(eval_iters):
            rng, split_key = jax.random.split(rng)
            batch = get_batch(val_data, split_key)
            losses.append(loss_fn(params, batch, split_key))
        print(f"step {iter}: train loss {loss:.4f}, val loss {jnp.mean(jnp.array(losses)):.4f}")