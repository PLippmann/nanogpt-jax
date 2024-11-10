import jax
import flax
import jax.numpy as jnp
from flax import struct
from flax import linen as nn
import optax 

from typing import Optional


# Use flax.struct to define a more Jax-compatible config
@struct.dataclass
class GPT2Config:
    block_size: int = 1024
    vocab_size: int = 50257
    n_embd: int = 768
    n_layers: int = 12
    n_heads: int = 12
    dropout: float = 0.0
    use_bias: bool = True
    dtype: Optional[str] = jnp.float32 # Use jnp.bfloat16 for TPU

class SelfAttentionFlax(nn.Module):
    """TODO Check speed vs self written."""
    config: GPT2Config

    @nn.compact
    def __call__(self, x, mask=None, deterministic=False):
        """https://flax.readthedocs.io/en/v0.5.3/_autosummary/flax.linen.MultiHeadDotProductAttention.html"""
        attn_output = nn.MultiHeadDotProductAttention(
            num_heads=self.config.n_heads,
            qkv_features=self.config.n_embd,
            out_features=self.config.n_embd,
            use_bias=self.config.use_bias,
            dropout_rate=self.config.dropout,
            deterministic=deterministic,
        )(x, x, mask=mask)

        return attn_output

class SelfAttention(nn.Module):
    config: GPT2Config

    @nn.compact
    def __call__(self, x, mask=None, deterministic=False):
        """Multi-head self-attention mechanism."""
        B, T, C = x.shape # B: batch size, T: sequence length, C: channel size
        head_dim = C // self.config.n_heads
        
        # Query, key, value projections  (B, T, C) -> (B, n_heads, T, head_dim)
        qkv = nn.Dense(3 * C, use_bias=self.config.use_bias, dtype=self.config.dtype, name='c_attn')(x)
        qkv = qkv.reshape(B, T, 3 * self.config.n_heads, head_dim)
        q, k, v = jnp.array_split(qkv, 3, axis=2)
        
        # Calculate attention matrix
        scale = 1.0 / jnp.sqrt(head_dim).astype(self.config.dtype)
        
        # Attn weight shape is (batch..., num_heads, q_length, kv_length)
        attn = jnp.einsum('...qhd,...khd->...hqk', q, k) * scale
        attn = jnp.where(mask, attn, jnp.finfo(self.config.dtype).min)
        attn = jax.nn.softmax(attn).astype(self.config.dtype)
        attn = nn.Dropout(self.config.dropout)(attn, deterministic=deterministic)

        # Return weighted sum over values for each query position
        x = jnp.einsum('...hqk,...khd->...qhd', attn, v).reshape(B, T, C)
        x = nn.Dense(C, use_bias=self.config.use_bias, dtype=self.config.dtype, name='c_proj')(x)

        x = nn.Dropout(rate=self.config.dropout)(x, deterministic=deterministic)
        return x

class MLP(nn.Module):
    config: GPT2Config

    @nn.compact
    def __call__(self, x, deterministic=False):
        """Two-layer MLP with GELU activation."""
        x = nn.Dense(4 * self.config.n_embd, dtype=self.config.dtype, use_bias=self.config.use_bias, name='c_fc')(x) # Fully connected layer
        x = nn.gelu(x)
        x = nn.Dense(self.config.n_embd, dtype=self.config.dtype, use_bias=self.config.use_bias, name='c_proj')(x) # Projection layer
        output = nn.Dropout(self.config.dropout)(x, deterministic=deterministic)
        return output

class Block(nn.Module):
    config: GPT2Config

    def setup(self):
        self.ln_1 = nn.LayerNorm(epsilon=1e-5, dtype=self.config.dtype, use_bias=self.config.use_bias)
        self.attn = SelfAttention(self.config)
        self.ln_2 = nn.LayerNorm(epsilon=1e-5, dtype=self.config.dtype, use_bias=self.config.use_bias)
        self.mlp = MLP(self.config)

    def __call__(self, x, mask=None, deterministic=False):
        """Compute the forward pass of a single transformer block. Layer norm -> self attention -> layer norm -> mlp."""
        # Layer normalization and attention
        x = x + self.attn(self.ln_1(x), mask, deterministic=deterministic) # With residual

        # Layer normalization and MLP
        x = x + self.mlp(self.ln_2(x), deterministic=deterministic) # With residual

        return x
    
class GPT2(nn.Module):
    """Define a GPT2 model using Flax/linen"""
    config: GPT2Config

    def init(self, rng):
        """
        by jitting init, traced values instead of concrete values are used
        which saves memory (since un-jitted model may not fit in memory)
        """
        tokens = jnp.zeros((2, self.config.block_size), dtype=jnp.uint16)
        params = jax.jit(super().init, static_argnums=(2,))(rng, tokens, None)
        return params

    @nn.compact
    def __call__(self, inputs, targets=None, deterministic=False):
        """Compute the forward pass of the model."""
        B, T = inputs.shape # B: batch size, T: sequence length
        assert T <= self.config.block_size, f"Input length {T} is longer than block size {self.config.block_size}"

        # Positional encoding vector of shape (1, T)
        positions = jnp.arange(0, T)[None]

        # Tril attention mask to avoid attending to future tokens
        mask = jnp.tril(jnp.ones((T, T)))
        #attn_mask = nn.make_causal_mask(inputs, dtype=bool) # TODO check which mask implementation is better

        # Token and positional embeddings
        wte = nn.Embed(self.config.vocab_size, self.config.n_embd, dtype=self.config.dtype, name='wte')
        wpe = nn.Embed(self.config.block_size, self.config.n_embd, dtype=self.config.dtype, name='wpe')

        # Sum embeddings and apply dropout
        x = nn.Dropout(rate=self.config.dropout)(wte(inputs) + wpe(positions), deterministic=deterministic)

        for i in range(self.config.n_layers):
            x = Block(self.config, name=str(i))(x, mask, deterministic=deterministic)

        x = nn.LayerNorm(epsilon=1e-5, dtype=self.config.dtype, use_bias=self.config.use_bias, name='ln_f')(x)

        logits = wte.attend(x)

        if targets is not None:
            loss = optax.softmax_cross_entropy_with_integer_labels(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1)).mean()
        else:
            loss = None

        return logits, loss
    
    def generate(self, inputs, rng, max_new_tokens=100, temperature=1.0, top_k=None):
        """
        Take a input indices (shape (B,T)) and complete the sequence max_new_tokens times, 
        feeding the predictions back into the model each time.
        """
        def top_k_logits(logits, k):
            values, _ = jax.lax.top_k(logits, k)
            min_values = jnp.expand_dims(values[:, -1], axis=-1)
            return jnp.where(logits < min_values, jnp.ones_like(logits) * -1e9, logits)

        for _ in range(max_new_tokens):
            logits, _ = self(inputs)
            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                logits = top_k_logits(logits, top_k)
            
            # TODO no softmax?
            # jax.random.categorical expects log probabilities, while torch.multinomial works with actual probabilities
            new_tokens = jax.random.categorical(rng, logits, 1)
            inputs = jnp.concatenate([inputs, new_tokens[:, None]], axis=-1)
        
        return inputs