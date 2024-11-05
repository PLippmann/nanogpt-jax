import jax
import flax
import jax.numpy as jnp
from flax import struct
from flax import linen as nn
import optax 

from typing import Optional


# Use flax.struct (instead of dataclass) to define a more Jax-compatible config
@struct.dataclass
class GPT2Config:
    block_size: int = 1024
    vocab_size: int = 50304
    n_embd: int = 768
    n_layer: int = 12
    n_head: int = 12
    dropout: float = 0.0
    bias: bool = True
    dtype: Optional[str] = None

class SelfAttentionFlax(nn.Module):
    """TODO Check speed vs self written. https://flax.readthedocs.io/en/v0.5.3/_autosummary/flax.linen.MultiHeadDotProductAttention.html"""
    config: GPT2Config
    bias: bool = True

    @nn.compact
    def __call__(self, x, mask=None):
        """Multi-head self-attention mechanism using Flax's built-in MultiHeadDotProductAttention."""
        attn_output = nn.MultiHeadDotProductAttention(
            num_heads=self.config.n_head,
            qkv_features=self.config.n_embd,
            out_features=self.config.n_embd,
            use_bias=self.bias,
            dropout_rate=self.config.dropout,
        )(x, x, mask=mask)

        return attn_output

class SelfAttention(nn.Module):
    config: GPT2Config
    bias: bool = True

    @nn.compact
    def __call__(self, x, mask=None):
        """Multi-head self-attention mechanism."""
        B, T, C = x.shape
        head_dim = C // self.config.n_head
        query = nn.Dense(self.config.n_embd, use_bias=self.config.bias)(x)
        key = nn.Dense(self.config.n_embd, use_bias=self.config.bias)(x)
        value = nn.Dense(self.config.n_embd, use_bias=self.config.bias)(x)

        # Split heads for multi-head attention
        query = query.reshape(B, T, self.config.n_head, head_dim).transpose(0, 2, 1, 3)
        key = key.reshape(B, T, self.config.n_head, head_dim).transpose(0, 2, 1, 3)
        value = value.reshape(B, T, self.config.n_head, head_dim).transpose(0, 2, 1, 3)

        # Scaled Dot-Product Attention
        attn_weights = jnp.einsum('bhtd,bhsd->bhts', query, key) / jnp.sqrt(head_dim)
        if mask is not None:
            attn_weights = jnp.where(mask, attn_weights, -1e9)
        attn_weights = nn.softmax(attn_weights, axis=-1)

        # Weighted sum of values
        attn_output = jnp.einsum('bhts,bhsd->bhtd', attn_weights, value)
        attn_output = attn_output.transpose(0, 2, 1, 3).reshape(B, T, C)

        # Output projection
        output = nn.Dense(self.config.n_embd, use_bias=self.bias)(attn_output)
        return output

class MLP(nn.Module):
    config: GPT2Config

    @nn.compact
    def __call__(self, x):
        """Two-layer MLP with activation."""
        hidden = nn.Dense(4 * self.config.n_embd)(x)
        hidden = nn.gelu(hidden)
        output = nn.Dense(self.config.n_embd)(hidden)
        return output

class Block(nn.Module):
    config: GPT2Config

    @nn.compact
    def __call__(self, x, mask=None):
        """Compute the forward pass of a single transformer block."""
        # Layer normalization
        ln_1 = nn.LayerNorm(epsilon=1e-5, dtype=self.config.dtype, use_bias=self.config.bias)(x)

        # Self attention
        attn_out = SelfAttention(self.config, bias=self.config.bias)(ln_1, mask)

        # Add and normalize
        x = x + attn_out
        x = nn.LayerNorm(epsilon=1e-5, dtype=self.config.dtype)(x)

        # MLP
        mlp_out = MLP(self.config)(x)

        x += mlp_out

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
        params = jax.jit(super().init, static_argnums=(2,))(rng, tokens, True)
        return params

    @nn.compact
    def __call__(self, inputs, targets=None, deterministic=False):
        """Compute the forward pass of the model."""
        B, T = inputs.shape # B: batch size, T: sequence length
        assert T <= self.config.block_size, f"Input length {T} is longer than block size {self.config.block_size}"

        # Positional encoding vector of shape (1, T)
        positions = jnp.arange(0, T)[None]

        # Attention mask to avoid attending to future tokens
        mask = jnp.less_equal(positions[:, :, None], positions[:, None, :]) # Lower triangular mask like tri

        # Token and positional embeddings
        wte = nn.Embed(self.config.vocab_size, self.config.n_embd, dtype=self.config.dtype, name='wte')
        wpe = nn.Embed(self.config.block_size, self.config.n_embd, dtype=self.config.dtype, name='wpe')

        # Sum embeddings and apply dropout
        x = nn.Dropout(rate=self.config.dropout, deterministic=deterministic)(wte(inputs) + wpe(positions))

        for _ in range(self.config.n_layer):
            x = Block(self.config)(x, mask)

        x = nn.LayerNorm(epsilon=1e-5, dtype=self.config.dtype, use_bias=self.config.bias)(x)

        if targets is not None:
            pass
            #logits = nn.Dense(features=self.config.vocab_size, use_bias=self.config.bias)(x)
            #loss = optax.softmax_cross_entropy_with_integer_labels(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1)).mean()
        else:
            pass
            
        logits = wte.attend(x)
        loss = None

        return logits, loss