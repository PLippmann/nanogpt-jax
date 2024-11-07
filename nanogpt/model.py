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
    use_bias: bool = True
    dtype: Optional[str] = None

class SelfAttentionFlax(nn.Module):
    """TODO Check speed vs self written."""
    config: GPT2Config

    @nn.compact
    def __call__(self, x, mask=None, deterministic=False):
        """https://flax.readthedocs.io/en/v0.5.3/_autosummary/flax.linen.MultiHeadDotProductAttention.html"""
        attn_output = nn.MultiHeadDotProductAttention(
            num_heads=self.config.n_head,
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
        head_dim = C // self.config.n_head
        query = nn.Dense(self.config.n_embd, use_bias=self.config.use_bias, dtype=self.config.dtype)(x)
        key = nn.Dense(self.config.n_embd, use_bias=self.config.use_bias, dtype=self.config.dtype)(x)
        value = nn.Dense(self.config.n_embd, use_bias=self.config.use_bias, dtype=self.config.dtype)(x)

        # Split heads for multi-head attention
        query = query.reshape(B, T, self.config.n_head, head_dim).transpose(0, 2, 1, 3) # (B, T, C) -> (B, n_head, T, head_dim)
        key = key.reshape(B, T, self.config.n_head, head_dim).transpose(0, 2, 1, 3) # (B, T, C) -> (B, n_head, T, head_dim)
        value = value.reshape(B, T, self.config.n_head, head_dim).transpose(0, 2, 1, 3) # (B, T, C) -> (B, n_head, T, head_dim)

        # Scaled Dot-Product Attention
        attn_weights = (jnp.einsum('bhtd,bhsd->bhts', query, key)) * (1.0 / jnp.sqrt(head_dim).astype(self.config.dtype))
        attn_weights = jnp.where(mask, attn_weights, -1e9) # TODO check -1e9
        attn_weights = nn.softmax(attn_weights, axis=-1).astype(self.config.dtype)

        # Weighted sum of values
        attn_output = jnp.einsum('bhts,bhsd->bhtd', attn_weights, value)
        attn_output = attn_output.transpose(0, 2, 1, 3).reshape(B, T, C) # (B, n_head, T, head_dim) -> (B, T, C)

        # Output projection
        output = nn.Dense(self.config.n_embd, use_bias=self.config.use_bias, dtype=self.config.dtype)(attn_output)
        output = nn.Dropout(self.config.dropout)(output, deterministic=deterministic)
        return output

class MLP(nn.Module):
    config: GPT2Config

    @nn.compact
    def __call__(self, x, deterministic=False):
        """Two-layer MLP with GELU activation."""
        x = nn.Dense(4 * self.config.n_embd, dtype=self.config.dtype, use_bias=self.config.use_bias)(x) # Fully connected layer
        x = nn.gelu(x)
        x = nn.Dense(self.config.n_embd, dtype=self.config.dtype, use_bias=self.config.use_bias)(x) # Projection layer
        output = nn.Dropout(self.config.dropout)(x, deterministic=deterministic)
        return output

class Block(nn.Module):
    config: GPT2Config

    def setup(self):
        self.layernorm1 = nn.LayerNorm(epsilon=1e-5, dtype=self.config.dtype, use_bias=self.config.use_bias)
        self.attention = SelfAttention(self.config)
        self.layernorm2 = nn.LayerNorm(epsilon=1e-5, dtype=self.config.dtype, use_bias=self.config.use_bias)
        self.mlp = MLP(self.config)

    @nn.compact
    def __call__(self, x, mask=None, deterministic=False):
        """Compute the forward pass of a single transformer block. Layer norm -> self attention -> layer norm -> mlp."""
        # Layer normalization and attention
        x = x + self.attention(self.layernorm1(x), mask, deterministic=deterministic) # With residual

        # Layer normalization and MLP
        x = x + self.mlp(self.layernorm2(x), deterministic=deterministic) # With residual

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

        for _ in range(self.config.n_layer):
            x = Block(self.config)(x, mask, deterministic=deterministic)

        x = nn.LayerNorm(epsilon=1e-5, dtype=self.config.dtype, use_bias=self.config.use_bias)(x)

        if targets is not None:
            logits = wte.attend(x)
            loss = optax.softmax_cross_entropy_with_integer_labels(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1)).mean()
        else:
            logits = wte.attend(x)
            loss = None

        return logits, loss
    
    def generate(self, inputs, max_new_tokens=1024, temperature=1.0, top_k=None):
        """
        Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
        the sequence max_new_tokens times, feeding the predictions back into the model each time.
        """
        def top_k_logits(logits, k):
            if k == 0 or k is None:
                return logits
            else:
                values, _ = jax.lax.top_k(logits, k)
                min_values = jnp.expand_dims(values[:, -1], axis=-1)
                return jnp.where(logits < min_values, jnp.ones_like(logits) * -1e9, logits)
            
        for _ in range(max_new_tokens):
            logits, _ = self(inputs)
            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                logits = top_k_logits(logits, top_k)
            new_tokens = jax.random.categorical(jax.random.PRNGKey(0), logits, 1)
            inputs = jnp.concatenate([inputs, new_tokens[:, None]], axis=-1)
        
        return inputs