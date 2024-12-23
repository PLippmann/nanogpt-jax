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
    vocab_size: int = 50304 # Real vocab size is 50257 but doesn't run as well
    n_embd: int = 768
    n_layers: int = 12
    n_heads: int = 12
    dropout: float = 0.0
    use_bias: bool = True
    rope_base: int = 10000  # Base for RoPE calculations
    dtype: Optional[str] = jnp.bfloat16 if jax.devices()[0].device_kind == "TPU" else jnp.float16

def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    """Precompute the frequency tensor for complex exponentials (cis) with given dimensions."""
    freqs = 1.0 / (theta ** (jnp.arange(0, dim // 2).astype(jnp.float32) / (dim // 2)))
    t = jnp.arange(end)
    freqs = jnp.outer(t, freqs)  # [end, dim//2]
    # Create complex rotations
    return jnp.stack([jnp.cos(freqs), jnp.sin(freqs)], axis=-1)  # [end, dim//2, 2]

def apply_rotary_emb(x, freqs_cis):
    """Apply rotary embeddings to input tensors using precomputed frequencies."""
    # reshape xq, xk to [batch..., seq_len, n_heads, head_dim//2, 2]
    x_reshape = x.reshape(*x.shape[:-1], -1, 2)
    
    # Create complex numbers from pairs of real numbers
    x_real, x_imag = x_reshape[..., 0], x_reshape[..., 1]
    x_complex = x_real + 1j * x_imag
    
    # Create complex rotation
    freqs_cos = freqs_cis[..., 0]
    freqs_sin = freqs_cis[..., 1]
    freqs_complex = freqs_cos + 1j * freqs_sin
    
    # Apply complex rotation
    x_rotated = x_complex * freqs_complex
    
    # Convert back to real numbers
    x_out_real = jnp.real(x_rotated)
    x_out_imag = jnp.imag(x_rotated)
    x_out = jnp.stack([x_out_real, x_out_imag], axis=-1)
    
    # Restore original shape
    return x_out.reshape(x.shape)

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
        """Multi-head self-attention mechanism with RoPE."""
        B, T, C = x.shape # B: batch size, T: sequence length, C: channel size
        head_dim = C // self.config.n_heads
        assert head_dim % 2 == 0, "Head dimension must be divisible by 2 for RoPE"
        
        # Query, key, value projections  (B, T, C) -> (B, T, n_heads, head_dim)
        qkv = nn.Dense(3 * C, use_bias=self.config.use_bias, dtype=self.config.dtype, name='c_attn')(x)
        qkv = qkv.reshape(B, T, 3, self.config.n_heads, head_dim)
        q, k, v = jnp.array_split(qkv, 3, axis=2)
        q = q.squeeze(2)  # Remove the split dimension
        k = k.squeeze(2)
        v = v.squeeze(2)
        
        # Compute RoPE embeddings
        freqs_cis = precompute_freqs_cis(head_dim, T, self.config.rope_base)
        freqs_cis = jnp.expand_dims(freqs_cis, axis=1)  # [T, 1, dim//2, 2]
        
        # Apply RoPE to queries and keys
        q_roped = apply_rotary_emb(q, freqs_cis)
        k_roped = apply_rotary_emb(k, freqs_cis)
        
        # Calculate attention matrix
        scale = 1.0 / jnp.sqrt(head_dim).astype(self.config.dtype)
        attn = jnp.einsum('bthd,bkhd->bhtk', q_roped, k_roped) * scale

        # Create causal mask and combine with attention scores
        if mask is not None:
            # Expand mask to match attention shape [B, n_heads, T, T]
            mask = mask[None, None, :, :]
            # Use where with a large negative value for masked positions
            attn = jnp.where(mask == 0, jnp.finfo(self.config.dtype).min, attn)

        attn = jax.nn.softmax(attn).astype(self.config.dtype)
        attn = nn.Dropout(self.config.dropout)(attn, deterministic=deterministic)

        # Return weighted sum over values for each query position
        x = jnp.einsum('bhtk,bkhd->bthd', attn, v).reshape(B, T, C)
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

        # Create causal mask (1s in lower triangle, 0s elsewhere)
        mask = jnp.tril(jnp.ones((T, T)))

        # Token embeddings
        wte = nn.Embed(self.config.vocab_size, self.config.n_embd, dtype=self.config.dtype, name='wte')
        x = nn.Dropout(rate=self.config.dropout)(wte(inputs), deterministic=deterministic)

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