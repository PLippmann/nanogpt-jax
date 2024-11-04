import jax
import jax.numpy as jnp
from tqdm import tqdm
import os 

from utils import load_encoder_hparams_and_params, load_and_preprocess_data


# GELU activation function
def gelu(x):
    return 0.5 * x * (1 + jnp.tanh(jnp.sqrt(2 / jnp.pi) * (x + 0.044715 * x**3)))

# Convert logits to probabilities
def softmax(x):
    exp_x = jnp.exp(x - jnp.max(x, axis=-1, keepdims=True))
    return exp_x / jnp.sum(exp_x, axis=-1, keepdims=True)

# Layer norm
def layer_norm(x, g, b, eps: float = 1e-5):
    mean = jnp.mean(x, axis=-1, keepdims=True)
    variance = jnp.var(x, axis=-1, keepdims=True)
    return g * (x - mean) / jnp.sqrt(variance + eps) + b

# Linear transformation
def linear(x, w, b):
    return x @ w + b

# Feed-forward network
def ffn(x, c_fc, c_proj):
    return linear(gelu(linear(x, **c_fc)), **c_proj)

# Single attention head computation
def attention(q, k, v, mask):
    return softmax(q @ k.T / jnp.sqrt(q.shape[-1]) + mask) @ v

# Multi-head attention layer
def mha(x, c_attn, c_proj, n_head):
    x = linear(x, **c_attn)
    qkv_heads = list(map(lambda x: jnp.split(x, n_head, axis=-1), jnp.split(x, 3, axis=-1)))
    causal_mask = (1 - jnp.tri(x.shape[0], dtype=x.dtype)) * -1e10
    out_heads = [attention(q, k, v, causal_mask) for q, k, v in zip(*qkv_heads)]
    x = linear(jnp.hstack(out_heads), **c_proj)
    return x

# Complete transformer block
def transformer_block(x, mlp, attn, ln_1, ln_2, n_head):
    x = x + mha(layer_norm(x, **ln_1), **attn, n_head=n_head)
    x = x + ffn(layer_norm(x, **ln_2), **mlp)
    return x

# Log softmax function for numerical stability
def log_softmax(x):
    x_max = jnp.max(x, axis=-1, keepdims=True)
    exp_x = jnp.exp(x - x_max)
    return x - x_max - jnp.log(jnp.sum(exp_x, axis=-1, keepdims=True))

# Forward pass
def gpt2(inputs, wte, wpe, blocks, ln_f, n_head):
    x = wte[inputs] + wpe[jnp.arange(len(inputs))]
    for block in blocks:
        x = transformer_block(x, **block, n_head=n_head)
    return layer_norm(x, **ln_f) @ wte.T

# Loss function for training
def lm_loss(params, inputs, n_head):
    x, y = inputs[:-1], inputs[1:]  # Split inputs into context and targets
    logits = gpt2(x, **params, n_head=n_head)
    loss = jnp.mean(-log_softmax(logits)[jnp.arange(len(y)), y])
    return loss

# Generate tokens using the model
def generate(inputs, params, n_head, n_tokens_to_generate):
    inputs = list(inputs)
    for _ in tqdm(range(n_tokens_to_generate), "generating"):
        logits = gpt2(jnp.array(inputs), **params, n_head=n_head)
        next_id = jnp.argmax(logits[-1])
        inputs.append(int(next_id))
    return inputs[len(inputs) - n_tokens_to_generate:]

# Training loop for one epoch
def train_epoch(params, sequences, hparams, learning_rate=1e-4, batch_size=4):
    total_loss = 0
    num_batches = len(sequences) // batch_size
    
    # Convert parameters to JAX arrays
    params = jax.tree.map(jnp.array, params)
    
    for i in tqdm(range(num_batches), desc="Training batches"):
        batch_start = i * batch_size
        batch_end = batch_start + batch_size
        batch_sequences = sequences[batch_start:batch_end]
        
        # Compute average loss and gradients for the batch
        batch_loss = 0
        batch_grads = None
        
        for sequence in batch_sequences:
            sequence = jnp.array(sequence)
            loss_value = lm_loss(params, sequence, hparams["n_head"])
            grads = jax.grad(lm_loss)(params, sequence, hparams["n_head"])
            
            if batch_grads is None:
                batch_grads = grads
            else:
                batch_grads = jax.tree.map(lambda x, y: x + y, batch_grads, grads)
            batch_loss += loss_value
            
        # Average the gradients and loss
        batch_grads = jax.tree.map(lambda x: x / batch_size, batch_grads)
        batch_loss /= batch_size
        
        # Update parameters
        params = jax.tree.map(
            lambda p, g: p - learning_rate * g,
            params,
            batch_grads
        )
        
        total_loss += batch_loss
        
    return params, total_loss / num_batches

def main(mode: str = "train", 
         data_path: str = "./data/tinysp/input.txt",
         model_size: str = "124M", 
         models_dir: str = "models",
         num_epochs: int = 1,
         learning_rate: float = 1e-4,
         batch_size: int = 4,
         max_sequence_length: int = 1024,
         inference_prompt: str = None,
         n_tokens_to_generate: int = 40,
         params = None):
    
    # Load model components
    encoder, hparams, params = load_encoder_hparams_and_params(model_size, models_dir)
    
    if mode == "train":
        print("Loading training data...")
        sequences = load_and_preprocess_data(
            data_path, 
            encoder, 
            max_length=max_sequence_length
        )
        print(f"Loaded {len(sequences)} training sequences")
        
        # Training loop
        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch + 1}/{num_epochs}")
            params, avg_loss = train_epoch(
                params, 
                sequences, 
                hparams,
                learning_rate=learning_rate,
                batch_size=batch_size
            )
            print(f"Average loss: {avg_loss:.4f}")
        
        # Save trained parameters
        output_dir = os.path.join(models_dir, f"{model_size}_finetuned")
        os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(output_dir, "params.pkl"), "wb") as f:
            import pickle
            pickle.dump(params, f)
            
        return params
        
    elif mode == "generate":
        if inference_prompt is None:
            raise ValueError("Please provide an inference_prompt for generation mode")
        
        input_ids = encoder.encode(inference_prompt)
        assert len(input_ids) + n_tokens_to_generate < hparams["n_ctx"]
        
        output_ids = generate(input_ids, params, hparams["n_head"], n_tokens_to_generate)
        output_text = encoder.decode(output_ids)
        return output_text

if __name__ == "__main__":
    print("Starting training...")
    trained_params = main(
        mode="train",
        data_path="./data/tinysp/input.txt",
        num_epochs=3,
        learning_rate=1e-4,
        batch_size=4
    )
    
    print("\nStarting inference...")
    generated_text = main(
        mode="generate",
        inference_prompt="Behold,",
        n_tokens_to_generate=100,
        params=trained_params  # Use the trained parameters
    )
    print(f"\nGenerated text:\n{generated_text}")