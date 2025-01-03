import os
import pickle
from contextlib import nullcontext
import jax
import jax.numpy as jnp
import flax.linen as nn
import optax
from flax import struct
from flax.training import checkpoints
import tiktoken
from typing import Optional, Tuple

from model import GPT2Config, GPT2
from pretrained import get_pretrained_params

# Use flax.struct
@struct.dataclass
class InferenceConfig:
    init_from: str = 'gpt2'  # either 'resume' (from an out_dir) or a gpt2 variant (e.g. 'gpt2-xl')
    out_dir: str = 'output'  # directory containing the checkpoints
    checkpoint_type: str = 'best'  # either 'best' or 'latest'
    prompt_source: str = 'cli'  # Can be 'file', 'cli', or 'inline'
    prompt_file: str = 'prompt.txt'
    prompt_inline: str = "What is the answer to life, the universe, and everything?" # or "<|endoftext|>" or etc.
    cli_prompt: str = ""
    num_samples: int = 1  # number of samples to draw
    max_new_tokens: int = 100  # number of tokens generated in each sample
    temperature: float = 0.8  # 1.0 = no change, < 1.0 = less random, > 1.0 = more random, in predictions
    top_k: int = 40  # retain only the top_k most likely tokens, clamp others to have 0 probability
    seed: int = 1337
    dtype: Optional[str] = jnp.float32  # Use full precision for inference

    @property
    def checkpoint_dir(self) -> str:
        """Get absolute path for checkpoint directory."""
        if os.path.isabs(self.out_dir):
            return self.out_dir
        # Get the project root directory (parent of the script directory)
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        return os.path.abspath(os.path.join(project_root, self.out_dir))

def get_initial_prompt(config: InferenceConfig) -> Tuple[str, jnp.ndarray]:
    if config.prompt_source == 'file':
        with open(config.prompt_file, 'r', encoding='utf-8') as f:
            start = f.read()
    elif config.prompt_source == 'inline':
        start = config.prompt_inline
    elif config.prompt_source == 'cli':
        start = input("Enter the initial prompt: ")
        config = config.replace(cli_prompt=start)
    else:
        raise ValueError("Invalid prompt_source. Must be 'file', 'inline', or 'cli'.")

    enc = tiktoken.get_encoding("gpt2")
    encode = lambda s: enc.encode(s, allowed_special={"<|endoftext|>"})
    start_ids = encode(start)
    initial_tokens = jnp.array([start_ids])
    return initial_tokens

def run_generation(
    inf_config: InferenceConfig,
    model: GPT2,
    params: jnp.ndarray,
    rng: jnp.ndarray
) -> None:
    print("Starting text generation...")
    enc = tiktoken.get_encoding("gpt2")
    decode = lambda l: enc.decode(l)

    initial_tokens = get_initial_prompt(inf_config)
    print(f"Initial prompt encoded to {len(initial_tokens[0])} tokens")
    print(f"Initial prompt: {decode(initial_tokens[0].tolist())}")

    # Move params and tokens to device
    print("Moving data to device...")
    params = jax.device_put(params)
    initial_tokens = jax.device_put(initial_tokens)
    
    print("Setting up generation...")
    
    # Define a simple wrapper for generation
    def generate_sample(rng, tokens):
        return model.apply(
            {'params': params},
            tokens,
            rng=rng,
            method=model.generate,
            max_new_tokens=inf_config.max_new_tokens,
            temperature=inf_config.temperature,
            top_k=inf_config.top_k
        )

    print(f"\nGenerating {inf_config.num_samples} samples with:")
    print(f"- max_new_tokens: {inf_config.max_new_tokens}")
    print(f"- temperature: {inf_config.temperature}")
    print(f"- top_k: {inf_config.top_k}")
    print("---------------")

    for k in range(inf_config.num_samples):
        rng, subkey = jax.random.split(rng)
        print(f"\nGenerating sample {k + 1}...")
        try:
            sample = generate_sample(subkey, initial_tokens)
            print(f"Generated {len(sample[0]) - len(initial_tokens[0])} new tokens")
            generated_text = decode(sample[0].tolist())
            print("Sample output:")
            print(generated_text)
        except Exception as e:
            print(f"Error during generation: {e}")
            print(f"Full error: {repr(e)}")
            import traceback
            traceback.print_exc()
        print('---------------')

def load_trained_checkpoint(config: InferenceConfig):
    """Load a checkpoint from training."""
    checkpoint_dir = config.checkpoint_dir
    prefix = f'checkpoint_{config.checkpoint_type}_'
    
    restored = checkpoints.restore_checkpoint(
        ckpt_dir=checkpoint_dir,
        target=None,
        prefix=prefix
    )
    
    if restored is None:
        raise ValueError(f"No {config.checkpoint_type} checkpoint found in {checkpoint_dir}")
    
    if 'state' in restored:
        state = restored['state']
        step = restored.get('step', 0)
        print(f"Loaded {config.checkpoint_type} checkpoint from step {step}")
        return state['params']
    else:
        # Old format where the entire dict was the state
        print(f"Loaded {config.checkpoint_type} checkpoint (old format)")
        return restored['params']

def main():
    # Load inference config
    inf_config = InferenceConfig()
    print(f"Running on device: {jax.devices()[0].device_kind}")

    # Set up key
    rng = jax.random.PRNGKey(inf_config.seed)

    # Model init with correct config
    if inf_config.init_from == 'resume':
        # Initialize with our trained parameters
        config = GPT2Config(dtype=inf_config.dtype)  # Use same config as training but with full precision
        print("\nLoading model with config:", config)
        model = GPT2(config)
        try:
            params = load_trained_checkpoint(inf_config)
            params = {'params': params}  # Match the structure expected by the model
        except Exception as e:
            print(f"Error loading checkpoint: {e}")
            return
    else:
        # Initialize with pretrained parameters and fix vocab size
        config = GPT2Config(vocab_size=50257, dtype=inf_config.dtype)
        print("\nLoading pretrained model with config:", config)
        _, params = get_pretrained_params(inf_config.init_from)
        model = GPT2(config)
    
    print("\nModel loaded successfully")
    run_generation(inf_config, model, params['params'], rng)

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--init_from', type=str, choices=['resume', 'gpt2'], default='gpt2', 
                       help="'resume' to load from checkpoint, or 'gpt2' for pretrained")
    parser.add_argument('--out_dir', type=str, default='output', help='Directory containing checkpoints')
    parser.add_argument('--checkpoint_type', type=str, default='best', choices=['best', 'latest'], 
                       help='Which checkpoint to load')
    parser.add_argument('--num_samples', type=int, default=1, help='Number of samples to generate')
    parser.add_argument('--max_new_tokens', type=int, default=100, help='Number of tokens to generate')
    parser.add_argument('--temperature', type=float, default=0.8, help='Sampling temperature')
    parser.add_argument('--top_k', type=int, default=40, help='Top-k sampling parameter')
    args = parser.parse_args()
    
    print(f"Loading model with init_from={args.init_from}")
    
    # Update config with command line arguments
    inf_config = InferenceConfig(
        init_from=args.init_from,
        out_dir=args.out_dir,
        checkpoint_type=args.checkpoint_type,
        num_samples=args.num_samples,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k
    )
    
    main()