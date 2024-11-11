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
    out_dir: str = 'out'  # ignored if init_from is not 'resume'
    prompt_source: str = 'cli'  # Can be 'file', 'cli', or 'inline'
    prompt_file: str = 'prompt.txt'
    prompt_inline: str = "What is the answer to life, the universe, and everything?" # or "<|endoftext|>" or etc.
    cli_prompt: str = ""
    num_samples: int = 5  # number of samples to draw
    max_new_tokens: int = 20  # number of tokens generated in each sample
    temperature: float = 1.0  # 1.0 = no change, < 1.0 = less random, > 1.0 = more random, in predictions
    top_k: int = 5  # retain only the top_k most likely tokens, clamp others to have 0 probability
    seed: int = 1337
    dtype: Optional[str] = jnp.bfloat16 if jax.devices()[0].device_kind == "TPU" else jnp.float16

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
    enc = tiktoken.get_encoding("gpt2")
    decode = lambda l: enc.decode(l)

    initial_tokens = get_initial_prompt(inf_config)

    for k in range(inf_config.num_samples):
        rng, subkey = jax.random.split(rng)
        sample = model.apply(
            {'params': params},
            initial_tokens,
            rng=subkey,
            rngs={'dropout': subkey},
            method=model.generate,
            max_new_tokens=inf_config.max_new_tokens,
            temperature=inf_config.temperature,
            top_k=inf_config.top_k
        )
        print(f"Sample {k + 1}:")
        print(decode(sample[0].tolist()))
        print('---------------')

def main():
    # Load inference config
    inf_config = InferenceConfig()

    # Set up key
    rng = jax.random.PRNGKey(inf_config.seed)

    # Model init with correct config. When loading from pretrained, 
    # nice vocab size must be overwritten with real size.
    if inf_config.init_from == 'resume':
        pass
        # config = GPT2Config(**model_args_chkpt)
        # model = GPT2(config)
        # TODO make loading from checkpoint a thing once training is implemented
        raise NotImplementedError("Loading from checkpoint is not yet implemented.")
    else:
        # Initialize with pretrained parameters and fix vocab size
        config = GPT2Config(vocab_size=50257) 
        _, params = get_pretrained_params(inf_config.init_from)
        model = GPT2(config)
        
    run_generation(inf_config, model, params['params'], rng)

if __name__ == "__main__":
    main()