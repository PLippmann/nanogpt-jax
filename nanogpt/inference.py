import os
import pickle
from contextlib import nullcontext
import jax
import jax.numpy as jnp
import flax.linen as nn
import optax
from flax.training import checkpoints
import tiktoken

from model import GPT2Config, GPT2
from pretrained import get_pretrained_params

# -----------------------------------------------------------------------------
init_from = 'gpt2'  # either 'resume' (from an out_dir) or a gpt2 variant (e.g. 'gpt2-xl')
out_dir = 'out'  # ignored if init_from is not 'resume'
start = "What is the answer to life, the universe, and everything?"  # or "<|endoftext|>" or etc. Can also specify a file, use as: "FILE:prompt.txt"
num_samples = 5  # number of samples to draw
max_new_tokens = 20  # number of tokens generated in each sample
temperature = 1.0  # 1.0 = no change, < 1.0 = less random, > 1.0 = more random, in predictions
top_k = 5  # retain only the top_k most likely tokens, clamp others to have 0 probability
seed = 1337
dtype = jnp.bfloat16 if jax.devices()[0].device_kind == "TPU" else jnp.float16
#exec(open('configurator.py').read())  # overrides from command line or config file
# -----------------------------------------------------------------------------

# Set up random seed
rng = jax.random.PRNGKey(seed)

# Model
config = GPT2Config()
model = GPT2(config)

if init_from == 'resume':
    pass
else:
    # Initialize with pretrained parameters
    params = get_pretrained_params(init_from)

# Look for the meta pickle in case it is available in the dataset folder
load_meta = False
if init_from == 'resume' and 'config' in state and 'dataset' in state['config']:
    meta_path = os.path.join('data', state['config']['dataset'], 'meta.pkl')
    load_meta = os.path.exists(meta_path)
if load_meta:
    print(f"Loading meta from {meta_path}...")
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)
    stoi, itos = meta['stoi'], meta['itos']
    encode = lambda s: [stoi[c] for c in s]
    decode = lambda l: ''.join([itos[i] for i in l])
else:
    print("No meta.pkl found, assuming GPT-2 encodings...")
    enc = tiktoken.get_encoding("gpt2")
    encode = lambda s: enc.encode(s, allowed_special={"<|endoftext|>"})
    decode = lambda l: enc.decode(l)

# Encode the beginning of the prompt
if start.startswith('FILE:'):
    with open(start[5:], 'r', encoding='utf-8') as f:
        start = f.read()
start_ids = encode(start)
x = jnp.array([start_ids])

# Initial tokens from 'start' string
initial_tokens = jnp.array([start_ids])

# TODO this is so hacky it hurts
params = params[1] if isinstance(params, tuple) else params
if 'params' in params:
    params = params['params']

# Run generation
for k in range(num_samples):
    rng, subkey = jax.random.split(rng)
    sample = model.apply(
        {'params': params},  # Correct format for params
        initial_tokens,
        rng=subkey, # Subkey for generation randomness
        rngs={'dropout': subkey},  # Subkey for dropout randomness
        method=model.generate,  # Specify generate method
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k
    )
    print(f"Sample {k + 1}:")
    print(decode(sample[0].tolist()))
    print('---------------')