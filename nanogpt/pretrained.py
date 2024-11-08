from flax.core.frozen_dict import FrozenDict, freeze, unfreeze
from flax.traverse_util import flatten_dict, unflatten_dict
from typing import Tuple
from transformers import FlaxGPT2LMHeadModel

from model import GPT2Config


def convert_hf_params(hf_params: FrozenDict, num_heads, num_embeds) -> FrozenDict:
    params = unfreeze(hf_params)
    for k, v in params.pop('h', {}).items():
        params[k] = v

    params = flatten_dict(params, sep='.')
    for k in params.keys():
        #if k.endswith('attn.c_attn.bias'):
        #    params[k] = params[k].reshape(num_heads, -1)
        if k.endswith('attn.c_attn.kernel'):
            #params[k] = params[k].reshape(num_embeds, num_heads, -1) 
            params[k] = params[k].T
        elif k.endswith('attn.c_proj.kernel'):
            #params[k] = params[k].reshape(num_heads, -1, num_embeds)
            params[k] = params[k].T
        elif k.split('.')[1] == 'mlp' and k.endswith('kernel'):
            params[k] = params[k].T

    params = unflatten_dict({f'params.{k}': v for k, v in params.items()}, sep='.')

    return freeze(params)

def get_pretrained_params(model_type: str) -> Tuple[GPT2Config, FrozenDict]:
    """
    Returns config and pretrained parameters from huggingface gpt models 
    """
    assert model_type in ('gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl')
    # only dropout can be overridden see more notes below
    print("loading weights from pretrained gpt: %s" % model_type)

    config = {
        'gpt2':         GPT2Config(n_layers=12, n_heads=12, n_embd=768),  # 124M params
        'gpt2-medium':  GPT2Config(n_layers=24, n_heads=16, n_embd=1024), # 350M params
        'gpt2-large':   GPT2Config(n_layers=36, n_heads=20, n_embd=1280), # 774M params
        'gpt2-xl':      GPT2Config(n_layers=48, n_heads=25, n_embd=1600), # 1558M params
    }[model_type]

    model_hf = FlaxGPT2LMHeadModel.from_pretrained(model_type)
    hf_params = model_hf.params['transformer']
    params = convert_hf_params(hf_params, config.n_heads, config.n_embd)
    return config, params