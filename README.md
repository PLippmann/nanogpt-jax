# nanogpt-jax
Implementations of [NanoGPT](https://github.com/karpathy/nanoGPT), [picoGPT](https://github.com/jaymody/picoGPT/tree/main), and something inbetween. All in Jax/Flax. Currently WIP.

This repo contains three things:
1. A performant implementation of GPT-2 in Jax that mostly sticks to Kaparthy's layout. (```./nanogpt```)
2. A simple GPT implementation in a single file of ~270 lines using Jax that you can run in Colab. Slow. (```./simple/simple.py```)
3. An implementation of picoGPT in just jnp in just ~150 lines. Very slow. (```./short/pico.py```)

## Install
Create venv
```
python3 -m venv myenv
```
Activate venv
```
source myenv/bin/activate
```
Install dependencies
```
pip install jax tiktoken optax
```

## Run
#### For nanogpt:
```
TBD
```

#### For the single file GPT: <a target="_blank" href="https://colab.research.google.com/github/PLippmann/nanogpt-jax/blob/main/simple/simple.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>
 
#### For picoGPT:
```
python ./short/pico.py
```

![an image of a landscape](assets/landscape.png)