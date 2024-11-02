# nanogpt-jax
Implementations of [NanoGPT](https://github.com/karpathy/nanoGPT), [picoGPT](https://github.com/jaymody/picoGPT/tree/main), and something inbetween. All in Jax/Flax. Currently WIP.

This repo contains three things:
1. A performant implementation of GPT-2 in Jax that mostly sticks to Kaparthy's layout. (```./nanogpt```)
2. A simple GPT implementation in a single file of ~200 lines using Jax that you can run in Colab. Slow. (```./simple/simple.py```)
3. An implementation of picoGPT in Jax in just xx lines. Very slow. (```./short/pico.py```)

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
For nanogpt:
```
TBD
```

For the single file GPT:
```
Copy and paste into Colab, connect to TPU, and you're good to go.
```
<a target="_blank" href="https://colab.research.google.com/github/https://colab.research.google.com/drive/1_o2cDNtgRI_AOUrEIhAzkvlzPkSx7NK7">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>

For picoGPT:
```
python ./short/pico.py
```

![an image of a landscape](assets/landscape.png)
