# PicoGPT in Jax

```pico_jnp.py``` is an implementation using just jax.numpy without any training. Model weights are loaded straight from the OpenAI tf checkpoint.

```pico.py``` implements an additional training loop that allows finetuning on data.

```utils.py``` contains utility functions for loading the model and tokenizing text.