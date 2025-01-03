# NanoGPT-JAX

A high-performance JAX/Flax implementation of [NanoGPT](https://github.com/karpathy/nanoGPT) optimized for TPU training.

## Overview

This project reimplements Andrej Karpathy's NanoGPT in JAX, focusing on performance and scalability. It leverages JAX's automatic differentiation and compilation capabilities along with Flax's neural network layers to create an efficient and maintainable codebase that runs distributedly on TPUs.

### Core Features
- 🚀 Full JAX/Flax implementation optimized for TPUs
- 📈 Distributed training with `@pmap`
- 🔄 Gradient accumulation for larger effective batch sizes
- 📊 Integrated Weights & Biases logging
- 💾 Support for inference straight from pretrained weights
- 🎯 Cosine learning rate schedule with warmup

### Training Results
We reach a validation loss of 3.17 after 270k steps, at which point the model had converged. This took roughly 18 hours on a TPU v3-8.

![Loss Plot](assets/loss.svg)

[![Weights & Biases](https://img.shields.io/badge/WandB-Logs-yellow?logo=wandb)](https://wandb.ai/teateam/nanogpt-jax/runs/sw0gw8vk?nw=dg8746gjz4)

Additionally, when training on a TPU, we hit an average duty cycle of 77%, indicating good accelerator utilization.

## Installation

1. Clone the repository:
```bash
git clone https://github.com/plippmann/nanogpt-jax && cd nanogpt-jax
```

2. Set up the environment:
```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
source ~/miniconda3/bin/activate
conda create -n myenv python=3.10
conda activate myenv
```

3. Install JAX (TPU version):
```bash
pip install -U "jax[tpu]" -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
```

4. Install dependencies:
```bash
pip install -r requirements.txt
```

## Structure
```
nanogpt-jax/
├── nanogpt/
│   ├── model.py      # Core GPT-2 implementation
│   ├── train.py      # Training loop and configuration
│   ├── inference.py  # Text generation utilities
│   ├── pretrained.py # Load pretrained weights
│   └── tests.py      # Sanity checks for model.py
├── data/
│   └── openwebtext/
│       └── prepare.py # Get data
│   └── shakespeare/
│       └── prepare.py # Get data
└── requirements.txt
```

## Implementation Details

### Model Architecture
- Implements the GPT-2 architecture using Flax's neural network modules
- Supports configurable model sizes
- Own implementation of causal self-attention as well as Flax's version

### Training
- Distributed training across TPU cores using `@pmap`
- Gradient accumulation for higher effective batch sizes
- Learning rate scheduling with warmup and cosine decay
- AdamW optimizer for now
- Integrated W&B logging for training metrics

### Project Status
1. ~~Implement the model in JAX~~
2. ~~Write tests~~
3. ~~Load pretrained weights~~
4. ~~Perform inference from pretrained weights~~
5. ~~Train the model on TPUs~~
6. ~~Make it fast with @pmap/@jit~~
7. ~~Run inference on the trained model~~
8. Post-training fun
9. ~~Implement RoPE, Muon optimizer, and other improvements~~

### TPU Training Guide

1. Prepare training data:
```bash
python data/openwebtext/prepare.py
```
Upload the resulting `train.bin` and `val.bin` to your GCP storage bucket.

1. Create a TPU VM:
```bash
ZONE=europe-west4-a
TPU_TYPE=v3-8
VM_NAME=jax-gpt-v3-8

gcloud alpha compute tpus tpu-vm create $VM_NAME \
    --zone=$ZONE \
    --accelerator-type=$TPU_TYPE \
    --version=tpu-ubuntu2204-base \
    --preemptible
```

3. SSH into the VM:
```bash
gcloud alpha compute tpus tpu-vm ssh $VM_NAME --zone=$ZONE
```

4. Start training:
```bash
python train.py
```

5. Generate text from best checkpoint:
```bash
python inference.py --init_from resume --checkpoint_type best
```

6. Clean up:
```bash
gcloud alpha compute tpus tpu-vm delete $VM_NAME --zone=$ZONE
```

## Useful Resources
- [NanoGPT](https://github.com/karpathy/nanoGPT/tree/master) by Karpathy
- [gpt-jax](https://github.com/jenkspt/gpt-jax/tree/main) by Penn Jenks
- [PyTorch to JAX blog](https://github.com/ROCm/rocm-blogs/tree/release/blogs/artificial-intelligence/nanoGPT-JAX) by Douglas Jia

## Contact
Feel free to reach out at [p.lippmann@tudelft.nl](mailto:p.lippmann@tudelft.nl).

![Trees](assets/landscape.png)