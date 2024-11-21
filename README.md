# nanogpt-jax
Implementation of [NanoGPT](https://github.com/karpathy/nanoGPT). All in Jax/Flax. 

Currently WIP. The goals are to:
1. ~~Implement the model in Jax~~
2. ~~Write tests~~
3. ~~Load pretrained weights~~
4. ~~Perform inference from pretrained weights~~
5. ~~Train the model on TPUs~~
6. ~~Make it fast with @pmap/@jit~~
7. Run inference on the trained model
8. Post-training
9. Implement RoPE, Muon optimizer, and other improvements

## Install
Clone the repository
```
git clone https://github.com/plippmann/nanogpt-jax && cd nanogpt-jax
```
Set up the environment
```
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
source ~/miniconda3/bin/activate
conda create -n myenv python=3.10
conda activate myenv
```
Install the TPU version of Jax
```
pip install -U "jax[tpu]" -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
```
Install the requirements
```
pip install -r requirements.txt
```
Prepare the data
```
python data/openwebtext/prepare.py
```
The resulting train.bin and val.bin should then be uploaded to a GCP storage bucket.

## Inference
The simplest inference example is to generate text from a set prompt and pretrained weights. 
```
python inference.py
```

More to follow...

## Train
Create a TPU VM
```
ZONE=europe-west4-a
TPU_TYPE=v3-8
VM_NAME=jax-gpt-v3-8

gcloud alpha compute tpus tpu-vm create $VM_NAME \
    --zone=$ZONE \
    --accelerator-type=$TPU_TYPE \
    --version=tpu-ubuntu2204-base \
    --preemptible
```
SSH into the VM
```
gcloud alpha compute tpus tpu-vm ssh $VM_NAME --zone=$ZONE
```
And finally run the training script with default settings
```
python train.py
```
Don't forget to delete the VM after you're done
```
gcloud alpha compute tpus tpu-vm delete $VM_NAME --zone=$ZONE
```

## Useful Resources
Things that have been helpful in the development of this project:
- [NanoGPT](https://github.com/karpathy/nanoGPT/tree/master) by Karpathy
- [gpt-jax](https://github.com/jenkspt/gpt-jax/tree/main) by Penn Jenks
- [PyTorch to Jax blog](https://github.com/ROCm/rocm-blogs/tree/release/blogs/artificial-intelligence/nanoGPT-JAX) by Douglas Jia

![an image of a landscape](assets/landscape.png)