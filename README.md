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
Install the requirements
```
pip -r requirements.txt
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
TPU_TYPE=v3-32
VM_NAME=jax-gpt-v3-8

gcloud alpha compute tpus tpu-vm create $VM_NAME \
    --zone=$ZONE \
    --accelerator-type=$TPU_TYPE \
    --version=v2-tf-stable \
    --preemptible
```
SSH into the VM
```
gcloud alpha compute tpus tpu-vm ssh $VM_NAME --zone=$ZONE
```
And finally run the training script with default settings
```
python train.py --tpu
```

## Useful Resources
Things that have been helpful in the development of this project:
- [NanoGPT](https://github.com/karpathy/nanoGPT/tree/master) by Karpathy
- [gpt-jax](https://github.com/jenkspt/gpt-jax/tree/main) by Penn Jenks
- [PyTorch to Jax blog](https://github.com/ROCm/rocm-blogs/tree/release/blogs/artificial-intelligence/nanoGPT-JAX) by Douglas Jia

![an image of a landscape](assets/landscape.png)