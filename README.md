# TPU Starter

Everything you want to know about Google Cloud TPUs

## Introduction to Cloud TPUs (work in progress)

(coming soon)

- Cloud TPU machine: TPU VM, TPU Node (deprecated), Colab TPU (different)
- Deep learning libraries: [Tensorflow](Tensorflow) (officially supported by Google), [PyTorch](https://pytorch.org/) (supports TPU via PyTorch XLA), [JAX](https://github.com/google/jax) (latest and most suitable for TPU)
- Linear algebra libraries: [NumPy](https://numpy.org/) (CPU only), [JAX](https://github.com/google/jax) (cross-platform)
- JAX ecosystem: [JAX](https://github.com/google/jax) (basis), [Flax](https://github.com/google/flax) (neural network), [DM Haiku](https://github.com/deepmind/dm-haiku) (neural network), [Optax](https://github.com/deepmind/optax) (optimizer)

## Resources (work in progress)

- Product page: https://cloud.google.com/tpu
- Documentation: https://cloud.google.com/tpu/docs
- TPU Research Cloud: https://sites.research.google/trc/about/
- Hugging Face Accelerate: https://github.com/huggingface/accelerate

## Environment Setup (work in progress)

Before you can SSH into the Cloud VM, you need to login by the `gcloud` command:

```sh
gcloud alpha compute tpus tpu-vm ssh node-1 --zone europe-west4-a
```

After logging in, you can add your public key to `~/.ssh/authorized_keys`.

Install packages:

```sh
sudo apt update
sudo apt upgrade
sudo apt install -y python-is-python3 python3.8-venv mosh byobu
sudo reboot
```

Create a virtual environment:

```sh
python -m venv ~/venv38
. ~/venv38/bin/activate
pip install -U pip
pip install -U wheel
```

Install common packages and libraries:

```sh
pip install -r requirements.txt
```

How can I know the TPU is working?

Run `00-basics/test_jax.py`.

## Community (work in progress)

As of 23 Feb, 2022, there is no official chat group for Cloud TPUs. You can join our unofficial chat group [@cloudtpu](https://t.me/cloudtpu) on Telegram.
