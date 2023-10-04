#!/bin/bash

gcloud alpha compute tpus tpu-vm ssh node-2 --zone us-central2-b --worker all --command '

# Confirm that the script is running on the host
uname -a

# Install common packages
export DEBIAN_FRONTEND=noninteractive
sudo apt-get update -y -qq
sudo apt-get upgrade -y -qq
sudo apt-get install -y -qq golang neofetch zsh mosh byobu aria2

# Install Python 3.11
sudo apt-get install -y -qq software-properties-common
sudo add-apt-repository -y ppa:deadsnakes/ppa
sudo apt-get install -y -qq python3.11-full python3.11-dev

# Install Oh My Zsh
sh -c "$(curl -fsSL https://raw.githubusercontent.com/ohmyzsh/ohmyzsh/master/tools/install.sh)" "" --unattended
sudo chsh $USER -s /usr/bin/zsh

# Change timezone
# timedatectl list-timezones  # list timezones
sudo timedatectl set-timezone Asia/Hong_Kong  # change to your timezone

# Install JAX with TPU support
pip install -U pip
pip install -U wheel
pip install -U "jax[tpu]" -f https://storage.googleapis.com/jax-releases/libtpu_releases.html

'
