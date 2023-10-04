# TPU Starter

<h4 align="center">
    <p>
        <b>English</b> |
        <a href="https://github.com/ayaka14732/tpu-starter/blob/main/README_ko.md">한국어</a> |
        <a href="https://github.com/ayaka14732/tpu-starter/blob/main/README_zh.md">中文</a>
    <p>
</h4>

Everything you want to know about Google Cloud TPU

* [1. Community](#1-community)
* [2. Introduction to TPU](#2-introduction-to-tpu)
    * [2.1. Why TPU?](#21-why-tpu)
    * [2.2. How can I get free access to TPU?](#22-how-can-i-get-free-access-to-tpu)
    * [2.3. TPU is so good, why haven't I seen many people using it?](#23-tpu-is-so-good-why-havent-i-seen-many-people-using-it)
    * [2.4. I know TPU is good now. Can I touch a real TPU?](#24-i-know-tpu-is-good-now-can-i-touch-a-real-tpu)
    * [2.5. What does it mean to create a TPU instance? What do I actually get?](#25-what-does-it-mean-to-create-a-tpu-instance-what-do-i-actually-get)
* [3. Introduction to the TRC Program](#3-introduction-to-the-trc-program)
    * [3.1. How to apply for the TRC program?](#31-how-to-apply-for-the-trc-program)
    * [3.2. Is it really free?](#32-is-it-really-free)
* [4. Create a TPU VM Instance](#4-create-a-tpu-vm-instance)
    * [4.1. Modify VPC firewall](#41-modify-vpc-firewall)
    * [4.2. Create the instance](#42-create-the-instance)
    * [4.3. SSH to the server](#43-ssh-to-the-server)
* [5. Environment Setup](#5-environment-setup)
* [6. Development Environment Setup](#6-development-environment-setup)
    * [6.1. Set up Mosh and Byobu](#61-set-up-mosh-and-byobu)
    * [6.2. Set up VSCode Remote-SSH](#62-set-up-vscode-remote-ssh)
    * [6.3. How can I verify that the TPU is working?](#63-how-can-i-verify-that-the-tpu-is-working)
* [7. JAX Basics](#7-jax-basics)
    * [7.1. Why JAX?](#71-why-jax)
    * [7.2. Parallelism](#72-parallelism)
        * [7.2.1. Basics of jax.pmap](#721-basics-of-jaxpmap)
        * [7.2.2. What if I want to have randomness in the update function?](#722-what-if-i-want-to-have-randomness-in-the-update-function)
        * [7.2.3. What if I want to use optax optimizers in the update function?](#723-what-if-i-want-to-use-optax-optimizers-in-the-update-function)
    * [7.3. Freeze certain model parameters](#73-freeze-certain-model-parameters)
    * [7.4. Integration with Hugging Face Transformers](#74-integration-with-hugging-face-transformers)
    * [7.5. What is a[:, None]?](#75-what-is-a-none)
* [8. TPU Best Practices](#8-tpu-best-practices)
    * [8.1. Prefer Google Cloud Platform to Google Colab](#81-prefer-google-cloud-platform-to-google-colab)
    * [8.2. Prefer TPU VM to TPU node](#82-prefer-tpu-vm-to-tpu-node)
* [9. JAX Best Practices](#9-jax-best-practices)
    * [9.1. Import convention](#91-import-convention)
    * [9.2. Manage random keys in JAX](#92-manage-random-keys-in-jax)
    * [9.3. Serialize model parameters](#93-serialize-model-parameters)
    * [9.4. Conversion between NumPy arrays and JAX arrays](#94-conversion-between-numpy-arrays-and-jax-arrays)
    * [9.5. Conversion between PyTorch tensors and JAX arrays](#95-conversion-between-pytorch-tensors-and-jax-arrays)
    * [9.6. Type annotation](#96-type-annotation)
    * [9.7. Check if an array is either a NumPy array or a JAX array](#97-check-if-an-array-is-either-a-numpy-array-or-a-jax-array)
    * [9.8. Get the shapes of all parameters in a nested dictionary](#98-get-the-shapes-of-all-parameters-in-a-nested-dictionary)
    * [9.9. The correct way to generate random numbers on CPU](#99-the-correct-way-to-generate-random-numbers-on-cpu)
    * [9.10. Use optimizers from Optax](#910-use-optimizers-from-optax)
    * [9.11. Use the cross-entropy loss implementation from Optax](#911-use-the-cross-entropy-loss-implementation-from-optax)
* [10. How Can I...](#10-how-can-i)
    * [10.1. Run Jupyter Notebook on TPU VM](#101-run-jupyter-notebook-on-tpu-vm)
    * [10.2. Share files across multiple TPU VM instances](#102-share-files-across-multiple-tpu-vm-instances)
    * [10.3. Monitor TPU usage](#103-monitor-tpu-usage)
    * [10.4. Start a server on TPU VM](#104-start-a-server-on-tpu-vm)
    * [10.5. Run separate processes on different TPU cores](#105-run-separate-processes-on-different-tpu-cores)
* [11. Working With Pods](#11-working-with-pods)
    * [11.1. Create a shared directory using NFS](#111-create-a-shared-directory-using-nfs)
    * [11.2. Run a command simultaneously on all TPU Pods](#112-run-a-command-simultaneously-on-all-tpu-pods)
* [12. Common Gotchas](#12-common-gotchas)
    * [12.1. TPU VMs will be rebooted occasionally](#121-tpu-vms-will-be-rebooted-occasionally)
    * [12.2. One TPU core can only be used by one process at a time](#122-one-tpu-core-can-only-be-used-by-one-process-at-a-time)
    * [12.3. TCMalloc breaks several programs](#123-tcmalloc-breaks-several-programs)
    * [12.4. libtpu.so already in used by another process](#124-libtpuso-already-in-used-by-another-process)
    * [12.5. JAX does not support the multiprocessing fork strategy](#125-jax-does-not-support-the-multiprocessing-fork-strategy)

<!-- Created by https://github.com/ekalinin/github-markdown-toc -->

This project was inspired by [Cloud Run FAQ](https://github.com/ahmetb/cloud-run-faq), a community-maintained knowledge base about another Google Cloud product.

## 1. Community

Google's official Discord has established the `#tpu-research-cloud` channel, as mentioned at https://twitter.com/googledeveurope/status/1583032777984835585.

## 2. Introduction to TPU

### 2.1. Why TPU?

**TL;DR**: TPU is to GPU as GPU is to CPU.

TPU is hardware specifically designed for machine learning. For performance comparisons, see [Performance Comparison](https://github.com/huggingface/transformers/blob/main/examples/flax/language-modeling/README.md#runtime-evaluation) in Hugging Face Transformers:

![](assets/5.png)

Moreover, Google's [TRC program](https://sites.research.google/trc/about/) offers free TPU resources to researchers. If you've ever wondered what computing resources to use to train a model, you should try the TRC program, as it's the best option I know of. More information about the TRC program is provided below.

### 2.2. How can I get free access to TPU?

Researchers can apply to the [TRC program](https://sites.research.google/trc/about/) to obtain free TPU resources.

### 2.3. If TPU is so good, why do I rarely see others using it?

If you want to use PyTorch, TPU may not be suitable for you. TPU is poorly supported by PyTorch. In one of my past experiments using PyTorch, a batch took 14 seconds on a CPU but required 4 hours on a TPU. Twitter user @mauricetpunkt also thinks that [PyTorch's performance on TPUs is bad](https://twitter.com/mauricetpunkt/status/1506944350281945090).

In conclusion, if you want to do deep learning with TPU, you should use JAX as your deep learning framework. In fact, many popular deep learning libraries support JAX. For instance:

- [Many models in Hugging Face Transformers support JAX](https://huggingface.co/docs/transformers/index#supported-frameworks)
- [Keras supports using JAX as a backend](https://keras.io/keras_core/announcement/)
- SkyPilot has [examples using Flax](https://github.com/skypilot-org/skypilot/blob/master/examples/tpu/tpuvm_mnist.yaml)

Furthermore, JAX's design is very clean and has been widely appreciated. For instance, JAX is my favorite open-source project. I've tweeted about [how JAX is better than PyTorch](https://twitter.com/ayaka14732/status/1688194164033462272).

### 2.4. I know TPU is great now. Can I touch a TPU?

Unfortunately, we generally can't physically touch a real TPU. TPUs are meant to be accessed via Google Cloud services.

In some exhibitions, TPUs are [displayed for viewing](https://twitter.com/walkforhours/status/1696654844134822130), which might be the closest you can get to physically touching one.

Perhaps only by becoming a Google Cloud Infrastructure Engineer can one truly feel the touch of a TPU.

### 2.5. What does it mean to create a TPU instance? What do I actually get?

After creating a TPU v3-8 instance on [Google Cloud Platform](https://cloud.google.com/tpu), you'll get a cloud server running the Ubuntu system with sudo privileges, 96 CPU cores, 335 GiB memory, and a TPU device with 8 cores (totalling 128 GiB TPU memory).

![](assets/0.png)

In fact, this is similar to how we use GPUs. Typically, when we use a GPU, we are using a Linux server connected to the GPU. Similarly, when we use a TPU, we're using a server connected to the TPU.

## 3. Introduction to the TRC Program

### 3.1. How do I apply for the TRC program?

Apart from the TRC program's [homepage](https://sites.research.google/trc/about/), Shawn wrote a wonderful article about the TRC program on [google/jax#2108](https://github.com/google/jax/issues/2108#issuecomment-866238579). Anyone who is interested in TPU should read it immediately.

### 3.2. Is it really free?

For the first three months, the TRC program is completely free due to the free trial credit given when registering for Google Cloud. After three months, I spend roughly HK$13.95 (about US$1.78) per month. This expense is for the network traffic of the TPU server, while the TPU device itself is provided for free by the TRC program.

## 4. Using TPU VM

### 4.1. Create a TPU VM

Open [Google Cloud Platform](https://cloud.google.com/tpu) and navigate to the [TPU Management Page](https://console.cloud.google.com/compute/tpus).

![](assets/1.png)

Click the console button on the top-right corner to activate Cloud Shell.

In Cloud Shell, type the following command to create a Cloud TPU v3-8 VM:

```sh
until gcloud alpha compute tpus tpu-vm create node-1 --project tpu-develop --zone europe-west4-a --accelerator-type v3-8 --version tpu-vm-base ; do : ; done
```

Here, `node-1` is the name of the TPU VM you want to create, and `--project` is the name of your Google Cloud project.

The above command will repeatedly attempt to create the TPU VM until it succeeds.

### 4.2. Add an SSH public key to Google Cloud

For Google Cloud's servers, if you want to SSH into them, using `ssh-copy-id` is the wrong approach. The correct method is:

First, type “SSH keys” into the Google Cloud webpage search box, go to the relevant page, then click edit, and add your computer's SSH public key.

To view your computer's SSH public key:

```sh
cat ~/.ssh/id_rsa.pub
```

If you haven't created an SSH key pair yet, use the following command to create one, then execute the above command to view:

```sh
ssh-keygen -t rsa -f ~/.ssh/id_rsa -N ""
```

When adding an SSH public key to Google Cloud, it's crucial to pay special attention to the value of the username. In the SSH public key string, the part preceding the `@` symbol at the end is the username. When added to Google Cloud, it will create a user with that name on all servers for the current project. For instance, with the string `ayaka@instance-1`, Google Cloud will create a user named `ayaka` on the server. If you wish for Google Cloud to create a different username, you can manually modify this string. Changing the mentioned string to `nixie@instance-1` would lead Google Cloud to create a user named `nixie`. Moreover, making such changes won't affect the functionality of the SSH key.

### 4.3. SSH into TPU VM

Create or edit your computer's `~/.ssh/config`:

```sh
nano ~/.ssh/config
```

Add the following content:

```
Host tpuv3-8-1
    User nixie
    Hostname 34.141.220.156
```

Here, `tpuv3-8-1` is an arbitrary name, `User` is the username created in Google Cloud from the previous step, and `Hostname` is the IP address of the TPU VM.

Then, on your own computer, use the following command to SSH into the TPU VM:

```sh
ssh tpuv3-8-1
```

Where `tpuv3-8-1` is the name set in `~/.ssh/config`.

### 4.4. Verify that TPU VM has TPU

```sh
ls /dev/accel*
```

If the following output appears:

```
/dev/accel0  /dev/accel1  /dev/accel2  /dev/accel3
```

This indicates that the TPU VM indeed has a TPU.

### 4.5. Setting up the development environment in TPU VM

Update software packages:

```sh
sudo apt-get update -y -qq
sudo apt-get upgrade -y -qq
sudo apt-get install -y -qq golang neofetch zsh byobu
```

Install the latest Python 3.11:

```sh
sudo apt-get install -y -qq software-properties-common
sudo add-apt-repository -y ppa:deadsnakes/ppa
sudo apt-get install -y -qq python3.11-full python3.11-dev
```

Install Oh My Zsh:

```sh
sh -c "$(curl -fsSL https://raw.githubusercontent.com/ohmyzsh/ohmyzsh/master/tools/install.sh)" "" --unattended
sudo chsh $USER -s /usr/bin/zsh
```

Create a virtual environment (venv):

```sh
python3.11 -m venv ~/venv
```

Activate the venv:

```sh
. ~/venv/bin/activate
```

Install JAX in the venv:

```sh
pip install -U pip
pip install -U wheel
pip install -U "jax[tpu]" -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
```

### 4.6. Verify JAX is working properly

After activating the venv, use the following command to verify JAX is working:

```sh
python -c 'import jax; print(jax.devices())'
```

If the output contains `TpuDevice`, this means JAX is working as expected.

### 4.7. Using Byobu to ensure continuous program execution

Many tutorials use the method of appending `&` to commands to run them in the background, so they continue executing even after exiting SSH. However, this is a basic method. The correct approach is to use a window manager like Byobu.

To run Byobu, simply use the `byobu` command. Then, execute commands within the opened window. To close the window, you can forcefully close the current window on your computer. Byobu will continue running on the server. The next time you connect to the server, you can retrieve the previous window using the `byobu` command.

Byobu has many advanced features. You can learn them by watching the official video [Learn Byobu while listening to Mozart](https://youtu.be/NawuGmcvKus).

### 4.8. Configure VSCode Remote-SSH

Open VSCode, access the Extensions panel on the left, search and install Remote - SSH.

Press <kbd>F1</kbd> to open the command palette. Type ssh, click "Remote-SSH: Connect to Host...", then click on the server name set in `~/.ssh/config` (e.g., `tpuv3-8-1`). Once VSCode completes the setup on the server, you can develop directly on the server with VSCode.

![](assets/3.png)

On your computer, you can use the following command to quickly open a directory on the server:

```sh
code --remote ssh-remote+tpuv3-8-1 /home/ayaka/tpu-starter
```

This command will open the directory `/home/ayaka/tpu-starter` on `tpuv3-8-1` using VSCode.

## 5. Using TPU Pod

### 5.1. Create a subnet

To create a TPU Pod, you first need to create a new VPC network and then create a subnet in the corresponding area of that network (e.g., `europe-west4-a`).

TODO: Purpose?

### 5.2. Disable Cloud Logging

TODO: Reason? Steps?

### 5.3. Create TPU Pod

Open Cloud Shell using the method described earlier for creating the TPU VM and use the following command to create a TPU v3-32 Pod:

```sh
until gcloud alpha compute tpus tpu-vm create node-1 --project tpu-advanced-research --zone europe-west4-a --accelerator-type v3-32 --version v2-alpha-pod --network advanced --subnetwork advanced-subnet-for-europe-west4 ; do : ; done
```

Where `node-1` is the name you want for the TPU VM, `--project` is the name of your Google Cloud project, and `--network` and `--subnetwork` are the names of the network and subnet created in the previous step.

### 5.4. SSH into TPU Pod

Since the TPU Pod consists of multiple hosts, we need to choose one host, designate it as Host 0, and then SSH into Host 0 to execute commands. Given that the SSH public key added on the Google Cloud web page will be propagated to all hosts, every host can be directly connected through the SSH key, allowing us to designate any host as Host 0. The method to SSH into Host 0 is the same as for the aforementioned TPU VM.

### 5.5. Modify the SSH configuration file on Host 0

After SSH-ing into Host 0, the following configurations need to be made:

```sh
nano ~/.ssh/config
```

Add the following content:

```
Host 172.21.12.*
    StrictHostKeyChecking no
    UserKnownHostsFile /dev/null
    LogLevel ERROR
```

Here, `172.21.12.*` is determined by the IP address range of the subnet created in the previous steps. We use `172.21.12.*` because the IP address range specified when creating the subnet was 172.21.12.0/24.

We need to do so because the `known_hosts` in ssh is created for preventing man-in-the-middle attacks. Since we are using an internal network environment here, we don't need to prevent such attacks or require this file, so we direct it to `/dev/null`. Additionally, having `known_hosts` requires manually confirming the server's fingerprint during the first connection, which is unnecessary in an internal network environment and is not conducive to automation.

Then, run the following command to modify the permissions of this configuration file. If the permissions are not modified, the configuration file will not take effect:

```sh
chmod 600 ~/.ssh/config
```

### 5.6. Add the SSH public key of Host 0 to all hosts

First, follow the above steps to generate a key pair on Host 0. Then add the generated public key to Google Cloud's SSH keys, and this public key will be automatically propagated to all hosts.

### 5.7. Configure the `podrun` command

The `podrun` command is a tool under development. When executed on Host 0, it can run commands on all hosts via SSH.

Download `podrun`:

```sh
wget https://raw.githubusercontent.com/ayaka14732/llama-2-jax/d8220b8c95789b14fe55417edc1d9482389aa2c4/podrun
chmod +x podrun
```

Save the internal IP addresses of the other hosts in `~/podips.txt` (one per line). To edit `~/podips.txt`, use the following command:

```sh
nano ~/podips.txt
```

Enter venv and install Paramiko:

```sh
. ~/venv/bin/activate
pip install paramiko
```

Use `podrun` to make all hosts purr like a kitty:

```sh
./podrun -iw -- echo meow
```

### 5.8. Configure NFS

Install the NFS server and client:

```sh
./podrun -- DEBIAN_FRONTEND=noninteractive sudo apt-get install -y -qq nfs-common
sudo apt install -y -qq nfs-kernel-server
sudo mkdir -p /nfs_share
sudo chown -R nobody:nogroup /nfs_share
sudo chmod 777 /nfs_share
```

Modify `/etc/exports`:

```sh
sudo nano /etc/exports
```

Add:

```
/nfs_share  172.21.12.0/24(rw,sync,no_subtree_check)
```

Execute:

```sh
sudo exportfs -a
sudo systemctl restart nfs-kernel-server

./podrun -- sudo mkdir -p /nfs_share
./podrun -- sudo mount 172.21.12.2:/nfs_share /nfs_share
./podrun -- ln -sf /nfs_share ~/nfs_share

cd ~/nfs_share
touch meow
./podrun -iw -- ls ~/nfs_share/meow
```

### 5.9. Setting up the development environment in TPU Pod

TODO: Refer to the steps in setting up the development environment in the TPU VM above, but each command should use `podrun -iw --` to run on all hosts.

### 5.10. Verify JAX is working properly

```sh
~/podrun -icw -- ~/venv/bin/python -c 'import jax; jax.distributed.initialize(); jax.process_index() == 0 and print(jax.devices())'
```

If the output contains `TpuDevice`, this means JAX is working as expected.

## 6. TPU Best Practices

### 6.1. Prefer Google Cloud Platform to Google Colab

[Google Colab](https://colab.research.google.com/) only provides TPU v2-8 devices, while on [Google Cloud Platform](https://cloud.google.com/tpu) you can select TPU v2-8 and TPU v3-8.

Besides, on Google Colab you can only use TPU through the Jupyter Notebook interface. Even if you [log in into the Colab server via SSH](https://ayaka.shn.hk/colab/), it is a docker image and you don't have root access. On Google Cloud Platform, however, you have full access to the TPU VM.

If you really want to use TPU on Google Colab, you need to run [the following script](01-basics/setup_colab_tpu.py) to set up TPU:

```python
import jax
from jax.tools.colab_tpu import setup_tpu

setup_tpu()

devices = jax.devices()
print(devices)  # should print TpuDevice
```

### 6.2. Prefer TPU VM to TPU node

When you are creating a TPU instance, you need to choose between TPU VM and TPU node. Always prefer TPU VM because it is the new architecture in which TPU devices are connected to the host VM directly. This will make it easier to set up the TPU device.

## 7. JAX Best Practices

### 7.1. Import convention

You may see two different kind of import conventions. One is to import `jax.numpy` as `np` and import the original numpy as `onp`. Another one is to import `jax.numpy` as `jnp` and leave original numpy as `np`.

On 16 Jan 2019, Colin Raffel wrote in [a blog article](https://colinraffel.com/blog/you-don-t-know-jax.html) that the convention at that time was to import original numpy as `onp`.

On 5 Nov 2020, Niru Maheswaranathan said in [a tweet](https://twitter.com/niru_m/status/1324078070546882560) that he thinks the convention at that time was to import `jax.numpy` as `jnp` and to leave original numpy as `np`.

We can conclude that the new convention is to import `jax.numpy` as `jnp`.

### 7.2. Manage random keys in JAX

The regular way is this:

```python
key, *subkey = rand.split(key, num=4)
print(subkey[0])
print(subkey[1])
print(subkey[2])
```

### 7.3. Serialize model parameters

Normally, the model parameters are represented by a nested dictionary like this:

```python
{
    "embedding": DeviceArray,
    "ff1": {
        "kernel": DeviceArray,
        "bias": DeviceArray
    },
    "ff2": {
        "kernel": DeviceArray,
        "bias": DeviceArray
    }
}
```

You can use [`flax.serialization.msgpack_serialize`](https://flax.readthedocs.io/en/latest/flax.serialization.html#flax.serialization.msgpack_serialize) to serialize the parameters into bytes, and use [`flax.serialization.msgpack_restore`](https://flax.readthedocs.io/en/latest/flax.serialization.html#flax.serialization.msgpack_serialize) to convert them back.

### 7.4. Conversion between NumPy arrays and JAX arrays

Use [`np.asarray`](https://jax.readthedocs.io/en/latest/_autosummary/jax.numpy.asarray.html) and [`onp.asarray`](https://numpy.org/doc/stable/reference/generated/numpy.asarray.html).

```python
import jax.numpy as np
import numpy as onp

a = np.array([1, 2, 3])  # JAX array
b = onp.asarray(a)  # converted to NumPy array

c = onp.array([1, 2, 3])  # NumPy array
d = np.asarray(c)  # converted to JAX array
```

### 7.5. Conversion between PyTorch tensors and JAX arrays

Convert a PyTorch tensor to a JAX array:

```python
import jax.numpy as np
import torch

a = torch.rand(2, 2)  # PyTorch tensor
b = np.asarray(a.numpy())  # JAX array
```

Convert a JAX array to a PyTorch tensor:

```python
import jax.numpy as np
import numpy as onp
import torch

a = np.zeros((2, 2))  # JAX array
b = torch.from_numpy(onp.asarray(a))  # PyTorch tensor
```

This will result in a warning:

```
UserWarning: The given NumPy array is not writable, and PyTorch does not support non-writable tensors. This means writing to this tensor will result in undefined behavior. You may want to copy the array to protect its data or make it writable before converting it to a tensor. This type of warning will be suppressed for the rest of this program. (Triggered internally at  ../torch/csrc/utils/tensor_numpy.cpp:178.)
```

If you need writable tensors, you can use `onp.array` instead of `onp.asarray` to make a copy of the original array.

### 7.6. Type annotation

[google/jaxtyping](https://github.com/google/jaxtyping)

### 7.7. Check if an array is either a NumPy array or a JAX array

```python
isinstance(a, (np.ndarray, onp.ndarray))
```

### 7.8. Get the shapes of all parameters in a nested dictionary

```python
jax.tree_map(lambda x: x.shape, params)
```

### 7.9. The correct way to generate random numbers on CPU

Use the [jax.default_device()](https://jax.readthedocs.io/en/latest/_autosummary/jax.default_device.html) context manager:

```python
import jax
import jax.random as rand

device_cpu = jax.devices('cpu')[0]
with jax.default_device(device_cpu):
    key = rand.PRNGKey(42)
    a = rand.poisson(key, 3, shape=(1000,))
    print(a.device())  # TFRT_CPU_0
```

See <https://github.com/google/jax/discussions/9691#discussioncomment-3650311>.

### 7.10. Use optimizers from Optax

### 7.11. Use the cross-entropy loss implementation from Optax

`optax.softmax_cross_entropy_with_integer_labels`

## 8. How Can I...

### 8.1. Run Jupyter Notebook on TPU VM

After setting up Remote-SSH, you can work with Jupyter notebook files in VSCode.

Alternatively, you can run a regular Jupyter Notebook server on the TPU VM, forward the port to your PC and connect to it. However, you should prefer VSCode because it is more powerful, offers better integration with other tools and is easier to set up.

### 8.2. Share files across multiple TPU VM instances

TPU VM instances in the same zone are connected with internal IPs, so you can [create a shared file system using NFS](https://tecadmin.net/how-to-install-and-configure-an-nfs-server-on-ubuntu-20-04/).

### 8.3. Monitor TPU usage

[jax-smi](https://github.com/ayaka14732/jax-smi)

### 8.4. Start a server on TPU VM

Example: Tensorboard

Although every TPU VM is allocated with a public IP, in most cases you should expose a server to the Internet because it is insecure.

Port forwarding via SSH

```
ssh -C -N -L 127.0.0.1:6006:127.0.0.1:6006 tpu1
```

### 8.5. Run separate processes on different TPU cores

https://gist.github.com/skye/f82ba45d2445bb19d53545538754f9a3

## 9. Common Gotchas

### 9.1. TPU VMs will be rebooted occasionally

As of 24 Oct 2022, the TPU VMs will be rebooted occasionally if there is a maintenance event.

The following things will happen:

1. All the running processes will be terminated
2. The external IP address will be changed

We can save the model parameters, optimiser states and other useful data occasionally, so that the model training can be easily resumed after termination.

We should use `gcloud` command instead of connect directly to it with SSH. If we have to use SSH (e.g. if we want to use VSCode, SSH is the only choice), we need to manually change the target IP address.

### 9.2. One TPU core can only be used by one process at a time

See also: §10.5.

Unlike GPU, you will get an error if you run two processes on TPU at a time:

```
I0000 00:00:1648534265.148743  625905 tpu_initializer_helper.cc:94] libtpu.so already in use by another process. Run "$ sudo lsof -w /dev/accel0" to figure out which process is using the TPU. Not attempting to load libtpu.so in this process.
```

### 9.3. TCMalloc breaks several programs

[TCMalloc](https://github.com/google/tcmalloc) is Google's customized memory allocation library. On TPU VM, `LD_PRELOAD` is set to use TCMalloc by default:

```sh
$ echo LD_PRELOAD
/usr/lib/x86_64-linux-gnu/libtcmalloc.so.4
```

However, using TCMalloc in this manner may break several programs like gsutil:

```sh
$ gsutil --help
/snap/google-cloud-sdk/232/platform/bundledpythonunix/bin/python3: /snap/google-cloud-sdk/232/platform/bundledpythonunix/bin/../../../lib/x86_64-linux-gnu/libm.so.6: version `GLIBC_2.29' not found (required by /usr/lib/x86_64-linux-gnu/libtcmalloc.so.4)
```

The [homepage of TCMalloc](http://goog-perftools.sourceforge.net/doc/tcmalloc.html) also indicates that `LD_PRELOAD` is tricky and this mode of usage is not recommended.

If you encounter problems related to TCMalloc, you can disable it in the current shell using the command:

```sh
unset LD_PRELOAD
```

### 9.4. `libtpu.so` already in used by another process

```sh
if ! pgrep -a -u $USER python ; then
    killall -q -w -s SIGKILL ~/.venv311/bin/python
fi
rm -rf /tmp/libtpu_lockfile /tmp/tpu_logs
```

See also <https://github.com/google/jax/issues/9220#issuecomment-1015940320>.

### 9.5. JAX does not support the multiprocessing `fork` strategy

Use the `spawn` or `forkserver` strategies.

See <https://github.com/google/jax/issues/1805#issuecomment-561244991>.
