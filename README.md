# TPU Starter

Everything you want to know about Google Cloud TPU

* [1. Introduction](#1-introduction)
    * [1.1. Why TPU?](#11-why-tpu)
    * [1.2. TPU is so good, why haven't I seen many people using it?](#12-tpu-is-so-good-why-havent-i-seen-many-people-using-it)
    * [1.3. I know TPU is good now. Can I touch a real TPU?](#13-i-know-tpu-is-good-now-can-i-touch-a-real-tpu)
    * [1.4. How do I get access to TPU?](#14-how-do-i-get-access-to-tpu)
    * [1.5. What does it mean to create a TPU instance? What do I actually get?](#15-what-does-it-mean-to-create-a-tpu-instance-what-do-i-actually-get)
    * [1.6. How to apply for the TRC program?](#16-how-to-apply-for-the-trc-program)
* [2. Environment Setup](#2-environment-setup)
    * [2.1. Modify VPC firewall](#21-modify-vpc-firewall)
    * [2.2. Create a TPU instance](#22-create-a-tpu-instance)
    * [2.3. Add public key to the server](#23-add-public-key-to-the-server)
    * [2.4. Basic configurations](#24-basic-configurations)
        * [2.4.1. Install common packages](#241-install-common-packages)
        * [2.4.2. Install Python 3.10](#242-install-python-310)
        * [2.4.3. Create a virtual environment](#243-create-a-virtual-environment)
        * [2.4.4. Install JAX with TPU support](#244-install-jax-with-tpu-support)
        * [2.4.5. Install common libraries](#245-install-common-libraries)
    * [2.5. How can I verify that the TPU is working?](#25-how-can-i-verify-that-the-tpu-is-working)
    * [2.6. Set up development environment](#26-set-up-development-environment)
        * [2.6.1. Install Oh My Zsh](#261-install-oh-my-zsh)
        * [2.6.2. Set up Mosh and Byobu](#262-set-up-mosh-and-byobu)
        * [2.6.3. Set up VSCode Remote-SSH](#263-set-up-vscode-remote-ssh)
* [3. JAX Basics](#3-jax-basics)
    * [3.1. Why JAX?](#31-why-jax)
    * [3.2. Compute gradients with jax.grad](#32-compute-gradients-with-jaxgrad)
    * [3.3. Use optimizers from Optax](#33-use-optimizers-from-optax)
    * [3.4. Load training data to CPU, then send batches to TPU](#34-load-training-data-to-cpu-then-send-batches-to-tpu)
    * [3.5. Integration with Hugging Face Transformers](#35-integration-with-hugging-face-transformers)
* [4. Best Practices](#4-best-practices)
    * [4.1. About TPU](#41-about-tpu)
        * [4.1.1. Prefer Google Cloud Platform to Google Colab](#411-prefer-google-cloud-platform-to-google-colab)
        * [4.1.2. Prefer TPU VM to TPU node](#412-prefer-tpu-vm-to-tpu-node)
        * [4.1.3. Share files across multiple TPU VM instances](#413-share-files-across-multiple-tpu-vm-instances)
        * [4.1.4. Monitor TPU usage](#414-monitor-tpu-usage)
    * [4.2. About JAX](#42-about-jax)
        * [4.2.1. Import convention](#421-import-convention)
        * [4.2.2. Manage random keys in JAX](#422-manage-random-keys-in-jax)
        * [4.2.3. Serialize model parameters](#423-serialize-model-parameters)
        * [4.2.4. Convertion between NumPy array and JAX array](#424-convertion-between-numpy-array-and-jax-array)
        * [4.2.5. Type annotation](#425-type-annotation)
        * [4.2.6. Check an array is either a NumPy array or a JAX array](#426-check-an-array-is-either-a-numpy-array-or-a-jax-array)
        * [4.2.7. Check the shapes of all parameters in a nested dictionary](#427-check-the-shapes-of-all-parameters-in-a-nested-dictionary)
* [5. Confusing Syntax](#5-confusing-syntax)
    * [5.1. What is a[:, None]?](#51-what-is-a-none)
    * [5.2. How to understand np.einsum?](#52-how-to-understand-npeinsum)
* [6. Common Gotchas](#6-common-gotchas)
    * [6.1. About TPU](#61-about-tpu)
        * [6.1.1. External IP of TPU machine changes occasionally](#611-external-ip-of-tpu-machine-changes-occasionally)
        * [6.1.2. One TPU device can only be used by one process at a time](#612-one-tpu-device-can-only-be-used-by-one-process-at-a-time)
        * [6.1.3. TCMalloc breaks several programs](#613-tcmalloc-breaks-several-programs)
        * [6.1.4. There is no TPU counterpart of nvidia-smi](#614-there-is-no-tpu-counterpart-of-nvidia-smi)
    * [6.2. About JAX](#62-about-jax)
        * [6.2.1. Indexing an array with an array](#621-indexing-an-array-with-an-array)
        * [6.2.2. np.dot and torch.dot are different](#622-npdot-and-torchdot-are-different)
        * [6.2.3. np.std and torch.std are different](#623-npstd-and-torchstd-are-different)
        * [6.2.4. Computations on TPU are in low precision by default](#624-computations-on-tpu-are-in-low-precision-by-default)
* [7. Community](#7-community)

<!-- Created by https://github.com/ekalinin/github-markdown-toc -->

## 1. Introduction

### 1.1. Why TPU?

**TL;DR**: TPU is to GPU as GPU is to CPU.

TPU is a special hardware designed specifically for machine learning. There is a [performance comparison](https://github.com/huggingface/transformers/blob/main/examples/flax/language-modeling/README.md#runtime-evaluation) in Hugging Face Transformers:

![](assets/5.png)

Moreover, for researchers, [the TRC program](https://sites.research.google/trc/about/) provides free TPU. As far as I know, this is the best computing resource available for research. For more details on the TRC program, please see below.

### 1.2. TPU is so good, why haven't I seen many people using it?

If you want to use PyTorch, TPU may not be suitable for you. TPU is poorly supported by PyTorch. In one of my experiments, one batch took about 14 seconds to run on CPU, but over 4 hours to run on TPU. Twitter user @mauricetpunkt also thinks [PyTorch's performance on TPUs is bad](https://twitter.com/mauricetpunkt/status/1506944350281945090).

Another problem is that although a single TPU v3-8 device has 8 cores (16 GiB memory for each core), you need to write extra code to make use of all the 8 cores (see [named axes and easy-to-revise parallelism](https://jax.readthedocs.io/en/latest/notebooks/xmap_tutorial.html) in the JAX documentation). Otherwise, only the first core is used.

### 1.3. I know TPU is good now. Can I touch a real TPU?

Unfortunately, in most cases you cannot touch a TPU physically. TPU is only available through cloud services.

### 1.4. How do I get access to TPU?

You can create TPU instances on [Google Cloud Platform](https://cloud.google.com/tpu). For more information on setting up TPU, please see below.

You can also use [Google Colab](https://colab.research.google.com/), but I don't recommend this way. Moreover, if you get free access to TPU from the [TRC program](https://sites.research.google/trc/about/), you will be using Google Cloud Platform, not Google Colab.

### 1.5. What does it mean to create a TPU instance? What do I actually get?

After creating a TPU v3-8 instance on [Google Cloud Platform](https://cloud.google.com/tpu), you will get a Ubuntu 20.04 cloud server with sudo access, 96 cores, 335 GiB memory and one TPU device with 8 cores (128 GiB TPU memory in total).

![](assets/0.png)

This is similar to the way we use GPU. In most cases, when you use a GPU, you use a Linux server that connects with a GPU. When you use a TPU, you use a Linux server that connects with a TPU.

### 1.6. How to apply for the TRC program?

You can learn more about the TRC program on its [homepage](https://sites.research.google/trc/about/). Shawn has written a wonderful article about the TRC program in [google/jax#2108](https://github.com/google/jax/issues/2108#issuecomment-866238579). Anyone who is interested in TPU should read it immediately.

## 2. Environment Setup

### 2.1. Modify VPC firewall

You need to loosen the restrictions of the firewall so that Mosh and other programs will not be blocked.

Open the [Firewall management page](https://console.cloud.google.com/networking/firewalls/list) in VPC network.

Click the button to create a new firewall rule.

![](assets/2.png)

Set name to 'allow-all', targets to 'All instances in the network', source filter to 0.0.0.0/0, protocols and ports to 'Allow all', and then click 'Create'.

### 2.2. Create a TPU instance

Open [Google Cloud Platform](https://cloud.google.com/tpu), navigate to the [TPU management page](https://console.cloud.google.com/compute/tpus).

![](assets/1.png)

Click the console button on the top-right corner to activate Cloud Shell.

In Cloud Shell, type the following command to create a Cloud TPU VM v3-8 with TPU software version v2-nightly20210914:

```sh
gcloud alpha compute tpus tpu-vm create node-1 --project tpu-develop --zone=europe-west4-a --accelerator-type=v3-8 --version=v2-nightly20210914
```

If the command fails because there are no more TPUs to allocate, you can re-run the command again.

### 2.3. Add public key to the server

In Cloud Shell, login to the Cloud VM by the `gcloud` command:

```sh
gcloud alpha compute tpus tpu-vm ssh node-1 --zone europe-west4-a
```

After logging in, add your public key to `~/.ssh/authorized_keys`.

### 2.4. Basic configurations

#### 2.4.1. Install common packages

```sh
sudo apt update
sudo apt upgrade
sudo apt install -y neofetch zsh mosh byobu
sudo reboot
```

#### 2.4.2. Install Python 3.10

Unfortunately, [Python shipped with Ubuntu 20.04 LTS is Python 3.8](https://wiki.ubuntu.com/FocalFossa/ReleaseNotes#Python3_by_default), so you need to install Python 3.10 manually.

```sh
sudo apt install -y software-properties-common
sudo add-apt-repository ppa:deadsnakes/ppa
sudo apt install -y python3.10 python3.10-distutils python3.10-dev
curl -sS https://bootstrap.pypa.io/get-pip.py | sudo python3.10
```

#### 2.4.3. Create a virtual environment

```sh
python3.10 -m pip install virtualenv
python3.10 -m virtualenv ~/.venv310
. ~/.venv310/bin/activate
```

#### 2.4.4. Install JAX with TPU support

```sh
pip install -U pip
pip install -U wheel
pip install "jax[tpu]==0.3.4" -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
```

#### 2.4.5. Install common libraries

Clone this repository. In the root directory of this repository, run:

```sh
pip install -r requirements.txt
```

### 2.5. How can I verify that the TPU is working?

```python
import jax.numpy as np
a = np.array([1, 2, 3])
print(a.device())  # should print TpuDevice
```

### 2.6. Set up development environment

#### 2.6.1. Install Oh My Zsh

[Oh My Zsh](https://ohmyz.sh/) makes the terminal much easier to use.

To install Oh My Zsh, run the following command:

```sh
sh -c "$(curl -fsSL https://raw.github.com/ohmyzsh/ohmyzsh/master/tools/install.sh)"
```

#### 2.6.2. Set up Mosh and Byobu

If you connect to the server directly with SSH, there is a risk of loss of connection. If this happens, the training script you are running in the foreground will be terminated.

[Mosh](https://mosh.org/) and [Byobu](https://www.byobu.org/) are two programs to solve this problem. Byobu will ensure that the script continues to run on the server even if the connection is lost, while Mosh guarantees that the connection will not be lost. 

Install [Mosh](https://mosh.org/#getting) on your local device, then log in into the server with:

```sh
mosh tpu1 -- byobu
```

You can learn more about Byobu from the video [Learn Byobu while listening to Mozart](https://youtu.be/NawuGmcvKus).

#### 2.6.3. Set up VSCode Remote-SSH

Open VSCode. Open the 'Extensions' panel on the left. Search for 'Remote - SSH' and install.

Press <kbd>F1</kbd> to open the command palette. Type 'ssh', then select 'Remote-SSH: Connect to Host...'. Input the server name you would like to connect and press Enter.

Wait for VSCode to be set up on the server. After it is finished, you can develop on the server using VSCode.

## 3. JAX Basics

### 3.1. Why JAX?

The three popular deep learning libraries supported by [Hugging Face Transformers](https://github.com/huggingface/transformers) are [JAX](https://github.com/google/jax), [PyTorch](https://pytorch.org/) and [TensorFlow](https://www.tensorflow.org/).

As mentioned earlier, PyTorch is poorly supported on TPU. For Tensorflow and JAX, I regard JAX as the next generation and simplified version of Tensorflow. JAX is easier to use than Tensorflow.

JAX uses the same APIs as [NumPy](https://numpy.org/). There are also a number of mutually compatible libraries built on top of JAX. A comprehensive list of the JAX ecosystem can be found at [n2cholas/awesome-jax](https://github.com/n2cholas/awesome-jax).

### 3.2. Compute gradients with `jax.grad`

### 3.3. Use optimizers from Optax

### 3.4. Load training data to CPU, then send batches to TPU

### 3.5. Integration with Hugging Face Transformers

[Hugging Face Transformers](https://huggingface.co/docs/transformers/index)

## 4. Best Practices

### 4.1. About TPU

#### 4.1.1. Prefer Google Cloud Platform to Google Colab

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

#### 4.1.2. Prefer TPU VM to TPU node

When you are creating a TPU instance, you need to choose between TPU VM and TPU node. Always prefer TPU VM because it is the new architecture in which TPU devices are connected to the host VM directly. This will make it easier to set up the TPU device.

#### 4.1.3. Share files across multiple TPU VM instances

TPU VM instances in the same zone are connected with internal IPs, so you can [create a shared file system using NFS](https://tecadmin.net/how-to-install-and-configure-an-nfs-server-on-ubuntu-20-04/).

#### 4.1.4. Monitor TPU usage

### 4.2. About JAX

#### 4.2.1. Import convention

You may see two different kind of import conventions. One is to import jax.numpy as np and import the original numpy as onp. Another one is to import jax.numpy as jnp and leave original numpy as np.

On 16 Jan 2019, Colin Raffel wrote in [a blog article](https://colinraffel.com/blog/you-don-t-know-jax.html) that the convention at that time was to import original numpy as onp.

On 5 Nov 2020, Niru Maheswaranathan said in [a tweet](https://twitter.com/niru_m/status/1324078070546882560) that he thinks the convention at that time was to import jax as jnp and to leave original numpy as np.

TODO: Conclusion?

#### 4.2.2. Manage random keys in JAX

#### 4.2.3. Serialize model parameters

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

#### 4.2.4. Convertion between NumPy array and JAX array

Use [`np.asarray`](https://jax.readthedocs.io/en/latest/_autosummary/jax.numpy.asarray.html) and [`onp.asarray`](https://numpy.org/doc/stable/reference/generated/numpy.asarray.html).

```python
import jax.numpy as np
import numpy as onp

a = np.array([1, 2, 3])  # JAX array
b = onp.asarray(a)  # converted to NumPy array

c = onp.array([1, 2, 3])  # NumPy array
d = np.asarray(c)  # converted to JAX array
```

#### 4.2.5. Type annotation

`np.ndarray`

#### 4.2.6. Check an array is either a NumPy array or a JAX array

```python
isinstance(a, (np.ndarray, onp.ndarray))
```

#### 4.2.7. Check the shapes of all parameters in a nested dictionary

```python
jax.tree_map(lambda x: x.shape, params)
```

## 5. Confusing Syntax

### 5.1. What is `a[:, None]`?

[`np.newaxis`](https://numpy.org/doc/stable/reference/constants.html#numpy.newaxis)

### 5.2. How to understand `np.einsum`?

## 6. Common Gotchas

### 6.1. About TPU

#### 6.1.1. External IP of TPU machine changes occasionally

As of 17 Feb 2022, the external IP address may change if there is a maintenance event. If this happens, you need to reconnect with the new IP address.

#### 6.1.2. One TPU device can only be used by one process at a time

Unlike GPU, you will get an error if you run two processes on TPU at a time:

```
I0000 00:00:1648534265.148743  625905 tpu_initializer_helper.cc:94] libtpu.so already in use by another process. Run "$ sudo lsof -w /dev/accel0" to figure out which process is using the TPU. Not attempting to load libtpu.so in this process.
```

Even if a TPU device has 8 cores and one process only utilizes the first core, the other processes will not be able to utilize the rest of the cores.

#### 6.1.3. TCMalloc breaks several programs

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

#### 6.1.4. There is no TPU counterpart of `nvidia-smi`

See [google/jax#9756](https://github.com/google/jax/discussions/9756).

### 6.2. About JAX

#### 6.2.1. Indexing an array with an array

```python
import jax.numpy as np
import numpy as onp

a = onp.arange(12).reshape((6, 2))
b = onp.arange(6).reshape((2, 3))

a_ = np.asarray(a)
b_ = np.asarray(b)

a[b]  # success
a_[b_]  # success
a_[b]  # success
a[b_]  # error: index 3 is out of bounds for axis 1 with size 2
```

Generally speaking, JAX supports NumPy arrays, but NumPy does not support JAX arrays.

#### 6.2.2. `np.dot` and `torch.dot` are different

```python
import numpy as onp
import torch

a = onp.random.rand(3, 4, 5)
b = onp.random.rand(4, 5, 6)
onp.dot(a, b)  # success

a_ = torch.from_numpy(a)
b_ = torch.from_numpy(b)
torch.dot(a_, b_)  # error: 1D tensors expected, but got 3D and 3D tensors
```

#### 6.2.3. `np.std` and `torch.std` are different

```python
import torch

x = torch.tensor([[-1., 1.]])

print(x.std(-1).numpy())  # [1.4142135]
print(x.numpy().std(-1))  # [1.]
```

This is because in [`np.std`](https://numpy.org/doc/stable/reference/generated/numpy.std.html) the denominator is _n_, while in [`torch.std`](https://pytorch.org/docs/stable/generated/torch.std.html) it is _n_-1. See [pytorch/pytorch#1854](https://github.com/pytorch/pytorch/issues/1854) for details.

#### 6.2.4. Computations on TPU are in low precision by default

JAX uses bfloat16 for matrix multiplication on TPU by default, even if the data type is float32.

```python
import jax.numpy as np

print(4176 * 5996)  # 25039296

a = np.array(0.4176, dtype=np.float32)
b = np.array(0.5996, dtype=np.float32)
print((a * b).item())  # 0.25039297342300415
```

To do matrix multiplication in float32, you need to add this line at the top of the script:

```python
jax.config.update('jax_default_matmul_precision', jax.lax.Precision.HIGHEST)
```

Other precision values can be found in [jax.lax.Precision](https://jax.readthedocs.io/en/latest/jax.lax.html#jax.lax.Precision). See [google/jax#9973](https://github.com/google/jax/issues/9973) for details.

## 7. Community

As of 23 Feb 2022, there is no official chat group for Cloud TPUs. You can join my chat group [@cloudtpu](https://t.me/cloudtpu) on Telegram. [Shawn's Discord server](https://github.com/shawwn/tpunicorn#contact) also has a channel for TPU.
