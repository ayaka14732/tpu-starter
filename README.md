# TPU Starter

Everything you want to know about Google Cloud TPU

* [1. Community](#1-community)
* [2. Introduction to TPU](#2-introduction-to-tpu)
    * [2.1. Why TPU?](#21-why-tpu)
    * [2.2. TPU is so good, why haven't I seen many people using it?](#22-tpu-is-so-good-why-havent-i-seen-many-people-using-it)
    * [2.3. I know TPU is good now. Can I touch a real TPU?](#23-i-know-tpu-is-good-now-can-i-touch-a-real-tpu)
    * [2.4. How do I get access to TPU?](#24-how-do-i-get-access-to-tpu)
    * [2.5. What does it mean to create a TPU instance? What do I actually get?](#25-what-does-it-mean-to-create-a-tpu-instance-what-do-i-actually-get)
* [3. Introduction to the TRC Program](#3-introduction-to-the-trc-program)
    * [3.1. How to apply for the TRC program?](#31-how-to-apply-for-the-trc-program)
    * [3.2. Is it really free?](#32-is-it-really-free)
* [4. Create a TPU VM Instance](#4-create-a-tpu-vm-instance)
    * [4.1. Modify VPC firewall](#41-modify-vpc-firewall)
    * [4.2. Create the instance](#42-create-the-instance)
    * [4.3. Add public key to the server](#43-add-public-key-to-the-server)
* [5. Environment Setup](#5-environment-setup)
    * [5.1. Install common packages](#51-install-common-packages)
    * [5.2. Install Python 3.10](#52-install-python-310)
    * [5.3. Install Oh My Zsh](#53-install-oh-my-zsh)
    * [5.4. Change timezone](#54-change-timezone)
    * [5.5. Create a Virtualenv](#55-create-a-virtualenv)
    * [5.6. Install JAX with TPU support](#56-install-jax-with-tpu-support)
    * [5.7. Install common libraries](#57-install-common-libraries)
    * [5.8. Install Tensorflow and Tensorboard Plugin Profile](#58-install-tensorflow-and-tensorboard-plugin-profile)
    * [5.9. Set up Mosh and Byobu](#59-set-up-mosh-and-byobu)
    * [5.10. Set up VSCode Remote-SSH](#510-set-up-vscode-remote-ssh)
    * [5.11. How can I verify that the TPU is working?](#511-how-can-i-verify-that-the-tpu-is-working)
* [6. JAX Basics](#6-jax-basics)
    * [6.1. Why JAX?](#61-why-jax)
    * [6.2. Compute gradients with jax.grad](#62-compute-gradients-with-jaxgrad)
    * [6.3. Load training data to CPU, then send batches to TPU](#63-load-training-data-to-cpu-then-send-batches-to-tpu)
    * [6.4. Data parallelism on 8 TPU cores](#64-data-parallelism-on-8-tpu-cores)
        * [6.4.1. Basics of jax.pmap](#641-basics-of-jaxpmap)
        * [6.4.2. What if I want to have randomness in the update function?](#642-what-if-i-want-to-have-randomness-in-the-update-function)
        * [6.4.3. What if I want to use optax optimizers in the update function?](#643-what-if-i-want-to-use-optax-optimizers-in-the-update-function)
    * [6.5. Use optimizers from Optax](#65-use-optimizers-from-optax)
    * [6.6. Freeze certain model parameters](#66-freeze-certain-model-parameters)
    * [6.7. Integration with Hugging Face Transformers](#67-integration-with-hugging-face-transformers)
* [7. Best Practices](#7-best-practices)
    * [7.1. About TPU](#71-about-tpu)
        * [7.1.1. Prefer Google Cloud Platform to Google Colab](#711-prefer-google-cloud-platform-to-google-colab)
        * [7.1.2. Prefer TPU VM to TPU node](#712-prefer-tpu-vm-to-tpu-node)
        * [7.1.3. Run Jupyter Notebook on TPU VM](#713-run-jupyter-notebook-on-tpu-vm)
        * [7.1.4. Share files across multiple TPU VM instances](#714-share-files-across-multiple-tpu-vm-instances)
        * [7.1.5. Monitor TPU usage](#715-monitor-tpu-usage)
        * [7.1.6. Start a server on TPU VM](#716-start-a-server-on-tpu-vm)
    * [7.2. About JAX](#72-about-jax)
        * [7.2.1. Import convention](#721-import-convention)
        * [7.2.2. Manage random keys in JAX](#722-manage-random-keys-in-jax)
        * [7.2.3. Serialize model parameters](#723-serialize-model-parameters)
        * [7.2.4. Convertion between NumPy array and JAX array](#724-convertion-between-numpy-array-and-jax-array)
        * [7.2.5. Type annotation](#725-type-annotation)
        * [7.2.6. Check if an array is either a NumPy array or a JAX array](#726-check-if-an-array-is-either-a-numpy-array-or-a-jax-array)
        * [7.2.7. Get the shapes of all parameters in a nested dictionary](#727-get-the-shapes-of-all-parameters-in-a-nested-dictionary)
* [8. Confusing Syntax](#8-confusing-syntax)
    * [8.1. What is a[:, None]?](#81-what-is-a-none)
    * [8.2. How to understand np.einsum?](#82-how-to-understand-npeinsum)
* [9. Common Gotchas](#9-common-gotchas)
    * [9.1. About TPU](#91-about-tpu)
        * [9.1.1. External IP of TPU machine changes occasionally](#911-external-ip-of-tpu-machine-changes-occasionally)
        * [9.1.2. One TPU device can only be used by one process at a time](#912-one-tpu-device-can-only-be-used-by-one-process-at-a-time)
        * [9.1.3. TCMalloc breaks several programs](#913-tcmalloc-breaks-several-programs)
        * [9.1.4. There is no TPU counterpart of nvidia-smi](#914-there-is-no-tpu-counterpart-of-nvidia-smi)
    * [9.2. About JAX](#92-about-jax)
        * [9.2.1. Indexing an array with an array](#921-indexing-an-array-with-an-array)
        * [9.2.2. np.dot and torch.dot are different](#922-npdot-and-torchdot-are-different)
        * [9.2.3. np.std and torch.std are different](#923-npstd-and-torchstd-are-different)
        * [9.2.4. Computations on TPU are in low precision by default](#924-computations-on-tpu-are-in-low-precision-by-default)
        * [9.2.5. Weight matrix of linear layer is transposed in PyTorch](#925-weight-matrix-of-linear-layer-is-transposed-in-pytorch)

<!-- Created by https://github.com/ekalinin/github-markdown-toc -->

This project is inspired by [Cloud Run FAQ](https://github.com/ahmetb/cloud-run-faq), a community-maintained knowledge base of another Google Cloud product.

## 1. Community

As of 23 Feb 2022, there is no official chat group for Cloud TPUs. You can join the [@cloudtpu](https://t.me/cloudtpu) chat group on Telegram or [TPU Podcast](https://github.com/shawwn/tpunicorn#contact) on Discord, which are connected with each other.

## 2. Introduction to TPU

### 2.1. Why TPU?

**TL;DR**: TPU is to GPU as GPU is to CPU.

TPU is a special hardware designed specifically for machine learning. There is a [performance comparison](https://github.com/huggingface/transformers/blob/main/examples/flax/language-modeling/README.md#runtime-evaluation) in Hugging Face Transformers:

![](assets/5.png)

Moreover, for researchers, [the TRC program](https://sites.research.google/trc/about/) provides free TPU. As far as I know, if you have ever been concerned about the computing resources for training models, this is the best solution. For more details on the TRC program, see below.

### 2.2. TPU is so good, why haven't I seen many people using it?

If you want to use PyTorch, TPU may not be suitable for you. TPU is poorly supported by PyTorch. In one of my experiments, one batch took about 14 seconds to run on CPU, but over 4 hours to run on TPU. Twitter user @mauricetpunkt also thinks [PyTorch's performance on TPUs is bad](https://twitter.com/mauricetpunkt/status/1506944350281945090).

Another problem is that although a single TPU v3-8 device has 8 cores (16 GiB memory for each core), you need to write extra code to make use of all the 8 cores (see below). Otherwise, only the first core is used.

### 2.3. I know TPU is good now. Can I touch a real TPU?

Unfortunately, in most cases you cannot touch a TPU physically. TPU is only available through cloud services.

### 2.4. How do I get access to TPU?

You can create TPU instances on [Google Cloud Platform](https://cloud.google.com/tpu). For more information on setting up TPU, see below.

You can also use [Google Colab](https://colab.research.google.com/), but I don't recommend this way. Moreover, if you get free access to TPU from the [TRC program](https://sites.research.google/trc/about/), you will be using Google Cloud Platform, not Google Colab.

### 2.5. What does it mean to create a TPU instance? What do I actually get?

After creating a TPU v3-8 instance on [Google Cloud Platform](https://cloud.google.com/tpu), you will get a Ubuntu 20.04 cloud server with sudo access, 96 cores, 335 GiB memory and one TPU device with 8 cores (128 GiB TPU memory in total).

![](assets/0.png)

This is similar to the way we use GPU. In most cases, when you use a GPU, you use a Linux server that connects with a GPU. When you use a TPU, you use a Linux server that connects with a TPU.

## 3. Introduction to the TRC Program

### 3.1. How to apply for the TRC program?

Besides its [homepage](https://sites.research.google/trc/about/), Shawn has written a wonderful article about the TRC program in [google/jax#2108](https://github.com/google/jax/issues/2108#issuecomment-866238579). Anyone who is interested in TPU should read it immediately.

### 3.2. Is it really free?

At the first three months, it is completely free because all the fees are covered by Google Cloud free trial. After that, I pay only about HK$13.95 (approx. US$1.78) for one month for the outbound Internet traffic.

## 4. Create a TPU VM Instance

### 4.1. Modify VPC firewall

You need to loosen the restrictions of the firewall so that Mosh and other programs will not be blocked.

Open the [Firewall management page](https://console.cloud.google.com/networking/firewalls/list) in VPC network.

Click the button to create a new firewall rule.

![](assets/2.png)

Set name to 'allow-all', targets to 'All instances in the network', source filter to 0.0.0.0/0, protocols and ports to 'Allow all', and then click 'Create'.

### 4.2. Create the instance

Open [Google Cloud Platform](https://cloud.google.com/tpu), navigate to the [TPU management page](https://console.cloud.google.com/compute/tpus).

![](assets/1.png)

Click the console button on the top-right corner to activate Cloud Shell.

In Cloud Shell, type the following command to create a Cloud TPU VM v3-8 with TPU software version v2-nightly20210914:

```sh
gcloud alpha compute tpus tpu-vm create node-1 --project tpu-develop --zone=europe-west4-a --accelerator-type=v3-8 --version=v2-nightly20210914
```

If the command fails because there are no more TPUs to allocate, you can re-run the command again.

### 4.3. Add public key to the server

In Cloud Shell, login to the Cloud VM by the `gcloud` command:

```sh
gcloud alpha compute tpus tpu-vm ssh node-1 --zone europe-west4-a
```

After logging in, add your public key to `~/.ssh/authorized_keys`.

## 5. Environment Setup

This section assumes you have no previous knowledge about developing on a server. You can skip this section if you are already familiar with developing on a server and have your preferred setting.

### 5.1. Install common packages

```sh
sudo apt update
sudo apt upgrade -y
sudo apt install -y golang neofetch zsh mosh byobu
sudo reboot
```

### 5.2. Install Python 3.10

Unfortunately, [Python shipped with Ubuntu 20.04 LTS is Python 3.8](https://wiki.ubuntu.com/FocalFossa/ReleaseNotes#Python3_by_default), so you need to install Python 3.10 manually.

```sh
sudo apt install -y software-properties-common
sudo add-apt-repository -y ppa:deadsnakes/ppa
sudo apt install -y python3.10-full python3.10-dev
```

### 5.3. Install Oh My Zsh

[Oh My Zsh](https://ohmyz.sh/) makes the terminal much easier to use.

To install Oh My Zsh, run the following command:

```sh
sh -c "$(curl -fsSL https://raw.github.com/ohmyzsh/ohmyzsh/master/tools/install.sh)"
```

### 5.4. Change timezone

```sh
timedatectl list-timezones
sudo timedatectl set-timezone Asia/Hong_Kong  # change to your timezone
```

### 5.5. Create a Virtualenv

```sh
python3.10 -m venv ~/.venv310
source ~/.venv310/bin/activate
```

You need to run the `source` command every time you open a shell.

### 5.6. Install JAX with TPU support

```sh
pip install -U pip
pip install -U wheel
pip install "jax[tpu]==0.3.6" -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
```

### 5.7. Install common libraries

Clone this repository. In the root directory of this repository, run:

```sh
pip install -r requirements.txt
```

### 5.8. Install Tensorflow and Tensorboard Plugin Profile

Although we are using JAX, we need to install Tensorflow as well to make `jax.profiler` work. Otherwise you will get an error:

```
E external/org_tensorflow/tensorflow/python/profiler/internal/python_hooks.cc:369] Can't import tensorflow.python.profiler.trace
```

You cannot install Tensorflow in the regular way because it is not built with TPU support.

TODO: Add a installation method.

See [gist](https://gist.github.com/ayaka14732/a22234f394d60a28545f76cff23397c0).

### 5.9. Set up Mosh and Byobu

If you connect to the server directly with SSH, there is a risk of loss of connection. If this happens, the training script you are running in the foreground will be terminated.

[Mosh](https://mosh.org/) and [Byobu](https://www.byobu.org/) are two programs to solve this problem. Byobu will ensure that the script continues to run on the server even if the connection is lost, while Mosh guarantees that the connection will not be lost. 

Install [Mosh](https://mosh.org/#getting) on your local device, then log in into the server with:

```sh
mosh tpu1 -- byobu
```

You can learn more about Byobu from the video [Learn Byobu while listening to Mozart](https://youtu.be/NawuGmcvKus).

### 5.10. Set up VSCode Remote-SSH

Open VSCode. Open the 'Extensions' panel on the left. Search for 'Remote - SSH' and install.

Press <kbd>F1</kbd> to open the command palette. Type 'ssh', then select 'Remote-SSH: Connect to Host...'. Input the server name you would like to connect and press Enter.

Wait for VSCode to be set up on the server. After it is finished, you can develop on the server using VSCode.

![](assets/3.png)

### 5.11. How can I verify that the TPU is working?

Run this command:

```sh
python3 -c 'import jax; print(jax.devices())'  # should print TpuDevice
```

Note that we are using `python3` instead of `python` here, so the command also works even without activating Virtualenv.

You can also run this command to link `python` to `python3` by default, but I do not recommend it:

```sh
sudo apt install -y python-is-python3
```

This is because we should always use a Virtualenv to run our projects. When the `python` command is Python 2, if we forget to source Virtualenv, in most cases the command will fail, and this will remind us to source Virtualenv.

TODO: If TPU is not working...

See also <https://github.com/google/jax/issues/9220#issuecomment-1015940320>.

## 6. JAX Basics

### 6.1. Why JAX?

The three popular deep learning libraries supported by [Hugging Face Transformers](https://github.com/huggingface/transformers) are [JAX](https://github.com/google/jax), [PyTorch](https://pytorch.org/) and [TensorFlow](https://www.tensorflow.org/).

As mentioned earlier, PyTorch is poorly supported on TPU. For Tensorflow and JAX, I regard JAX as the next generation and simplified version of Tensorflow. JAX is easier to use than Tensorflow.

JAX uses the same APIs as [NumPy](https://numpy.org/). There are also a number of mutually compatible libraries built on top of JAX. A comprehensive list of the JAX ecosystem can be found at [n2cholas/awesome-jax](https://github.com/n2cholas/awesome-jax).

### 6.2. Compute gradients with `jax.grad`

### 6.3. Load training data to CPU, then send batches to TPU

### 6.4. Data parallelism on 8 TPU cores

#### 6.4.1. Basics of `jax.pmap`

There are four key points here.

1\. `params` and `opt_state` should be replicated across the devices:

```python
replicated_params = jax.device_put_replicated(params, jax.devices())
```

2\. `data` and `labels` should be split to the devices:

```python
n_devices = jax.device_count()
batch_size, *data_shapes = data.shape
assert batch_size % n_devices == 0, 'The data cannot be split evenly to the devices'
data = data.reshape(n_devices, batch_size // n_devices, *data_shapes)
```

3\. Decorate the target function with `jax.pmap`:

```
@partial(jax.pmap, axis_name='num_devices')
```

4\. In the `loss` function, use `jax.lax.pmean` to calculate the mean value across devices:

```python
grads = jax.lax.pmean(grads, axis_name='num_devices')  # calculate mean across devices
```

See [01-basics/test_pmap.py](01-basics/test_pmap.py) for a complete working example.

See also <https://jax.readthedocs.io/en/latest/jax-101/06-parallelism.html#example>.

#### 6.4.2. What if I want to have randomness in the update function?

```python
key, subkey = (lambda keys: (keys[0], keys[1:]))(rand.split(key, num=9))
```

Note that you cannot use the regular way to split the keys:

```python
key, *subkey = rand.split(key, num=9)
```

Because in this way, `subkey` is a list rather than an array.

#### 6.4.3. What if I want to use optax optimizers in the update function?

`opt_state` should be replicated as well.

### 6.5. Use optimizers from Optax

### 6.6. Freeze certain model parameters

Use [`optax.set_to_zero`](https://optax.readthedocs.io/en/latest/api.html#optax.set_to_zero) together with [`optax.multi_transform`](https://optax.readthedocs.io/en/latest/api.html#optax.multi_transform).

```python
params = {
    'a': { 'x1': ..., 'x2': ... },
    'b': { 'x1': ..., 'x2': ... },
}

param_labels = {
    'a': { 'x1': 'freeze', 'x2': 'train' },
    'b': 'train',
}

optimizer_scheme = {
    'train': optax.adam(...),
    'freeze': optax.set_to_zero(),
}

optimizer = optax.multi_transform(optimizer_scheme, param_labels)
```

See [Freeze Parameters Example](https://colab.research.google.com/drive/1-qLk5l09bq1NxZwwbu_yDk4W7da5TnFx) for details.

### 6.7. Integration with Hugging Face Transformers

[Hugging Face Transformers](https://huggingface.co/docs/transformers/index)

## 7. Best Practices

### 7.1. About TPU

#### 7.1.1. Prefer Google Cloud Platform to Google Colab

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

#### 7.1.2. Prefer TPU VM to TPU node

When you are creating a TPU instance, you need to choose between TPU VM and TPU node. Always prefer TPU VM because it is the new architecture in which TPU devices are connected to the host VM directly. This will make it easier to set up the TPU device.

#### 7.1.3. Run Jupyter Notebook on TPU VM

After setting up Remote-SSH, you can work with Jupyter notebook files in VSCode.

Alternatively, you can run a regular Jupyter Notebook server on the TPU VM, forward the port to your PC and connect to it. However, you should prefer VSCode because it is more powerful, offers better integration with other tools and is easier to set up.

#### 7.1.4. Share files across multiple TPU VM instances

TPU VM instances in the same zone are connected with internal IPs, so you can [create a shared file system using NFS](https://tecadmin.net/how-to-install-and-configure-an-nfs-server-on-ubuntu-20-04/).

#### 7.1.5. Monitor TPU usage

#### 7.1.6. Start a server on TPU VM

Example: Tensorboard

Although every TPU VM is allocated with a public IP, in most cases you should expose a server to the Internet because it is insecure.

Port forwarding via SSH

```
ssh -C -N -L 127.0.0.1:6006:127.0.0.1:6006 tpu1
```

### 7.2. About JAX

#### 7.2.1. Import convention

You may see two different kind of import conventions. One is to import jax.numpy as np and import the original numpy as onp. Another one is to import jax.numpy as jnp and leave original numpy as np.

On 16 Jan 2019, Colin Raffel wrote in [a blog article](https://colinraffel.com/blog/you-don-t-know-jax.html) that the convention at that time was to import original numpy as onp.

On 5 Nov 2020, Niru Maheswaranathan said in [a tweet](https://twitter.com/niru_m/status/1324078070546882560) that he thinks the convention at that time was to import jax as jnp and to leave original numpy as np.

TODO: Conclusion?

#### 7.2.2. Manage random keys in JAX

The regular way is this:

```python
key, *subkey = rand.split(key, num=4)
print(subkey[0])
print(subkey[1])
print(subkey[2])
```

#### 7.2.3. Serialize model parameters

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

#### 7.2.4. Convertion between NumPy array and JAX array

Use [`np.asarray`](https://jax.readthedocs.io/en/latest/_autosummary/jax.numpy.asarray.html) and [`onp.asarray`](https://numpy.org/doc/stable/reference/generated/numpy.asarray.html).

```python
import jax.numpy as np
import numpy as onp

a = np.array([1, 2, 3])  # JAX array
b = onp.asarray(a)  # converted to NumPy array

c = onp.array([1, 2, 3])  # NumPy array
d = np.asarray(c)  # converted to JAX array
```

#### 7.2.5. Type annotation

`np.ndarray`

#### 7.2.6. Check if an array is either a NumPy array or a JAX array

```python
isinstance(a, (np.ndarray, onp.ndarray))
```

#### 7.2.7. Get the shapes of all parameters in a nested dictionary

```python
jax.tree_map(lambda x: x.shape, params)
```

## 8. Confusing Syntax

### 8.1. What is `a[:, None]`?

[`np.newaxis`](https://numpy.org/doc/stable/reference/constants.html#numpy.newaxis)

### 8.2. How to understand `np.einsum`?

## 9. Common Gotchas

### 9.1. About TPU

#### 9.1.1. External IP of TPU machine changes occasionally

As of 17 Feb 2022, the external IP address may change if there is a maintenance event. If this happens, you need to reconnect with the new IP address.

#### 9.1.2. One TPU device can only be used by one process at a time

Unlike GPU, you will get an error if you run two processes on TPU at a time:

```
I0000 00:00:1648534265.148743  625905 tpu_initializer_helper.cc:94] libtpu.so already in use by another process. Run "$ sudo lsof -w /dev/accel0" to figure out which process is using the TPU. Not attempting to load libtpu.so in this process.
```

Even if a TPU device has 8 cores and one process only utilizes the first core, the other processes will not be able to utilize the rest of the cores.

#### 9.1.3. TCMalloc breaks several programs

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

#### 9.1.4. There is no TPU counterpart of `nvidia-smi`

See [google/jax#9756](https://github.com/google/jax/discussions/9756).

### 9.2. About JAX

#### 9.2.1. Indexing an array with an array

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

#### 9.2.2. `np.dot` and `torch.dot` are different

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

#### 9.2.3. `np.std` and `torch.std` are different

```python
import torch

x = torch.tensor([[-1., 1.]])

print(x.std(-1).numpy())  # [1.4142135]
print(x.numpy().std(-1))  # [1.]
```

This is because in [`np.std`](https://numpy.org/doc/stable/reference/generated/numpy.std.html) the denominator is _n_, while in [`torch.std`](https://pytorch.org/docs/stable/generated/torch.std.html) it is _n_-1. See [pytorch/pytorch#1854](https://github.com/pytorch/pytorch/issues/1854) for details.

#### 9.2.4. Computations on TPU are in low precision by default

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

#### 9.2.5. Weight matrix of linear layer is transposed in PyTorch

Weight matrix of linear layer is transposed in PyTorch, but not in Flax. Therefore, if you want to convert model parameters between PyTorch and Flax, you needed to transpose the weight matrices.

In Flax:

```python
import flax.linen as nn
import jax.numpy as np
import jax.random as rand
linear = nn.Dense(5)
key = rand.PRNGKey(42)
params = linear.init(key, np.zeros((3,)))
print(params['params']['kernel'].shape)  # (3, 5)
```

In PyTorch:

```python
import torch.nn as nn
linear = nn.Linear(3, 5)
print(linear.weight.shape)  # (5, 3), not (3, 5)
```
