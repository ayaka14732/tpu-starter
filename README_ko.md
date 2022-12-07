# TPU Starter

<h4 align="center">
    <p>
        <b>한국어</b> |
        <a href="https://github.com/ayaka14732/tpu-starter/blob/main/README.md">English</a> | 
    <p>
</h4>

Google Cloud TPU에 대한 모든 것

* [1. 커뮤니티](#1-community)
* [2. TPU 소개](#2-introduction-to-tpu)
    * [2.1. TPU를 사용하는 이유?](#21-why-tpu)
    * [2.2. TPU를 많은 사람들이 사용하지 않는 이유?](#22-tpu-is-so-good-why-havent-i-seen-many-people-using-it)
    * [2.3. TPU, 좋은건 알겠는데 실제로 소유할 수 있을까?](#23-i-know-tpu-is-good-now-can-i-touch-a-real-tpu)
    * [2.4. TPU에 액세스 하는 방법?](#24-how-do-i-get-access-to-tpu)
    * [2.5. TPU instance를 만들어야 한다구? 그게 뭔데?](#25-what-does-it-mean-to-create-a-tpu-instance-what-do-i-actually-get)
* [3. TRC Program 소개](#3-introduction-to-the-trc-program)
    * [3.1. TRC program 신청방법?](#31-how-to-apply-for-the-trc-program)
    * [3.2. 정말 공짜야?](#32-is-it-really-free)
* [4. TPU VM Instance 만들기](#4-create-a-tpu-vm-instance)
    * [4.1. VPC firewall 수정](#41-modify-vpc-firewall)
    * [4.2. instance 만들기](#42-create-the-instance)
    * [4.3. 서버에 SSH 접속하기](#43-ssh-to-the-server)
* [5. 환경 설정](#5-environment-setup)
* [6. 개발 환경 설정](#6-development-environment-setup)
    * [6.1. Mosh and Byobu 설치](#61-set-up-mosh-and-byobu)
    * [6.2. VSCode Remote-SSH 설치](#62-set-up-vscode-remote-ssh)
    * [6.3. TPU 작동 확인하는 방법?](#63-how-can-i-verify-that-the-tpu-is-working)
* [7. JAX 기초](#7-jax-basics)
    * [7.1. JAX를 사용하는 이유?](#71-why-jax)
    * [7.2. Parallelism](#72-parallelism)
        * [7.2.1. jax.pmap 기본](#721-basics-of-jaxpmap)
        * [7.2.2. update function에 무작위성을 얻고 싶다면?](#722-what-if-i-want-to-have-randomness-in-the-update-function)
        * [7.2.3. update function에 optax optimizers를 사용하고 싶다면?](#723-what-if-i-want-to-use-optax-optimizers-in-the-update-function)
    * [7.3. 특정 모델 파라미터 고정](#73-freeze-certain-model-parameters)
    * [7.4. 허깅페이스 트랜스포머와 통합하기](#74-integration-with-hugging-face-transformers)
    * [7.5. What is a[:, None]?](#75-what-is-a-none)
* [8. TPU 사용 모범 사례](#8-tpu-best-practices)
    * [8.1. Google Colab 보다 Google Cloud Ploatform](#81-prefer-google-cloud-platform-to-google-colab)
    * [8.2. TPU node 보다 TPU VM ](#82-prefer-tpu-vm-to-tpu-node)
    * [8.3. TPU VM에서 주피터 노트북 실행](#83-run-jupyter-notebook-on-tpu-vm)
    * [8.4. TPU VM instances끼리 file 공유](#84-share-files-across-multiple-tpu-vm-instances)
    * [8.5. TPU 사용 모니터링](#85-monitor-tpu-usage)
    * [8.6. TPU VM server 시작하기](#86-start-a-server-on-tpu-vm)
* [9. JAX 사용 모범 사례](#9-jax-best-practices)
    * [9.1. Import convention](#91-import-convention)
    * [9.2. JAX random keys 관리](#92-manage-random-keys-in-jax)
    * [9.3. 모델 파라미터 시리얼라이즈](#93-serialize-model-parameters)
    * [9.4. NumPy arrays 와 JAX arrays 변환](#94-convertion-between-numpy-arrays-and-jax-arrays)
    * [9.5. PyTorch tensors 와 JAX arrays 변환](#95-convertion-between-pytorch-tensors-and-jax-arrays)
    * [9.6. 타입 어노테이션](#96-type-annotation)
    * [9.7. NumPy array , a JAX array 여부 확인하기](#97-check-if-an-array-is-either-a-numpy-array-or-a-jax-array)
    * [9.8. 중첩 딕셔너리 구조에서 모든 파라미터 shape 확인](#98-get-the-shapes-of-all-parameters-in-a-nested-dictionary)
    * [9.9. CPU에서 무작위 숫자 생성하는 올바른 방법](#99-the-correct-way-to-generate-random-numbers-on-cpu)
    * [9.10. Optax로 optimizers 사용하기](#910-use-optimizers-from-optax)
    * [9.11. Optax로 크로스엔트로피 loss 사용하기](#911-use-the-cross-entropy-loss-implementation-from-optax)
* [10. 사용 방법 모음](#10-how-can-i)
    * [10.1. TPU VM에서 주피터 노트북 사용하기](#101-run-jupyter-notebook-on-tpu-vm)
    * [10.2. 여러개의 TPU VM 인스턴스간의 파일 공유하기](#102-share-files-across-multiple-tpu-vm-instances)
    * [10.3. TPU 사용 모니터링](#103-monitor-tpu-usage)
    * [10.4. TPU VM에서 서버 시작하기](#104-start-a-server-on-tpu-vm)
    * [10.5. 다른 TPU 코어 간 분리된 프로세스 실행하기](#105-run-separate-processes-on-different-tpu-cores)
* [11. Pods 사용하기](#11-working-with-pods)
    * [11.1. NFS를 사용해 공유 디렉토리 만들기](#111-create-a-shared-directory-using-nfs)
    * [11.2. 모든 TPU Pods에서 동시에 command 실행하기](#112-run-a-command-simultaneously-on-all-tpu-pods)
* [12. 일반적인 문제들](#12-common-gotchas)
    * [12.1. TPU VM이 가끔씩 재부팅 되는 현상](#121-external-ip-of-tpu-machine-changes-occasionally)
    * [12.2. 1개 TPU device는 1개 프로세스만 사용가능](#122-one-tpu-device-can-only-be-used-by-one-process-at-a-time)
    * [12.3. 여러 프로그램과 충돌나는 TCMalloc](#123-tcmalloc-breaks-several-programs)
    * [12.4. 다른 프로세스에 의해 libtpu.so가 사용중인 현상](#125-libtpuso-already-in-used-by-another-process)
    * [12.5. fork 방식의 multiprocessing을 지원하지 않는 JAX](#126-jax-does-not-support-the-multiprocessing-fork-strategy)

<!-- Created by https://github.com/ekalinin/github-markdown-toc -->

이 프로젝트는 [Cloud Run FAQ](https://github.com/ahmetb/cloud-run-faq)에 영감을 받아서 만들어졌으며, 커뮤니티 기반으로 관리하는 Google Cloud의 기술 자료입니다.

## 1. 커뮤니티

2022 2. 23을 기준으로 Cloud TPUs 관련 공식 대화 채널은 존재하지 않으나, 텔레그램 채널 [@cloudtpu](https://t.me/cloudtpu)이나, 디스코드 채널 [TPU Podcast](https://github.com/shawwn/tpunicorn#ml-community)에 참여할 수 있습니다.  
여기엔 TRC Cloud TPU v4 유저가 그룹안에 있습니다


## 2. TPU 소개

### 2.1. TPU를 사용하는 이유?

**한줄요약**: GPU가 CPU를 대체하듯, TPU는 GPU를 대체할 수 있습니다

TPU는 머신러닝을 위해 설계된 특별한 하드웨어 입니다. Huggingface Transforemrs 퍼포먼스를 참고할 수 있습니다.  
[performance comparison](https://github.com/huggingface/transformers/blob/main/examples/flax/language-modeling/README.md#runtime-evaluation):

![](assets/5.png)

게다가 [the TRC program](https://sites.research.google/trc/about/)은 연구자들을 위해 free TPU를 제공합니다. 제가 아는 한 모델을 학습할 때 컴퓨팅 리소스를 고민해본 적이 있다면 이게 가장 최적의 해결책입니다.  
자세한 내용은 아래에 TRC program의 내용을 참고하세요.  


### 2.2. TPU를 많은 사람들이 사용하지 않는 이유?

만약 Pytorch를 사용한다면, TPU는 적합하지 않을 수 있습니다. TPU는 Pytorch에서 제대로 지원되지 않습니다. 제 실험으로 비춰봤을 때, 1개 batch가 cpu에서 14초가 걸린 반면 TPU에선 4시간이 넘게 걸렸습니다.  
트위터 유저 @mauricetpunkt 또한 [TPU에서 Pytorch 퍼포먼스가 좋지 않다고 했습니다.](https://twitter.com/mauricetpunkt/status/1506944350281945090).  
  
추가적인 문제로, 1개의 TPU v3-8은 8개 코어로(각 16GB memory) 이뤄져있으며, 이걸 전부 사용하려면 부가적인 코드를 사용해야 합니다. 그렇지 않으면 1개 코어만 사용됩니다.

### 2.3. TPU, 좋은건 알겠는데 실제로 소유할 수 있을까?
  
불행히도 TPU를 물리적으로 가질 순 없고, 클라우드 서비스를 활용해야만 가능합니다.  

### 2.4. TPU에 액세스 하는 방법?

TPU 인스턴스를 [Google Cloud Platform](https://cloud.google.com/tpu)에서 생성할 수 있습니다. 자세한 정보는 아래를 참고하세요.

[Google Colab](https://colab.research.google.com/)을 사용할 수 있지만, 별로 추천하진 않습니다. 게다가 [TRC program](https://sites.research.google/trc/about/)을 통해 무료로 TPU를 받게 된다면 코랩보단 Google Cloud Platform을 사용하게 될겁니다.

### 2.5. TPU instance를 만들어야 한다구? 그게 뭔데?

TPU v3-8 인스턴스를 [Google Cloud Platform](https://cloud.google.com/tpu)에서 만들면, Ubuntu 20.04 cloud server에 슈퍼유저 권한을 가지게 되며, 96개 코어, 335GB 메모리, 그리고 TPU 장비 1개(8개코어, 128GB vram)를 받게 됩니다

![](assets/0.png)

TPU는 우리가 GPU를 쓰는 방법과 유사합니다. 대부분 우리가 GPU를 사용할 때 GPU가 딸린 리눅스 서버를 사용하듯이 사용하면 됩니다. 단지 그 GPU가 TPU와 연결된 것 뿐입니다

## 3. TRC Program 소개

### 3.1. TRC program 신청방법?

[homepage](https://sites.research.google/trc/about/)의 내용이 있지만서도, Shawn이 TRC program에 대해서 [google/jax#2108](https://github.com/google/jax/issues/2108#issuecomment-866238579)에 상세하게 써두었습니다. TPU에 관심있다면 바로 읽는게 좋습니다.

### 3.2. 정말 공짜야?

첫 3달 동안 완전히 무료로 사용할 수 있으며 이후 한달에 HK$13.95, US$1.78정도를 사용하는데 이건 인터넷 트래픽에 대한 outbound 비용입니다.


## 4. TPU VM Instance 만들기

### 4.1. VPC firewall 수정

Mosh나 기타 프로그램이 막히지 않도록 방화벽의 제한을 완화해야 합니다.  

VPC network에 있는 [Firewall management page](https://console.cloud.google.com/networking/firewalls/list)를 여세요

새로운 방화벽 규칙 생성을 위해 버튼 클릭.

![](assets/2.png)

이름을 allow-all로 명명하고, target은 All instances in the network, source filter는 0.0.0.0/0, protocols and prots를 allow all로, 이후 생성 버튼을 클릭합니다.

대외비 데이터셋을 사용하거나, 높은 수준의 보안이 필요한 사용자는 더 엄격하게 방화벽 규칙을 적용하는 것이 좋습니다.

### 4.2. instance 만들기

[Google Cloud Platform](https://cloud.google.com/tpu)페이지에 들어간 후, 네비게이터 메뉴에서 [TPU management page](https://console.cloud.google.com/compute/tpus)에 들어갑니다.

![](assets/1.png)

우측 상단에 있는 Cloud Shell 콘솔 버튼을 누릅니다.(클라우드 쉘 실행)

Cloud Shell에서 Cloud TPU VM v3-8을 만들기 위해 아래의 명령어를 command 창에 입력합니다
(버전은 변경 가능)

```sh
gcloud alpha compute tpus tpu-vm create node-1 --project tpu-develop --zone europe-west4-a --accelerator-type v3-8 --version v2-nightly20210914
```

만약 명령어 실행이 실패하면 TPU가 모두 점유중인 것으로, 다시 실행합니다

gcloud 커맨드를 로컬 머신에 설치하면 Cloud shell을 열어 커맨드를 실행하는거보다 더 편합니다.

TPU Pod을 만들려면 아래의 명령어를 실행하세요.

```sh
gcloud alpha compute tpus tpu-vm create node-3 --project tpu-advanced-research --zone us-central2-b --accelerator-type v4-16 --version v2-alpha-tpuv4
```

### 4.3. 서버에 SSH 접속하기

TPU VM에 SSH로 접속:

```sh
gcloud alpha compute tpus tpu-vm ssh node-1 --zone europe-west4-a
```

TPU Pods중 하나에 SSH 접속:

```sh
gcloud alpha compute tpus tpu-vm ssh node-3 --zone us-central2-b --worker 0
```

## 5. 환경 설정

`setup.sh`에 아래의 스크립트를 저장 후 실행하세요 .

```sh
gcloud alpha compute tpus tpu-vm ssh node-2 --zone us-central2-b --worker all --command '

# Confirm that the script is running on the host
uname -a

# Install common packages
export DEBIAN_FRONTEND=noninteractive
sudo apt-get update -y -qq
sudo apt-get upgrade -y -qq
sudo apt-get install -y -qq golang neofetch zsh mosh byobu aria2

# Install Python 3.10
sudo apt-get install -y -qq software-properties-common
sudo add-apt-repository -y ppa:deadsnakes/ppa
sudo apt-get install -y -qq python3.10-full python3.10-dev

# Install Oh My Zsh
sh -c "$(curl -fsSL https://raw.githubusercontent.com/ohmyzsh/ohmyzsh/master/tools/install.sh)" "" --unattended
sudo chsh $USER -s /usr/bin/zsh

# Change timezone
# timedatectl list-timezones  # list timezones
sudo timedatectl set-timezone Asia/Hong_Kong  # change to your timezone

# Create venv
python3.10 -m venv $HOME/.venv310
. $HOME/.venv310/bin/activate

# Install JAX with TPU support
pip install -U pip
pip install -U wheel
pip install -U "jax[tpu]" -f https://storage.googleapis.com/jax-releases/libtpu_releases.html

'
```

이 스크립트는 `~/.venv310` 가상환경을 생성하기 때문에 가상환경을 활성화 할 때 `. ~/.venv310/bin/activate` 명렁어를 사용하거나, `~/.venv310/bin/python`를 통해 파이썬 인터프리터를 호출하면 됩니다.

이 레포를 clone한 뒤에 레포의 root 디렉토리에서 실행하세요.

```sh
pip install -r requirements.txt
```

## 6. 개발 환경 설정

### 6.1. Mosh and Byobu 설치

서버에 SSH를 통해 다이렉트로 접속하면 연결이 끊길 위험이 발생합니다.
접속이 끊기면 학습하던 프로세스는 강제로 종료되버립니다.

[Mosh](https://mosh.org/) 와 [Byobu](https://www.byobu.org/)는 이런 문제를 해결합니다.
Byobu는 연결이 끊기더라도 스크립트가 서버에서 계속 동작할 수 있도록 보장하며, Mosh는 접속이 끊기지 않는 부분을 보장합니다.

Mosh를 로컬에 설치하고, 아래 스크립트를 통해 login 하세요.

```sh
mosh tpu1 -- byobu
```

Byobu 참고 영상[Learn Byobu while listening to Mozart](https://youtu.be/NawuGmcvKus).

### 6.2. VSCode Remote-SSH 설치

VSCode를 실행 후 'Extensions' 탭에서 'Remote-SSH'를 설치하세요

<kbd>F1</kbd>을 눌러 커맨드창을 실행 후 'ssh'를 타이핑 후 'Remote-SSH: ...를 선택 후 연결하고자 하는 서버의 정보를 입력하고 엔터를 치세요.

VScode가 서버에 설치되기까지 기다리고나면 VSCode를 사용해 서버에서 개발할 수 있습니다.


![](assets/3.png)

### 6.3. TPU 작동 확인하는 방법?

아래 명령어 실행:

```sh
~/.venv310/bin/python -c 'import jax; print(jax.devices())'  # should print TpuDevice
```

TPU Pods의 경우, 아래 명령어를 로컬에서 실행하세요:

```sh
gcloud alpha compute tpus tpu-vm ssh node-2 --zone us-central2-b --worker all --command '~/.venv310/bin/python -c "import jax; jax.process_index() == 0 and print(jax.devices())"'
```

## 7. JAX 기초

### 7.1. JAX를 사용하는 이유?

JAX는 차세대 딥러닝 라이브러리로, TPU에 대한 지원이 매우 잘됩니다.  
JAX에 대한 내용으로 공식 튜토리얼을 확인해보세요.[tutorial](https://jax.readthedocs.io/en/latest/jax-101/index.html).

### 7.2. Parallelism

#### 7.2.1. `jax.pmap` 기본

4가지 키 포인트

1\. `params` 와 `opt_state` 는 디바이스간에 복제되어야 합니다.

```python
replicated_params = jax.device_put_replicated(params, jax.devices())
```

2\. `data` 와 `labels` 디바이스간에 나뉘어야 합니다.

```python
n_devices = jax.device_count()
batch_size, *data_shapes = data.shape
assert batch_size % n_devices == 0, 'The data cannot be split evenly to the devices'
data = data.reshape(n_devices, batch_size // n_devices, *data_shapes)
```

3\. `jax.pmap`과 함께 타겟 함수를 데코레이션에 사용하세요

```
@partial(jax.pmap, axis_name='num_devices')
```

4\. 디바이스간에 로스 평균을 계산하기 위해 로스 함수에 `jax.lax.pmean`을 사용하세요

```python
grads = jax.lax.pmean(grads, axis_name='num_devices')  # calculate mean across devices
```

[01-basics/test_pmap.py](01-basics/test_pmap.py) 작동 예시를 참고하세요

공식문서<https://jax.readthedocs.io/en/latest/jax-101/06-parallelism.html#example>.

#### 7.2.2. update function에 무작위성을 얻고 싶다면?

```python
key, subkey = (lambda keys: (keys[0], keys[1:]))(rand.split(key, num=9))
```

일반적인 split 방식은 사용할 수 없습니다.

```python
key, *subkey = rand.split(key, num=9)
```

일반적인 split을 사용할 경우, `subkey`가 array가 아닌 list가 되어버립니다.

#### 7.2.3. update function에 optax optimizers를 사용하고 싶다면?

`opt_state` 또한 복제되어야 합니다

### 7.3. 특정 모델 파라미터 고정


[`optax.set_to_zero`](https://optax.readthedocs.io/en/latest/api.html#optax.set_to_zero)와 [`optax.multi_transform`](https://optax.readthedocs.io/en/latest/api.html#optax.multi_transform) 사용.

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

[Freeze Parameters Example](https://colab.research.google.com/drive/1-qLk5l09bq1NxZwwbu_yDk4W7da5TnFx) 참고하세요.

### 7.4. 허깅페이스 트랜스포머와 통합하기

[Hugging Face Transformers](https://huggingface.co/docs/transformers/index)

### 7.5. What is `a[:, None]`?

[`np.newaxis`](https://numpy.org/doc/stable/reference/constants.html#numpy.newaxis)

## 8. TPU 사용 모범 사례

### 8.1. Google Colab 보다 Google Cloud Ploatform
  
[Google Colab](https://colab.research.google.com/)은 TPU v2-8 장비만 제공하는 반면, [Google Cloud Platform](https://cloud.google.com/tpu)은 TPU v3-8 장비도 제공합니다.

게다가, Colab은 Jupyter Notebook 인터페이스로만 TPU에 접근할 수 있으며, [log in into the Colab server via SSH](https://ayaka.shn.hk/colab/)링크의 방법을 사용하더라도, docker image이기 때문에 root 권한을 가질 수 없습니다.
Google Cloud platform에선 root 권한을 가질 수 있습니다.

굳이 Google Colab에서 TPU를 사용하고 싶다면, [스크립트](01-basics/setup_colab_tpu.py)를 사용해서 TPU를 세팅하세요


```python
import jax
from jax.tools.colab_tpu import setup_tpu

setup_tpu()

devices = jax.devices()
print(devices)  # should print TpuDevice
```

### 8.2. TPU node 보다 TPU VM

TPU 인스턴스를 생성할 때 TPU VM과 TPU node 중 선택해야 하는데, TPU VM을 추천합니다.  
TPU VM은 TPU host에 다이렉트로 연결되며, TPU 장비를 세팅하기 쉽게 만들어 줍니다.

### 8.3. TPU VM에서 주피터 노트북 실행

Remote-SSH를 세팅 후 VSCode에서 Jupyter Notebook 파일로 작업할 수 있습니다.
또는 PC에 포트포워딩을 통해 TPU VM에서 Jupyter Notebook 서버를 실행할 수도 있습니다.
그러나 VSCode가 더 파워풀하고, 더 나은 통합기능을 제공하고 세팅하기 유리하기 때문에 VSCode를 추천합니다.

### 8.4. TPU VM instances끼리 file 공유

같은 Zone에 있는 TPU VM 인스턴스들은 internal IP를 통해 연결되어 있기 때문에 
[NFS를 활용한 공유 파일 시스템 만들기](https://tecadmin.net/how-to-install-and-configure-an-nfs-server-on-ubuntu-20-04/)가 가능합니다

### 8.5. TPU 사용 모니터링

### 8.6. TPU VM server 시작하기

예시 : 텐서보드

모든 TPU VM은 public IP를 가지고 있지만, 안전하지 않으므로 인터넷에 IP를 노출해선 안됩니다.

SSH를 통한 포트 포워딩

```
ssh -C -N -L 127.0.0.1:6006:127.0.0.1:6006 tpu1
```

## 9. JAX 사용 모범 사례

### 9.1. Import convention

import 방법에 대해 다른 종류가 있습니다. 
import jax.numpy as np, 와 import numpy as onp, 다른 방법으로는  
import jax.numpy as jnp, 와import numpy as np 가 있습니다.

19.1.16 Colin Raffel의 경우 [a blog article](https://colinraffel.com/blog/you-don-t-know-jax.html)에서 numpy as onp 방식을 사용했습니다.

20.11.5 Niru Maheswaranathan의 경우 [a tweet](https://twitter.com/niru_m/status/1324078070546882560)에서 numpy as np, jax as jnp 방식을 사용했습니다

TODO: Conclusion?

### 9.2. JAX random keys 관리

일반적인 방법:

```python
key, *subkey = rand.split(key, num=4)
print(subkey[0])
print(subkey[1])
print(subkey[2])
```

### 9.3. 모델 파라미터 시리얼라이즈

일반적으로 모델 파라미터들은 중첩된 딕셔너리 구조로 표현됩니다.

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

[`flax.serialization.msgpack_serialize`](https://flax.readthedocs.io/en/latest/flax.serialization.html#flax.serialization.msgpack_serialize)를 사용하면 모델 파라미터를 시리얼라이즈해서 바이트로 바꿀 수 있으며, [`flax.serialization.msgpack_restore`](https://flax.readthedocs.io/en/latest/flax.serialization.html#flax.serialization.msgpack_serialize)를 사용하면 다시 중첩된 딕셔너리로 변경 가능합니다.

### 9.4. NumPy arrays 와 JAX arrays 변환


[`np.asarray`](https://jax.readthedocs.io/en/latest/_autosummary/jax.numpy.asarray.html) 와 [`onp.asarray`](https://numpy.org/doc/stable/reference/generated/numpy.asarray.html) 사용.

```python
import jax.numpy as np
import numpy as onp

a = np.array([1, 2, 3])  # JAX array
b = onp.asarray(a)  # converted to NumPy array

c = onp.array([1, 2, 3])  # NumPy array
d = np.asarray(c)  # converted to JAX array
```

### 9.5. PyTorch tensors 와 JAX arrays 변환

PyTorch tensor를 JAX array로 변환:

```python
import jax.numpy as np
import torch

a = torch.rand(2, 2)  # PyTorch tensor
b = np.asarray(a.numpy())  # JAX array
```

a JAX array를 PyTorch tensor로 변환:

```python
import jax.numpy as np
import numpy as onp
import torch

a = np.zeros((2, 2))  # JAX array
b = torch.from_numpy(onp.asarray(a))  # PyTorch tensor
```

아래 warning 메세지가 뜹니다:

```
UserWarning: The given NumPy array is not writable, and PyTorch does not support non-writable tensors. This means writing to this tensor will result in undefined behavior. You may want to copy the array to protect its data or make it writable before converting it to a tensor. This type of warning will be suppressed for the rest of this program. (Triggered internally at  ../torch/csrc/utils/tensor_numpy.cpp:178.)
```

쓰기 가능한 텐서가 필요하다면 `onp.asarray`가 아닌 `onp.array`를 사용해 original array를 카피하면 됩니다.

### 9.6. 타입 어노테이션

[google/jaxtyping](https://github.com/google/jaxtyping)

### 9.7. NumPy array , a JAX array 여부 확인하기

```python
isinstance(a, (np.ndarray, onp.ndarray))
```

### 9.8. 중첩 딕셔너리 구조에서 모든 파라미터 shape 확인

```python
jax.tree_map(lambda x: x.shape, params)
```

### 9.9. CPU에서 무작위 숫자 생성하는 올바른 방법


[jax.default_device()](https://jax.readthedocs.io/en/latest/_autosummary/jax.default_device.html)를 컨텍스트 매니저와 사용:

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

### 9.10. Optax로 optimizers 사용하기

### 9.11. Optax로 크로스엔트로피 loss 사용하기

`optax.softmax_cross_entropy_with_integer_labels`

## 10. 사용 방법 모음

### 10.1. TPU VM에서 주피터 노트북 사용하기

Remote-SSH를 세팅 후에, VSCode 안에서 주피터 노트북 파일로 작업할 수 있습니다.

대체 방법으로, 사용하는 PC에서 포트 포워딩으로 TPU VM에서 주피터 노트북 서버를 실행시킬 수도 있습니다. 하지만,
VSCode가 더 강력하고 다른 도구(확장프로그램 등)들과 더 잘 통합되며 설정하기 쉽기 때문에 VSCode를 쓰는 편이 좋습니다.  
[번역자 블로그 참고](https://okdone.tistory.com/144)

### 10.2. 여러개의 TPU VM 인스턴스간의 파일 공유하기

같은 존에 있는 TPU VM 인스턴스는 내부 IP를 통해 연결되어 있습니다. 이를 사용하면 [NFS를 사용한 공유 파일 시스템](https://tecadmin.net/how-to-install-and-configure-an-nfs-server-on-ubuntu-20-04/)을 만들 수 있습니다.

### 10.3. TPU 사용 모니터링

[jax-smi](https://github.com/ayaka14732/jax-smi)

### 10.4. TPU VM에서 서버 시작하기

예시: Tensorboard

비록 TPU VM에 공인 IP가 있어서 대부분 인터넷에 IP를 노출해야 하는데, 이건 보안에 좋지 않습니다

SSH를 통한 포트 포워딩

```
ssh -C -N -L 127.0.0.1:6006:127.0.0.1:6006 tpu1
```

### 10.5. 다른 TPU 코어 간 분리된 프로세스 실행하기

https://gist.github.com/skye/f82ba45d2445bb19d53545538754f9a3

## 11. Pods 사용하기

### 11.1. NFS를 사용해 공유 디렉토리 만들기

참고: §8.4.

### 11.2. 모든 TPU Pods에서 동시에 command 실행하기

```sh
#!/bin/bash

while read p; do
  ssh "$p" "cd $PWD; rm -rf /tmp/libtpu_lockfile /tmp/tpu_logs; . ~/.venv310/bin/activate; $@" &
done < external-ips.txt
rm -rf /tmp/libtpu_lockfile /tmp/tpu_logs; . ~/.venv310/bin/activate; "$@"
wait
```

See <https://github.com/ayaka14732/bart-base-jax/blob/f3ccef7b32e2aa17cde010a654eff1bebef933a4/startpod>.

## 12. 일반적인 문제들

### 12.1. TPU VM이 가끔 재부팅 되는 현상

2022년 10월 24일부터 유지보수 작업이 있는 경우 TPU VM이 가끔 재부팅됩니다.

아래와 같은 현상이 발생할 예정입니다:

1. 실행중인 모든 프로세스 종료
2. 외부 IP 주소 변경

모델 매개변수, 옵티마이저 상태 및 기타 유용한 데이터를 때때로 저장할 수 있으므로 종료 후 모델 교육을 쉽게 재개할 수 있습니다.(자주 저장할 것)

SSH로 직접 연결하는 대신 `gcloud` 명령을 사용해야 합니다. SSH를 사용해야 하는 경우(예: VSCode를 사용하려는 경우 SSH가 유일한 선택) 대상 IP 주소를 수동으로 변경해야 합니다.

### 12.2. 1개 TPU device는 1개 프로세스만 사용가능

GPU와 다르게 두개의 프로세스가 TPU에 동시에 접근하면 에러가 발생합니다.

```
I0000 00:00:1648534265.148743  625905 tpu_initializer_helper.cc:94] libtpu.so already in use by another process. Run "$ sudo lsof -w /dev/accel0" to figure out which process is using the TPU. Not attempting to load libtpu.so in this process.
```

TPU 디바이스가 8개의 코어이지만, 1개의 프로세스만 첫번째 코어에 접근하며 다른 프로세스는 여분의 코어를 활용할 수 없습니다.

### 12.3. 여러 프로그램과 충돌나는 TCMalloc

[TCMalloc](https://github.com/google/tcmalloc)은 구글의 커스텀 메모리 배정 라이브러리 입니다. TPU VM에서 `LD_PRELOAD`은 TCMalloc을 디폴트로 사용하게 되어 있습니다. :

```sh
$ echo LD_PRELOAD
/usr/lib/x86_64-linux-gnu/libtcmalloc.so.4
```

그러나 TCMalloc은 gsutil과 같은 여러 프로그램과 충돌합니다:

```sh
$ gsutil --help
/snap/google-cloud-sdk/232/platform/bundledpythonunix/bin/python3: /snap/google-cloud-sdk/232/platform/bundledpythonunix/bin/../../../lib/x86_64-linux-gnu/libm.so.6: version `GLIBC_2.29' not found (required by /usr/lib/x86_64-linux-gnu/libtcmalloc.so.4)
```

[homepage of TCMalloc](http://goog-perftools.sourceforge.net/doc/tcmalloc.html)에서도 `LD_PRELOAD`의 사용이 까다로우며, 이 사용모드에서 권장되지 않습니다.

TCMalloc과 연관된 문제에 직면할 경우, 아래 명령어를 활용해 TCMalloc을 disable 하세요:

```sh
unset LD_PRELOAD
```

### 12.4. 다른 프로세스에 의해 `libtpu.so`가 사용중인 현상

```sh
if ! pgrep -a -u $USER python ; then
    killall -q -w -s SIGKILL ~/.venv310/bin/python
fi
rm -rf /tmp/libtpu_lockfile /tmp/tpu_logs
```

참고 <https://github.com/google/jax/issues/9220#issuecomment-1015940320>.

### 12.5. `fork` 방식의 multiprocessing을 지원하지 않는 JAX

 `spawn` 이나 `forkserver` 방법을 사용하세요.

참고 <https://github.com/google/jax/issues/1805#issuecomment-561244991>.
