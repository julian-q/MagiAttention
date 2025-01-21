# Zeus(A super fast distributed attention solver)


## Installation

### Step1: activate the NGC Pytorch Docker container

* release note: [here](https://docs.nvidia.com/deeplearning/frameworks/pytorch-release-notes/rel-24-04.html#rel-24-04)
* docker image version: nvcr.io/nvidia/pytorch:24.04-py3
* docker run command:

    ```bash
    docker run --name zeus_dev_{your_name} -v {host_mnt_root}:{container_mnt_root} -it -d --privileged --gpus all --network host --ipc host --ulimit memlock=-1 --ulimit stack=67108864 nvcr.io/nvidia/pytorch:24.04-py3 /bin/bash
    ```

* docker exec command:

    ```bash
    docker exec -it zeus_dev_{your_name} /bin/bash
    ```


### Step2: install ffa with other submodules

* command:

    ```bash
    # clone submodules including ffa
    git submodule update --init --recursive

    # get into ffa folder
    cd third_party/flexible-flash-attention/hopper

    # install ffa
    python setup.py install
    ```

### Step3: install other required packages

* command:

    ```bash
    # install required packages
    pip install -r requirements.txt
    ```
