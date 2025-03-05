# Zeus(A super fast distributed attention solver)

<div align="center">
  <img src="./assets/zeus_logo.png" alt="Logo" width="1000">
</div>


## Installation

### Step1: activate the NGC Pytorch Docker container

* release note: [here](https://docs.nvidia.com/deeplearning/frameworks/pytorch-release-notes/rel-24-04.html#rel-24-04)
* docker image version: nvcr.io/nvidia/pytorch:24.04-py3
* docker run command:

    ```bash
    docker run --name {contaier_name} -v {host_mnt_root}:{container_mnt_root} -it -d --privileged --gpus all --network host --ipc host --ulimit memlock=-1 --ulimit stack=67108864 nvcr.io/nvidia/pytorch:24.04-py3 /bin/bash
    ```

* docker exec command:

    ```bash
    docker exec -it {contaier_name} /bin/bash
    ```


### Step2: install zeus with other submodules

* command:

    ```bash
    # 1-1. install zeus for developer
    pip install -e .

    # 1-2. or, install zeus for user
    make refresh

    # 2. clone submodules including ffa
    git submodule update --init --recursive

    # 3-1. get into ffa folder
    cd third_party/flexible-flash-attention/hopper

    # 3-2. install ffa
    python setup.py install; cd -
    ```

### Step3: install other required packages

* command:

    ```bash
    # install required packages
    pip install -r requirements.txt
    ```

### Step4: setup pre-commit (for developer)

* pre-commit:
    ```bash
    # after `pip install pre-commit` (done in step3)
    # you need to set up the hooks for the first time
    # which might take a while but only need to be done once
    pre-commit install

    # then each time before you run `git commit`
    # please run the pre-commit to polish your code
    pre-commit run -a

    # if anything has been automatically fixed
    # or required to be manually fixed by pre-commit
    # please rerun `git add` to track the changes
    # when everything is fixed and ready to be committed

    # for more detailed information about pre-commit
    # you can check: https://pre-commit.com/
    ```
