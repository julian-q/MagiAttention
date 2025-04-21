#! /bin/bash

# Copyright (c) 2025 SandAI. All Rights Reserved.
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

export GPUS_PER_NODE=${GPUS_PER_NODE:-8}
export NNODES=$WORLD_SIZE
export NODE_RANK=${RANK:-0}
export MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
export MASTER_PORT=${MASTER_PORT:-16988}

# hack to set world size
export WORLD_SIZE=$(($GPUS_PER_NODE*$NNODES))

export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
export CUDA_DEVICE_MAX_CONNECTIONS=${CUDA_DEVICE_MAX_CONNECTIONS:-1}

# set the following three env variables when you need deterministic mode, otherwise, just comment them
# export CUBLAS_WORKSPACE_CONFIG=:4096:8

DISTRIBUTED_ARGS="
    --nproc_per_node $GPUS_PER_NODE \
    --nnodes $NNODES \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT
"

TORCHRUN_CMD="torchrun $DISTRIBUTED_ARGS main.py"
$TORCHRUN_CMD

# generate a timestamp for the nsys output file
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

NSYS_CMD="
nsys profile \
    --force-overwrite true \
    -o outs/magi_attention_exp_${TIMESTAMP}.nsys-rep \
    --capture-range=cudaProfilerApi \
    $TORCHRUN_CMD
"
$NSYS_CMD
