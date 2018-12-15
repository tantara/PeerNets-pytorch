#!/bin/bash

ATTACK_GPU_ID=${1}
EXP=${2} # lenet5, pr-lenet-5, resnet32, pr-resnet32, resnet110, pr-resnet110
#N_SAMPLE=1000
#RHO=0.2
N_SAMPLE=${3}
RHO=${4}

if [ "${EXP}" == "lenet5" ]; then
  CUDA_VISIBLE_DEVICES=$ATTACK_GPU_ID PYTHONPATH=/base/src:$PYTHONPATH \
    python3 src/attack.py \
    --type mnist \
    --input_size 28 \
    --clip_min 0 \
    --clip_max 1 \
    --n_sample $N_SAMPLE \
    --rho $RHO
fi

if [ "${EXP}" == "resnet32" ]; then
  CUDA_VISIBLE_DEVICES=$ATTACK_GPU_ID PYTHONPATH=/base/src:$PYTHONPATH \
    python3 src/attack.py \
    --type cifar10 \
    --input_size 32 \
    --clip_min -1 \
    --clip_max 1 \
    --n_sample $N_SAMPLE \
    --rho $RHO
fi

if [ "${EXP}" == "resnet110" ]; then
  CUDA_VISIBLE_DEVICES=$ATTACK_GPU_ID PYTHONPATH=/base/src:$PYTHONPATH \
    python3 src/attack.py \
    --type cifar100 \
    --input_size 32 \
    --clip_min -1 \
    --clip_max 1 \
    --n_classes 100 \
    --n_sample $N_SAMPLE \
    --rho $RHO
fi
