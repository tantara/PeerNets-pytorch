#!/bin/bash

EXP=${1} # lenet5, pr-lenet-5, resnet32, pr-resnet32, resnet110, pr-resnet110
TRAIN_GPU_ID=${2}

# Experiment: LeNet-5
if [ "${EXP}" == "lenet5" ]; then
  CUDA_VISIBLE_DEVICES=$TRAIN_GPU_ID PYTHONPATH=/base/src:$PYTHONPATH \
    python3 src/mnist/train.py \
    --batch_size 128 \
    --epochs 100 \
    --optimizer adam \
    --lr 0.001 \
    --wd 0.0001
fi

# Experiment: PR-LeNet-5
if [ "${EXP}" == "lenet5" ]; then
  CUDA_VISIBLE_DEVICES=$TRAIN_GPU_ID PYTHONPATH=/base/src:$PYTHONPATH \
    python3 src/mnist/train.py \
    --batch_size 128 \
    --epochs 32 \
    --optimizer adam \
    --lr 0.001 \
    --wd 0.0001 \
    --has_pr true

# Experiment: ResNet-32
if [ "${EXP}" == "resnet32" ]; then
  CUDA_VISIBLE_DEVICES=$TRAIN_GPU_ID PYTHONPATH=/base/src:$PYTHONPATH \
    python3 src/cifar/train.py \
    --type cifar10 \
    --batch_size 128 \
    --epochs 350 \
    --optimizer sgd \
    --lr 0.1 \
    --wd 0.001 \
    --momentum 0.9 \
    --decreasing_lr 100,175,250
fi

# Experiment: PR-ResNet-32
if [ "${EXP}" == "pr-resnet32" ]; then
  CUDA_VISIBLE_DEVICES=$TRAIN_GPU_ID PYTHONPATH=/base/src:$PYTHONPATH \
    python3 src/cifar/train.py \
    --type cifar10 \
    --batch_size 64 \
    --epochs 350 \
    --optimizer sgd \
    --lr 0.01 \
    --wd 0.001 \
    --momentum 0.9 \
    --decreasing_lr 100,175,250 \
    --has_pr true
fi

# Experiment: ResNet-110
if [ "${EXP}" == "resnet110" ]; then
  CUDA_VISIBLE_DEVICES=$TRAIN_GPU_ID PYTHONPATH=/base/src:$PYTHONPATH \
    python3 src/cifar/train.py \
    --type cifar100 \
    --batch_size 128 \
    --epochs 350 \
    --optimizer sgd \
    --lr 0.1 \
    --wd 0.002 \
    --momentum 0.9 \
    --decreasing_lr 100,175,250
fi

# Experiment: PR-ResNet-110
if [ "${EXP}" == "pr-resnet110" ]; then
  CUDA_VISIBLE_DEVICES=$TRAIN_GPU_ID PYTHONPATH=/base/src:$PYTHONPATH \
    python3 src/cifar/train.py \
    --type cifar100 \
    --batch_size 64 \
    --epochs 350 \
    --optimizer sgd \
    --lr 0.01 \
    --wd 0.002 \
    --momentum 0.9 \
    --decreasing_lr 100,175,250 \
    --has_pr true
fi
