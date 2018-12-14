#!/bin/bash

# Experiment: LeNet-5
# epochs: 100->20
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=/base/src:$PYTHONPATH \
  python3 src/mnist/train.py \
  --batch_size 128 \
  --epochs 20 \
  --optimizer adam \
  --lr 0.001 \
  --wd 0.0001

# # Experiment: PR-LeNet-5
# CUDA_VISIBLE_DEVICES=0 PYTHONPATH=/base/src:$PYTHONPATH \
#   python3 src/mnist/train.py \
#   --batch_size 128 \
#   --epochs 32 \
#   --optimizer adam \
#   --lr 0.001 \
#   --wd 0.0001 \
#   --has_pr true

# # Experiment: ResNet-32
# CUDA_VISIBLE_DEVICES=0 PYTHONPATH=/base/src:$PYTHONPATH \
#   python3 src/cifar10/train.py \
#   --batch_size 128 \
#   --epochs 350 \
#   --optimizer sgd \
#   --lr 0.01 \
#   --wd 0.001 \
#   --momentum 0.9 \
#   --decreasing_lr 100,175,250


# # Experiment: PR-ResNet-32
# CUDA_VISIBLE_DEVICES=0 PYTHONPATH=/base/src:$PYTHONPATH \
#   python3 src/cifar10/train.py \
#   --batch_size 64 \
#   --epochs 350 \
#   --optimizer sgd \
#   --lr 0.001 \
#   --wd 0.001 \
#   --momentum 0.9 \
#   --decreasing_lr 100,175,250 \
#   --has_pr true