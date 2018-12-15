#!/bin/bash

EVAL_GPU_ID=${1}
EXP=${2} # lenet5, pr-lenet-5, resnet32, pr-resnet32, resnet110, pr-resnet110

if [ "${EXP}" == "lenet5" ]; then
  CUDA_VISIBLE_DEVICES=$EVAL_GPU_ID PYTHONPATH=/base/src:$PYTHONPATH \
    python3 src/eval.py \
    --type mnist
fi

if [ "${EXP}" == "resnet32" ]; then
  CUDA_VISIBLE_DEVICES=$EVAL_GPU_ID PYTHONPATH=/base/src:$PYTHONPATH \
    python3 src/eval.py \
    --type cifar10
fi

if [ "${EXP}" == "resnet110" ]; then
  CUDA_VISIBLE_DEVICES=$EVAL_GPU_ID PYTHONPATH=/base/src:$PYTHONPATH \
    python3 src/eval.py \
    --type cifar100
fi
