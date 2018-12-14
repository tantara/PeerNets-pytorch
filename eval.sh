#!/bin/bash

CUDA_VISIBLE_DEVICES=7 PYTHONPATH=/base/src:$PYTHONPATH \
  python3 src/main.py \
  --type mnist
