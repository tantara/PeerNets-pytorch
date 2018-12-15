#!/usr/bin/env bash

BASE_PATH=`pwd`
SSH_PORT=8706
JUPYTER_PORT=8707
TFBOARD_PORT=8708
PRJ_NAME=peernets-pytorch-kr

nvidia-docker run -it -d \
  -p $SSH_PORT:22 \
  -p $JUPYTER_PORT:8888 \
  -p $TFBOARD_PORT:8008 \
  -v $BASE_PATH:/base \
  --shm-size 16G \
  --name $PRJ_NAME tantara/peernets-pytorch
