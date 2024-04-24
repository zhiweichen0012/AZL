#!/bin/bash
set -x
set -e

GPU=$1
NET=$2
MODLE_DIR=$3

# bash scripts/test.sh 0 vgg logs/vgg/model_epoch69.pth.tar

/path/to/python3 test_load.py --gpu ${GPU} --arch ${NET} --model_p ${MODLE_DIR}