#!/bin/bash
set -x
set -e

GPU=$1
NET=$2
EPOCH=$3
L1=$4
L2=$5
L3=$6
L4=$7
L5=$8
EXP_DIR="logs/${NET}/${9}_${L1}_${L2}_${L3}_${L4}_${L5}"
LOG="${EXP_DIR}/train_${9}_${L1}_${L2}_${L3}_${L4}_${L5}_`date +'%Y-%m-%d_%H-%M-%S'`.log"

if [ ! -d "${EXP_DIR}" ]
then
  mkdir -p "${EXP_DIR}"
else
  read -p "The path exists, DELETE? (y/n)" DE
  case ${DE} in
    Y | y)
      rm -rf "${EXP_DIR}"
      mkdir -p "${EXP_DIR}";;
    *)
      exit
  esac
fi

exec &> >(tee -a "$LOG")
echo Logging output to "$LOG"

/path/to/python3 train.py --gpu ${GPU} --arch ${NET} --epochs ${EPOCH} --save_path ${EXP_DIR} --lambda_a ${L1} --lambda_b ${L2} --lambda_c ${L3} --lambda_d ${L4} --lambda_e ${L5}
