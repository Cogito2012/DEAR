#!/bin/bash

export CUDA_HOME='/usr/local/cuda'

pwd_dir=$pwd
cd ../../

source activate mmaction

DEVICE=$1
IND_DATA="data/ucf101/annotations/testlist01.txt"
OOD_DATA="data/hmdb51/annotations/testlist01.txt"
RESULT_PATH="experiments/tpn_slowonly/results/tSNE"


CUDA_VISIBLE_DEVICES=${DEVICE} python experiments/analyze_features.py \
    --config configs/recognition/tpn/inference_tpn_slowonly_bnn.py \
    --checkpoint work_dirs/tpn_slowonly/finetune_ucf101_tpn_slowonly_bnn/latest.pth \
    --known_split ${IND_DATA} \
    --unknown_split ${OOD_DATA} \
    --result_file ${RESULT_PATH}/tSNE_bnn.png

CUDA_VISIBLE_DEVICES=${DEVICE} python experiments/analyze_features.py \
    --config configs/recognition/tpn/inference_tpn_slowonly_dnn.py \
    --checkpoint work_dirs/tpn_slowonly/finetune_ucf101_tpn_slowonly_celoss/latest.pth \
    --known_split ${IND_DATA} \
    --unknown_split ${OOD_DATA} \
    --result_file ${RESULT_PATH}/tSNE_dnn.png

CUDA_VISIBLE_DEVICES=${DEVICE} python experiments/analyze_features.py \
    --config configs/recognition/tpn/inference_tpn_slowonly_enn.py \
    --checkpoint work_dirs/tpn_slowonly/finetune_ucf101_tpn_slowonly_edlloss_nokl/latest.pth \
    --known_split ${IND_DATA} \
    --unknown_split ${OOD_DATA} \
    --result_file ${RESULT_PATH}/tSNE_enn.png

CUDA_VISIBLE_DEVICES=${DEVICE} python experiments/analyze_features.py \
    --config configs/recognition/tpn/inference_tpn_slowonly_enn.py \
    --checkpoint work_dirs/tpn_slowonly/finetune_ucf101_tpn_slowonly_edlloss_nokl_avuc_debias/latest.pth \
    --known_split ${IND_DATA} \
    --unknown_split ${OOD_DATA} \
    --result_file ${RESULT_PATH}/tSNE_enn_avuc_debias.png

cd $pwd_dir
echo "Experiments finished!"