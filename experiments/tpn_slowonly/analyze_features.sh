#!/bin/bash

export CUDA_HOME='/usr/local/cuda'

pwd_dir=$pwd
cd ../../

source activate mmaction

DEVICE=$1

CUDA_VISIBLE_DEVICES=${DEVICE} python experiments/analyze_features.py \
    --config configs/recognition/tpn/inference_tpn_slowonly_enn.py \
    --checkpoint work_dirs/tpn_slowonly/finetune_ucf101_tpn_slowonly_edlloss_nokl_avuc_debias/latest.pth \
    --known_split data/ucf101/annotations/testlist01.txt \
    --unknown_split data/hmdb51/annotations/testlist01.txt \
    --result_file experiments/tpn_slowonly/results/tSNE_edlloss_nokl_avuc_debias.png



cd $pwd_dir
echo "Experiments finished!"