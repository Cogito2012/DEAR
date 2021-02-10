#!/bin/bash

pwd_dir=$pwd
cd ../../

source activate mmaction

# --validate
CUDA_VISIBLE_DEVICES=$1 python tools/train.py configs/recognition/tsm/tsm_edlnokl_avuc_debias_r50_dense_1x1x8_100e_kinetics400_rgb.py \
	--work-dir work_dirs/tsm/finetune_ucf101_tsm_edlnokl_avuc_debias \
	--seed 0 \
	--deterministic \
	--gpu-ids 0 \
	--validate

cd $pwd_dir
echo "Experiments finished!"
