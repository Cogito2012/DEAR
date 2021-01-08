#!/bin/bash

pwd_dir=$pwd
cd ..

source activate mmaction

# --validate
CUDA_VISIBLE_DEVICES=$1 python tools/train.py configs/recognition/i3d/finetune_ucf101_i3d_edllogloss_r50_dense_32x2x1_100e_kinetics400_rgb.py \
	--work-dir work_dirs/finetune_ucf101_i3d_edllogloss_exp_avuloss_r50_dense_32x2x1_100e_kinetics400_rgb \
	--seed 0 \
	--deterministic \
	--gpu-ids 0

cd $pwd_dir
echo "Experiments finished!"
