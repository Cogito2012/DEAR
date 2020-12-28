#!/bin/bash

pwd_dir=$pwd
cd ..

source activate mmaction

CUDA_VISIBLE_DEVICES=$1 python tools/test.py configs/recognition/i3d/finetune_ucf101_i3d_edlloss_r50_dense_32x2x1_100e_kinetics400_rgb.py \
	work_dirs/finetune_ucf101_i3d_edlloss_exp_r50_dense_32x2x1_100e_kinetics400_rgb/latest.pth \
	--out work_dirs/test_ucf101_i3d_edlloss_exp_r50_dense_32x2x1_100e_kinetics400_rgb.pkl \
	--eval top_k_accuracy mean_class_accuracy

cd $pwd_dir
echo "Experiments finished!"
