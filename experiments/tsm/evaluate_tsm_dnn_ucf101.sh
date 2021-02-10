#!/bin/bash

pwd_dir=$pwd
cd ../../

source activate mmaction

CUDA_VISIBLE_DEVICES=$1 python tools/test.py configs/recognition/tsm/tsm_dnn_r50_dense_1x1x8_100e_kinetics400_rgb.py \
	work_dirs/tsm/finetune_ucf101_tsm_dnn/latest.pth \
	--out work_dirs/tsm/test_ucf101_tsm_dnn.pkl \
	--eval top_k_accuracy mean_class_accuracy

cd $pwd_dir
echo "Experiments finished!"
