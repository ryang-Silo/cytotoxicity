#!/bin/bash
EXP_DATA_PATH="/cache/dataset"
EXP_TRAIN_PATH="s3://obs-app-2020070109331808081/plantai/plant/v3/train_data_flag0_v3/"
EXP_TST_DATA_PATH="s3://obs-app-2020070109331808081/plantai/plant/v3/validation_data_flag0_v3/"
EXPERIMENT_PATH="/cache/results/plantai_train_fullsup_20k_ep300_lr03_cosine_cutmix"

mkdir -p $EXPERIMENT_PATH
mkdir -p $EXP_DATA_PATH

python /home/work/user-job-dir/plantai_full_super/src/copy_data.py \
--train_path $EXP_TRAIN_PATH \
--tst_path $EXP_TST_DATA_PATH \
--data_path $EXP_DATA_PATH \

python -m torch.distributed.launch --nproc_per_node=8 /home/work/user-job-dir/plantai_full_super/src/train_20k_cutmix.py \
--data_path $EXP_DATA_PATH \
--output_path $EXPERIMENT_PATH \
--lr 0.0001 \
--lr_scheduler cosine \
--cutmix_prob 1.0 \
--beta 1.0 \
--save_path s3://obs-app-2020070109331808081/rwx950856/PlatAI/experiments_fullsup/ \