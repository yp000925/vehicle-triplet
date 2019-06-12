#!/usr/bin/env bash


IMAGE_ROOT=/Users/zhangyunping/PycharmProjects/vehicle-triplet-reid/VRIC/train_images ; shift
# INIT_CHECKPT=./pretrained_models/resnet_v1_101.ckpt ; shift
INIT_CHECKPT=./pre_trained_model/mobilenet_v1_1.0_224/mobilenet_v1_1.0_224.ckpt ; shift
EXP_ROOT=./experiments/baseline ; shift


python train.py \
    --train_set VRIC/vric_train.txt \
    --model_name mobilenet_v1_1_224 \
    --image_root $IMAGE_ROOT \
    --experiment_root $EXP_ROOT \
    --initial_checkpoint $INIT_CHECKPT \
    --flip_augment \
    --crop_augment \
    --detailed_logs \
    --embedding_dim 128 \
    --batch_p 18 \
    --batch_k 4 \
    --pre_crop_height 300 --pre_crop_width 300 \
    --net_input_height 224 --net_input_width 224 \
    --margin soft \
    --metric euclidean \
    --loss batch_hard \
    --learning_rate 1e-3 \
    --train_iterations 50000 \
    --decay_start_iteration 10000 \
    --lr_decay_factor 0.96 \
    --lr_decay_steps 4000 \
    --weight_decay_factor 0.0002 \
    --detailed_logs \
    #--resume \
    "$@"
