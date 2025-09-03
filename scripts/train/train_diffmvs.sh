#!/usr/bin/env bash
MVS_TRAINING="/cluster/project/cvg/fawang/mvs_training/"
LOG_DIR="./checkpoints/diffmvs"
if [ ! -d $LOG_DIR ]; then
    mkdir -p $LOG_DIR
fi
filename="diffmvs_dtu"
dirAndName="$LOG_DIR/$filename.log"
if [ ! -d $dirAndName ]; then
    touch $dirAndName
fi

##DTU
python train.py --mode='train' --dataset=dtu --batch_size=4 --epochs=12 \
    --lr=0.001 --lr_sche onecycle \
    --logdir $LOG_DIR --trainpath=$MVS_TRAINING \
    --trainviews=5 --testviews=5 \
    --numdepth=384 --numdepth_initial=48 \
    --stage_iters 1 1 0 --cost_dim_stage 4 4 0 --CostNum 0 6 0 \
    --min_radius 0.25 --max_radius 4 \
    --scale 0 0.5 0.0 --conf_weight 0.05 \
    --hidden_dim 0 32 0 --context_dim 32 32 0 --unet_dim 0 16 8 \
    --trainlist lists/dtu/train.txt --testlist lists/dtu/val.txt | tee -i $dirAndName

##BlendedMVS
MVS_TRAINING="/cluster/project/cvg/fawang/BlendedMVS/"
LOG_DIR="./checkpoints/diffmvs/blend"
LOAD_CKPT="./checkpoints/diffmvs/model_000011.ckpt"
if [ ! -d $LOG_DIR ]; then
    mkdir -p $LOG_DIR
fi
filename="diffmvs_blend"
dirAndName="$LOG_DIR/$filename.log"
if [ ! -d $dirAndName ]; then
    touch $dirAndName
fi
python -u train.py --mode='train' --dataset=blend --batch_size=4 --epochs=12 \
    --train_epochs=6 --loadckpt=$LOAD_CKPT \
    --lr=0.001 --lr_sche onecycle \
    --logdir=$LOG_DIR --trainpath=$MVS_TRAINING \
    --trainviews=9 --testviews=9 \
    --numdepth=384 --numdepth_initial=48 \
    --stage_iters 1 1 0 --cost_dim_stage 4 4 0 --CostNum 0 6 0 \
    --min_radius 0.25 --max_radius 4 \
    --scale 0 0.5 0.0 --conf_weight 0.05 \
    --hidden_dim 0 32 0 --context_dim 32 32 0 --unet_dim 0 16 8 \
    --trainlist lists/blend/train.txt --testlist lists/blend/val.txt | tee -i $dirAndName

python -u train.py --mode='train' --dataset=blend --batch_size=4 --epochs=12 \
    --lr=0.001 --lr_sche onecycle --resume \
    --logdir $LOG_DIR --trainpath=$MVS_TRAINING \
    --trainviews=9 --testviews=9 \
    --numdepth=384 --numdepth_initial=48 \
    --stage_iters 1 1 0 --cost_dim_stage 4 4 0 --CostNum 0 6 0 \
    --min_radius 0.25 --max_radius 4 \
    --scale 0 0.5 0.0 --conf_weight 0.05 \
    --hidden_dim 0 32 0 --context_dim 32 32 0 --unet_dim 0 16 8 \
    --trainlist lists/blend/train.txt --testlist lists/blend/val.txt | tee -i $dirAndName
