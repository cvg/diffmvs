#!/usr/bin/env bash
MVS_TRAINING="/cluster/project/cvg/fawang/mvs_training/"
LOG_DIR="./checkpoints/casdiffmvs"
if [ ! -d $LOG_DIR ]; then
    mkdir -p $LOG_DIR
fi
filename="casdiffmvs_dtu"
dirAndName="$LOG_DIR/$filename.log"
if [ ! -d $dirAndName ]; then
    touch $dirAndName
fi

##DTU
python train.py --mode='train' --dataset=dtu --batch_size=4 --epochs=16 \
    --lr=0.001 --lr_sche onecycle \
    --logdir $LOG_DIR --trainpath=$MVS_TRAINING \
    --trainviews=5 --testviews=5 \
    --numdepth=384 --numdepth_initial=48 \
    --stage_iters 1 3 3 --cost_dim_stage 4 4 4 --CostNum 0 4 4 \
    --min_radius 0.125 --max_radius 8 \
    --scale 0 0.5 0.1 --conf_weight 0.05 \
    --hidden_dim 0 32 20 --context_dim 32 32 16 --unet_dim 0 16 8 \
    --trainlist lists/dtu/train.txt --testlist lists/dtu/val.txt | tee -i $dirAndName

##BlendedMVS
MVS_TRAINING="/cluster/project/cvg/fawang/BlendedMVS/"
LOG_DIR="./checkpoints/casdiffmvs/blend"
LOAD_CKPT="./checkpoints/casdiffmvs/model_000015.ckpt"
if [ ! -d $LOG_DIR ]; then
    mkdir -p $LOG_DIR
fi
filename="casdiffmvs_blend"
dirAndName="$LOG_DIR/$filename.log"
if [ ! -d $dirAndName ]; then
    touch $dirAndName
fi
python -u train.py --mode='train' --dataset=blend --batch_size=4 --epochs=16 \
    --train_epochs=8 --loadckpt=$LOAD_CKPT \
    --lr=0.001 --lr_sche onecycle \
    --logdir=$LOG_DIR --trainpath=$MVS_TRAINING \
    --trainviews=9 --testviews=9 \
    --numdepth=384 --numdepth_initial=48 \
    --stage_iters 1 3 3 --cost_dim_stage 4 4 4 --CostNum 0 4 4 \
    --min_radius 0.125 --max_radius 8 \
    --scale 0 0.25 0.05 --conf_weight 0.05 \
    --hidden_dim 0 32 20 --context_dim 32 32 16 --unet_dim 0 16 8 \
    --trainlist lists/blend/train.txt --testlist lists/blend/val.txt | tee -i $dirAndName

python -u train.py --mode='train' --dataset=blend --batch_size=4 --epochs=16 \
    --lr=0.001 --lr_sche onecycle --resume \
    --logdir $LOG_DIR --trainpath=$MVS_TRAINING \
    --trainviews=9 --testviews=9 \
    --numdepth=384 --numdepth_initial=48 \
    --stage_iters 1 3 3 --cost_dim_stage 4 4 4 --CostNum 0 4 4 \
    --min_radius 0.125 --max_radius 8 \
    --scale 0 0.125 0.025 --conf_weight 0.05 \
    --hidden_dim 0 32 20 --context_dim 32 32 16 --unet_dim 0 16 8 \
    --trainlist lists/blend/train.txt --testlist lists/blend/val.txt | tee -i $dirAndName