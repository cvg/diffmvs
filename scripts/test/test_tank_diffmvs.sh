#!/usr/bin/env bash


TANK_TESTING="/tankandtemples/"
CKPT_FILE="./checkpoints/diffmvs_blend.ckpt"
OUT_DIR='./outputs_tank'
if [ ! -d $OUT_DIR ]; then
    mkdir -p $OUT_DIR
fi

python test.py --dataset=tank --batch_size=1 --num_view=10 --method=diffmvs --save_depth \
    --testpath=$TANK_TESTING  --numdepth_initial=96 --numdepth=384 \
    --testlist=lists/tank/intermediate.txt --loadckpt=$CKPT_FILE --outdir=$OUT_DIR \
    --scale 0.0 0.125 0.0 --sampling_timesteps 0 1 1 --ddim_eta 0 1 0 \
    --stage_iters 1 4 0 --cost_dim_stage 4 4 0 --CostNum 0 6 0 \
    --hidden_dim 0 32 0 --context_dim 32 32 0 --unet_dim 0 16 8 \
    --min_radius 0.25 --max_radius 4

python test.py --dataset=tank --batch_size=1 --num_view=10 --method=diffmvs --save_depth \
    --testpath=$TANK_TESTING  --numdepth_initial=96 --numdepth=384 \
    --testlist=lists/tank/advanced.txt --loadckpt=$CKPT_FILE --outdir=$OUT_DIR \
    --scale 0.0 0.125 0.0 --sampling_timesteps 0 1 1 --ddim_eta 0 1 0 \
    --stage_iters 1 4 0 --cost_dim_stage 4 4 0 --CostNum 0 6 0 \
    --hidden_dim 0 32 0 --context_dim 32 32 0 --unet_dim 0 16 8 \
    --min_radius 0.25 --max_radius 4
