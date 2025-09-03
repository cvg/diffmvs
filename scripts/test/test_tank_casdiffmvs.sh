#!/usr/bin/env bash


TANK_TESTING="/tankandtemples/"
CKPT_FILE="./checkpoints/casdiffmvs_blend.ckpt"
OUT_DIR='./outputs_tank_cas'
if [ ! -d $OUT_DIR ]; then
    mkdir -p $OUT_DIR
fi

python test.py --dataset=tank --batch_size=1 --num_view=10 --method=casdiffmvs --save_depth \
    --testpath=$TANK_TESTING  --numdepth_initial=48 --numdepth=384 \
    --testlist=lists/tank/intermediate.txt --loadckpt=$CKPT_FILE --outdir=$OUT_DIR \
    --scale 0.0 0.125 0.025 --sampling_timesteps 0 1 1 --ddim_eta 0 1 1 \
    --stage_iters 1 3 3 --cost_dim_stage 4 4 4 --CostNum 0 4 4 \
    --hidden_dim 0 32 20 --context_dim 32 32 16 --unet_dim 0 16 8 \
    --min_radius 0.125 --max_radius 8

python test.py --dataset=tank --batch_size=1 --num_view=10 --method=casdiffmvs --save_depth \
    --testpath=$TANK_TESTING  --numdepth_initial=48 --numdepth=384 \
    --testlist=lists/tank/advanced.txt --loadckpt=$CKPT_FILE --outdir=$OUT_DIR \
    --scale 0.0 0.125 0.025 --sampling_timesteps 0 1 1 --ddim_eta 0 1 1 \
    --stage_iters 1 3 3 --cost_dim_stage 4 4 4 --CostNum 0 4 4 \
    --hidden_dim 0 32 20 --context_dim 32 32 16 --unet_dim 0 16 8 \
    --min_radius 0.125 --max_radius 8
