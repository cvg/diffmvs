#!/usr/bin/env bash

DTU_TESTING="/dtu/"

CKPT_FILE="./checkpoints/diffmvs_dtu.ckpt"

OUT_DIR='./outputs_dtu'

if [ ! -d $OUT_DIR ]; then
    mkdir -p $OUT_DIR
fi

python test.py --dataset=dtu --batch_size=1 --num_view=5 --method=diffmvs --save_depth \
    --testpath=$DTU_TESTING  --numdepth_initial=48 --numdepth=384 \
    --testlist=lists/dtu/test.txt --loadckpt=$CKPT_FILE --outdir=$OUT_DIR \
    --scale 0.0 0.5 0.0 --sampling_timesteps 0 1 1 --ddim_eta 0 1 0 \
    --stage_iters 1 4 0 --cost_dim_stage 4 4 0 --CostNum 0 6 0 \
    --hidden_dim 0 32 0 --context_dim 32 32 0 --unet_dim 0 16 8 \
    --min_radius 0.25 --max_radius 4 \
    --geo_pixel_thres 0.125 --geo_depth_thres 0.01 --geo_mask_thres 2 \
    --photo_thres 0.3 0.0 0.0

