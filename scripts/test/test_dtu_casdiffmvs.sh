#!/usr/bin/env bash

DTU_TESTING="/dtu/"

CKPT_FILE="./checkpoints/casdiffmvs_dtu.ckpt"

OUT_DIR='./outputs_dtu_cas'

if [ ! -d $OUT_DIR ]; then
    mkdir -p $OUT_DIR
fi

python test.py --dataset=dtu --batch_size=1 --num_view=5 --method=casdiffmvs --save_depth \
    --testpath=$DTU_TESTING  --numdepth_initial=48 --numdepth=384 \
    --testlist=lists/dtu/test.txt --loadckpt=$CKPT_FILE --outdir=$OUT_DIR \
    --scale 0.0 0.5 0.1 --sampling_timesteps 0 1 1 --ddim_eta 0 1 1 \
    --stage_iters 1 3 3 --cost_dim_stage 4 4 4 --CostNum 0 4 4 \
    --hidden_dim 0 32 20 --context_dim 32 32 16 --unet_dim 0 16 8 \
    --min_radius 0.125 --max_radius 8 \
    --geo_pixel_thres 0.125 --geo_depth_thres 0.01 --geo_mask_thres 2 \
    --photo_thres 0.3 0.0 0.0

