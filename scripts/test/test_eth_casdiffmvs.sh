#!/usr/bin/env bash

ETH3D_TESTING="/eth3d_high_res_test/"
CKPT_FILE="./checkpoints/casdiffmvs_blend.ckpt"
OUT_DIR='./outputs_eth3d_cas'
if [ ! -d $OUT_DIR ]; then
    mkdir -p $OUT_DIR
fi

python test.py --dataset=eth3d --batch_size=1 --num_view=10 --method=casdiffmvs --save_depth \
    --testpath=$ETH3D_TESTING  --numdepth_initial=48 --numdepth=384 \
    --testlist=lists/eth3d/train.txt --loadckpt=$CKPT_FILE --outdir=$OUT_DIR \
    --scale 0.0 0.125 0.025 --sampling_timesteps 0 1 1 --ddim_eta 0 1 1 \
    --stage_iters 1 3 3 --cost_dim_stage 4 4 4 --CostNum 0 4 4 \
    --hidden_dim 0 32 20 --context_dim 32 32 16 --unet_dim 0 16 8 \
    --min_radius 0.125 --max_radius 8 \
    --photo_thres 0.3 0.5 0.5

python test.py --dataset=eth3d --batch_size=1 --num_view=10 --method=casdiffmvs --save_depth \
    --testpath=$ETH3D_TESTING  --numdepth_initial=48 --numdepth=384 \
    --testlist=lists/eth3d/test.txt --loadckpt=$CKPT_FILE --outdir=$OUT_DIR \
    --scale 0.0 0.125 0.025 --sampling_timesteps 0 1 1 --ddim_eta 0 1 1 \
    --stage_iters 1 3 3 --cost_dim_stage 4 4 4 --CostNum 0 4 4 \
    --hidden_dim 0 32 20 --context_dim 32 32 16 --unet_dim 0 16 8 \
    --min_radius 0.125 --max_radius 8 \
    --photo_thres 0.3 0.5 0.5