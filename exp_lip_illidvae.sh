#!/bin/bash


ILs=(0.0 0.1 0.2 0.3 0.4)
betas=(0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0)
seeds=(42 43 44 45 46 47 48 49 50)
DEVICE=1

for beta in "${betas[@]}"
do
  for IL in "${ILs[@]}"
  do
    for seed in "${seeds[@]}"
    do
      CUDA_VISIBLE_DEVICES=$DEVICE python lipschitz.py --model lidvae --epochs 1000 --IL $IL --beta $beta --K 16 --K_z 16 --z_min -3 --z_max 3 --device cuda:0 --output_dir ./results/ablation_IL/IL_${IL}_beta_${beta}_seed_${seed} --seed $seed
    done
  done
done