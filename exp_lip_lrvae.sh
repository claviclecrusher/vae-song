#!/bin/bash


alphas=(0.0 0.1 0.2 0.3 0.4)
#betas=(0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0)
betas=(0.7 0.8 0.9 1.0)
#seeds=(42 43 44 45 46 47 48 49 50)
seeds=(42 43 44 45)


for beta in "${betas[@]}"
do
  for alpha in "${alphas[@]}"
  do
    for seed in "${seeds[@]}"
    do
      CUDA_VISIBLE_DEVICES=1 python lipschitz.py --epochs 1000 --alpha $alpha --beta $beta --K 16 --K_z 16 --z_min -3 --z_max 3 --device cuda:0 --output_dir ./results/ablation_lrvae_linear/alpha_${alpha}_beta_${beta}_seed_${seed} --seed $seed --wu_strat linear --wu_start_epoch 0
    done
  done
done


# nohup bash exp_lip_lrvae.sh > nohup_exp_lip_lrvae.out 2>&1 &