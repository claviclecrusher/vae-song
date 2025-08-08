alphas=(0.0 0.01 0.001 0.0001 0.1 0.2 0.3 0.4 1.0)
betas=(0.1 0.2 1.0)

for beta in "${betas[@]}"
do
  for alpha in "${alphas[@]}"
  do
    python run_vis_lip_kl_exp.py --alpha $alpha --beta $beta --K 16 --K_z 16 --z_min -3 --z_max 3 --device cuda:0 --output_dir ./results/vis_lip_kl_exp/alpha_${alpha}_beta_${beta}
  done
done