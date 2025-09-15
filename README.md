# VAE-Song

Flexible Variational Autoencoder Framework for Multiple Datasets

## Project Overview
This repository provides a flexible implementation of Variational Autoencoders (VAEs) that can be applied to both image and two-dimensional point datasets. Supported datasets include MNIST, Fashion-MNIST, CelebA, CIFAR-10, Omniglot, as well as synthetic 2D distributions such as Pinwheel and Chessboard.

## Key Features
- Modular architecture supporting MLP, convolutional, and residual blocks
- Choice of decoder: standard neural network or ICNN-based Brenier map
- Support for different VAE variants: Beta-VAE, Latent Reconstruction VAE (LR-VAE), and Lipschitz Invertible Decoder VAE (LID-VAE)
- YAML-based configuration for reproducible experiments
- Integrated logging with TensorBoard and CSV output
- Built-in visualization tools: PCA plots, t-SNE, optimal transport heatmaps, and sample trajectories

## Installation
```bash
git clone <repository_url>
cd vae-song
# Create and activate conda environment
conda env create -f lrvae_env.yaml
conda activate lrvae
# Install additional requirements if needed
pip install -r requirements.txt
```


# data download (shapenet)
```bash
gdown --fuzzy https://drive.google.com/file/d/1sw9gdk_igiyyt7MqALyxZhRrtPvAn0sX/view?usp=drive_link
```

## Usage
To run a default experiment on the Pinwheel dataset:
```bash
python main.py
```
To run with a different configuration file:
```bash
python main.py --config configs/config_vae.yaml
python run_vis_lip_kl_exp.py \
    --alpha 0.0 \
    --K 8 \
    --std 0.3 \
    --epochs 100 \
    --lr 1e-3 \
    --beta 0.001 \
    --batch_size 256 \
    --output_dir results/kl_lips_x_z_space_exp \
    --train_total_samples 20000 \
    --test_total_samples 10000 \
    --distribution_pattern corner_heavy \
    --latent_dim 2 \
    --hidden_channels 128 64 64 32 16 8 4 2 \
    --num_training_components 8 \
    --seed 20 \
    --K_z 16 \
    --z_min -3.0 \
    --z_max 3.0
```
Experiment logs will be saved under `runs/` for TensorBoard and `results/` for images and model checkpoints.

## Project Structure
```
vae-song/
├── main.py          # Main training and evaluation script
├── dataset.py       # Data loading and preprocessing
├── model.py         # VAE model definitions (FlexibleVAE, VanillaVAE, LIDVAE, etc.)
├── utils.py         # Utility functions for visualization and metrics
├── module.py        # Building blocks (residual blocks, ICNN, etc.)
├── configs/         # YAML configuration files for experiments
├── runs/            # TensorBoard log directory
├── results/         # Output images and saved model checkpoints
├── README.md        # Project overview and instructions
└── lrvae_env.yaml   # Conda environment specification
```

## Authors
- Hyunsoo Song (clavicle.shatter@gmail.com / song@nims.re.kr)
- Seungwhan Kim (overnap@gmail.com / seunghwan.kim@snu.ac.kr)

## License
This project is licensed under the MIT License. See the LICENSE file for details.



