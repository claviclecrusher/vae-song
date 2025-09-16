#!/usr/bin/env python3
"""
Sample generation script for trained VAE models.
Loads a saved checkpoint and generates samples.

Usage:
    python test.py --config configs/config_shapenet_setvae.yaml --param_dir results/result_setvae/SetVAE\ 09151706_b=0.2/params/model_99.pt --n_samples 100
"""

import argparse
import os
import torch
import numpy as np
import yaml
from torchvision.utils import save_image
import model as Model
import dataset

# For point cloud visualization
try:
    import open3d as o3d
    HAS_OPEN3D = True
except ImportError:
    HAS_OPEN3D = False
    print("Warning: open3d not available. .ply files will not be saved.")

def load_config(config_path):
    """Load YAML configuration file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def create_model_from_config(config):
    """Create model instance based on configuration."""
    exp_type = config['experiment_type']
    common_params = config['common_params']
    model_params = config['model_params']
    
    if exp_type == 'setvae':
        model = Model.SetVAE(
            beta=model_params.get('beta_list', [1.0])[0],
            latent_channel=model_params.get('latent_channel', 128),
            num_points=model_params.get('num_points', 2048),
            encoder_hidden=model_params.get('encoder_hidden', [128, 256, 512]),
            decoder_hidden=model_params.get('decoder_hidden', [512, 256, 128]),
            dataset='shapenet',
            pool_type=model_params.get('pool_type', 'max'),
        )
    elif exp_type == 'setlrvae':
        model = Model.SetLRVAE(
            alpha=model_params.get('alpha_list', [0.01])[0],
            beta=model_params.get('beta_list', [1.0])[0],
            latent_channel=model_params.get('latent_channel', 128),
            num_points=model_params.get('num_points', 2048),
            encoder_hidden=model_params.get('encoder_hidden', [128, 256, 512]),
            decoder_hidden=model_params.get('decoder_hidden', [512, 256, 128]),
            dataset='shapenet',
            pool_type=model_params.get('pool_type', 'max'),
        )
    elif exp_type == 'vae':
        model = Model.VanillaVAE(
            beta=model_params.get('beta_list', [1.0])[0],
            dataset=common_params.get('exp_data', 'mnist'),
            hidden_channels=model_params.get('hchans', None),
            encoder_type=model_params.get('encoder_type', 'conv'),
            decoder_type=model_params.get('decoder_type', 'mlp'),
            fixed_var=model_params.get('fixed_var', False),
            residual_connection=model_params.get('residual_connection', False)
        )
    elif exp_type == 'lrvae':
        model = Model.LRVAE(
            alpha=model_params.get('alpha_list', [0.01])[0],
            beta=model_params.get('beta_list', [1.0])[0],
            z_source=model_params.get('z_source', 'Ex'),
            dataset=common_params.get('exp_data', 'mnist'),
            hidden_channels=model_params.get('hchans', None),
            pwise_reg=model_params.get('pwise_reg', False),
            encoder_type=model_params.get('encoder_type', 'conv'),
            decoder_type=model_params.get('decoder_type', 'mlp'),
            residual_connection=model_params.get('residual_connection', False)
        )
    elif exp_type == 'nae':
        model = Model.NaiveAE(
            dataset=common_params.get('exp_data', 'mnist'),
            hidden_channels=model_params.get('hchans', None),
            encoder_type=model_params.get('encoder_type', 'conv'),
            decoder_type=model_params.get('decoder_type', 'mlp')
        )
    elif exp_type == 'lidvae':
        model = Model.LIDVAE(
            is_log_mse=model_params.get('log_mse', False),
            inverse_lipschitz=model_params.get('il_list', [0.0])[0],
            beta=model_params.get('beta_list', [1.0])[0],
            dataset=common_params.get('exp_data', 'mnist'),
            hidden_channels=model_params.get('hchans', None)
        )
    else:
        raise ValueError(f"Unsupported experiment type: {exp_type}")
    
    return model

def save_point_cloud(points, filepath):
    """Save point cloud in both .npy and .ply formats."""
    # Save as .npy
    np.save(filepath + '.npy', points)
    
    # Save as .ply if open3d is available
    if HAS_OPEN3D:
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        o3d.io.write_point_cloud(filepath + '.ply', pcd)

def generate_samples(model, n_samples, device='cuda', batch_size=32):
    """Generate samples from the model."""
    model.eval()
    samples = []
    
    with torch.no_grad():
        for i in range(0, n_samples, batch_size):
            current_batch_size = min(batch_size, n_samples - i)
            
            # Generate random latent vectors
            z = torch.randn(current_batch_size, model.latent_channel, device=device)
            
            # Generate samples
            if hasattr(model, 'data_type') and model.data_type == 'set':
                # Point cloud generation
                generated_points = model.decode(z)  # [B, N, 3]
                samples.append(generated_points.cpu().numpy())
            else:
                # Image generation
                generated_images = model.decode(z)  # [B, C, H, W]
                samples.append(generated_images.cpu())
    
    if hasattr(model, 'data_type') and model.data_type == 'set':
        # Concatenate point cloud samples
        return np.concatenate(samples, axis=0)
    else:
        # Concatenate image samples
        return torch.cat(samples, dim=0)

def main():
    parser = argparse.ArgumentParser(description='Generate samples from trained VAE model')
    parser.add_argument('--config', type=str, required=True, help='Path to YAML config file')
    parser.add_argument('--param_dir', type=str, required=True, help='Path to .pt checkpoint file')
    parser.add_argument('--n_samples', type=int, default=100, help='Number of samples to generate')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use (cuda/cpu)')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for generation')
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Create model
    model = create_model_from_config(config)
    
    # Load checkpoint
    if not os.path.exists(args.param_dir):
        raise FileNotFoundError(f"Checkpoint file not found: {args.param_dir}")
    
    checkpoint = torch.load(args.param_dir, map_location=args.device)
    model.load_state_dict(checkpoint)
    model = model.to(args.device)
    
    print(f"Loaded model from: {args.param_dir}")
    print(f"Model type: {type(model).__name__}")
    print(f"Generating {args.n_samples} samples...")
    
    # Create output directory
    output_dir = os.path.join(os.path.dirname(args.param_dir), 'gen_samples')
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate samples
    samples = generate_samples(model, args.n_samples, args.device, args.batch_size)
    
    # Save samples
    if hasattr(model, 'data_type') and model.data_type == 'set':
        # Point cloud samples
        print(f"Saving point cloud samples to: {output_dir}")
        for i, points in enumerate(samples):
            filepath = os.path.join(output_dir, f'sample_{i:04d}')
            save_point_cloud(points, filepath)
            if i % 10 == 0:
                print(f"Saved {i+1}/{len(samples)} samples")
    else:
        # Image samples
        print(f"Saving image samples to: {output_dir}")
        for i in range(0, len(samples), 16):  # Save in batches of 16
            batch = samples[i:i+16]
            filepath = os.path.join(output_dir, f'samples_{i//16:04d}.png')
            save_image(batch, filepath, normalize=True, nrow=4)
            if i % 64 == 0:
                print(f"Saved {i+len(batch)}/{len(samples)} samples")
    
    print(f"Generation complete! Samples saved to: {output_dir}")

if __name__ == "__main__":
    main()
