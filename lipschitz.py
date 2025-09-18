import argparse
import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch.nn.functional as F

# Import all necessary components from dataset, model, utils
from dataset import load_dataset, GridMixtureDataset, WeightedGridMixtureDataset, SimpleGaussianMixtureDataset
from model import LRVAE, LIDVAE
from utils import compute_local_reg, estimate_local_lipschitz, plot_heatmap, plot_2d_histogram, reparameterize, apply_grad_clip
import time
import random

# --- Constants ---
DEFAULT_EMPTY_CELL_FILL_VALUE = -5.0 # Default value to fill empty cells in heatmaps
# --- Constants End ---

# --- train_model function definition ---
def train_model(model, loader, epochs, lr, device, grad_clip=None, wu_strat='linear', wu_start_epoch=0, wu_up_amount=None, wu_repeat_interval=10, experiment_logger=None):
    model.to(device).train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    for epoch in tqdm(range(epochs), desc="Training Model"):
        # Apply warmup
        model.warmup(epoch=epoch, max_epoch=epochs, wu_strat=wu_strat, 
                    up_amount=wu_up_amount, start_epoch=wu_start_epoch, 
                    repeat_interval=wu_repeat_interval)
        
        # Alpha warmup 값 로깅 (LR 모델인 경우)
        if experiment_logger and hasattr(model, 'wu_alpha'):
            experiment_logger.log_alpha_value(epoch, model.wu_alpha)
        
        for X, _ in loader:
            X = X.to(device)
            optimizer.zero_grad()
            recon, mu, log_var, input_z_stack, z_recon_stack = model(X)
            total_loss, _, _, _ = model.loss(X, recon, mu, log_var, input_z_stack, z_recon_stack)
            total_loss.backward()
            apply_grad_clip(model, grad_clip)
            optimizer.step()
    return model
# --- train_model function end ---

# Calculate KL and Lipschitz for each grid cell (X-space)
def _get_kl_and_lipschitz_for_x_cells(model, test_dataset, K, device, nsamples_z=10, num_pairs_lips=100, empty_cell_fill_value=DEFAULT_EMPTY_CELL_FILL_VALUE):
    kl_vals = np.full(K * K, empty_cell_fill_value, dtype=np.float32)
    lips_vals = np.full(K * K, empty_cell_fill_value, dtype=np.float32)
    inv_lips_vals = np.full(K * K, empty_cell_fill_value, dtype=np.float32)
    bi_lips_vals = np.full(K * K, empty_cell_fill_value, dtype=np.float32)
    model.eval()
    with torch.no_grad():
        for cell_idx in range(K * K):
            X_cell = test_dataset.X[test_dataset.y == cell_idx].to(device)

            if X_cell.size(0) == 0:
                continue 
            
            mu_cell, log_var_cell = model.encode(X_cell)
            kl_div_per_sample = -0.5 * torch.sum(1 + log_var_cell - mu_cell.pow(2) - log_var_cell.exp(), dim=1)
            kl_vals[cell_idx] = kl_div_per_sample.mean().item()

            if X_cell.size(0) < 2:
                continue

            z_samples_for_cell = reparameterize(mu_cell, log_var_cell, nsamples=nsamples_z).view(-1, mu_cell.size(-1))
            use_grad_decode = (type(model).__name__ == 'LIDVAE')
            # LIDVAE의 decode는 autograd.grad를 사용하므로 grad를 활성화해야 함
            if use_grad_decode:
                # Gradient를 켜되 파라미터 업데이트는 방지
                with torch.enable_grad():
                    inv_lips, lips, bi_lips = estimate_local_lipschitz(
                        model.decode, z_samples_for_cell, num_pairs=num_pairs_lips, use_grad=use_grad_decode
                    )
            else:
                with torch.no_grad():
                    inv_lips, lips, bi_lips = estimate_local_lipschitz(
                        model.decode, z_samples_for_cell, num_pairs=num_pairs_lips, use_grad=use_grad_decode
                    )
            lips_vals[cell_idx] = lips
            inv_lips_vals[cell_idx] = inv_lips
            bi_lips_vals[cell_idx] = bi_lips
            
    return kl_vals, lips_vals, inv_lips_vals, bi_lips_vals

# Calculate KL and Lipschitz for each grid cell (Z-space)
def _get_kl_and_lipschitz_for_z_cells(model, K_z, z_min, z_max, actual_latent_dim, device, nsamples_z_per_cell=100, num_pairs_lips=100, empty_cell_fill_value=DEFAULT_EMPTY_CELL_FILL_VALUE):
    # Z-space grid evaluation is only meaningful for 2D latent spaces.
    if actual_latent_dim != 2:
        raise ValueError(f"Skipping Z-space grid evaluation: Model's actual latent dimension is {actual_latent_dim}D, not 2D.")
        #return np.full(K_z * K_z, empty_cell_fill_value, dtype=np.float32), np.full(K_z * K_z, empty_cell_fill_value, dtype=np.float32)

    kl_vals_z = np.full(K_z * K_z, empty_cell_fill_value, dtype=np.float32)
    lips_vals_z = np.full(K_z * K_z, empty_cell_fill_value, dtype=np.float32)
    inv_lips_vals_z = np.full(K_z * K_z, empty_cell_fill_value, dtype=np.float32)
    bi_lips_vals_z = np.full(K_z * K_z, empty_cell_fill_value, dtype=np.float32)
    
    model.eval()
    # Z-space grid centers 정의 (gradient 없이)
    z_centers_x = np.linspace(z_min, z_max, K_z)
    z_centers_y = np.linspace(z_min, z_max, K_z)
    
    z_grid_centers = []
    for y_idx in range(K_z):
        for x_idx in range(K_z):
            z_grid_centers.append([z_centers_x[x_idx], z_centers_y[y_idx]])
    z_grid_centers = torch.tensor(z_grid_centers, dtype=torch.float32, device=device)

    for cell_idx in range(K_z * K_z):
        z_center_sample = z_grid_centers[cell_idx]
        
        # Generate z samples around the Z-space cell center
        z_samples = z_center_sample.repeat(nsamples_z_per_cell, 1) + torch.randn(nsamples_z_per_cell, actual_latent_dim, device=device) * 0.1 
        
        # --- KL Calculation (Z -> X_recon -> Z_re_encoded) ---
        use_grad_decode = (type(model).__name__ == 'LIDVAE')
        if use_grad_decode:
            # Gradient를 켜되 파라미터 업데이트는 방지
            with torch.enable_grad():
                z_samples_req = z_samples.clone().detach().requires_grad_(True)
                x_recon = model.decode(z_samples_req)
                # encode는 gradient 없이
                with torch.no_grad():
                    mu_re, log_var_re = model.encode(x_recon.detach())
        else:
            with torch.no_grad():
                x_recon = model.decode(z_samples)
                mu_re, log_var_re = model.encode(x_recon)
        # KL Divergence for re-encoded Z (vs. standard normal prior p(z) = N(0,I))
        kl_div_per_sample = -0.5 * torch.sum(1 + log_var_re - mu_re.pow(2) - log_var_re.exp(), dim=1)
        kl_vals_z[cell_idx] = kl_div_per_sample.mean().item()

        # --- Decoder Lipschitz Calculation (Z -> X_recon) ---
        if z_samples.size(0) < 2:
            continue
        use_grad_decode = (type(model).__name__ == 'LIDVAE')
        if use_grad_decode:
            # Gradient를 켜되 파라미터 업데이트는 방지
            with torch.enable_grad():
                inv_lips, lips, bi_lips = estimate_local_lipschitz(
                    model.decode, z_samples, num_pairs=num_pairs_lips, use_grad=use_grad_decode
                )
        else:
            with torch.no_grad():
                inv_lips, lips, bi_lips = estimate_local_lipschitz(
                    model.decode, z_samples, num_pairs=num_pairs_lips, use_grad=use_grad_decode
                )
        lips_vals_z[cell_idx] = lips
        inv_lips_vals_z[cell_idx] = inv_lips
        bi_lips_vals_z[cell_idx] = bi_lips
            
    return kl_vals_z, lips_vals_z, inv_lips_vals_z, bi_lips_vals_z

# Calculate L(z) from actual data distribution
def _get_data_based_lipschitz(model, test_dataset, device, num_samples=5000, num_pairs_lips=5000, empty_cell_fill_value=DEFAULT_EMPTY_CELL_FILL_VALUE):
    """
    실제 데이터 분포로부터 충분한 수의 샘플을 생성하여 L(z)를 측정
    """
    model.eval()
    
    # 충분한 수의 데이터 포인트 인코딩
    with torch.no_grad():
        X_data = test_dataset.X.to(device)
        mu_data, log_var_data = model.encode(X_data)
        
        # 더 많은 z 샘플 생성 (데이터 개수보다 많게)
        if X_data.size(0) < num_samples:
            # 데이터가 부족하면 reparameterization으로 더 많은 샘플 생성
            z_samples = reparameterize(mu_data, log_var_data, nsamples=num_samples // X_data.size(0) + 1)
            z_samples = z_samples.view(-1, mu_data.size(-1))[:num_samples]
        else:
            # 데이터가 충분하면 일부만 사용
            indices = torch.randperm(X_data.size(0))[:num_samples]
            mu_subset = mu_data[indices]
            log_var_subset = log_var_data[indices]
            z_samples = reparameterize(mu_subset, log_var_subset, nsamples=1).squeeze(1)
    
    # LIDVAE인 경우 gradient 활성화
    use_grad_decode = (type(model).__name__ == 'LIDVAE')
    if use_grad_decode:
        with torch.enable_grad():
            inv_lips, lips, bi_lips = estimate_local_lipschitz(
                model.decode, z_samples, num_pairs=num_pairs_lips, use_grad=use_grad_decode
            )
    else:
        with torch.no_grad():
            inv_lips, lips, bi_lips = estimate_local_lipschitz(
                model.decode, z_samples, num_pairs=num_pairs_lips, use_grad=use_grad_decode
            )
    
    return inv_lips, lips, bi_lips

# Calculate KL divergence from actual data distribution
def _get_data_based_kl(model, test_dataset, device, num_samples=5000):
    """
    실제 데이터 분포로부터 충분한 수의 샘플을 생성하여 KL divergence를 측정
    """
    model.eval()
    
    with torch.no_grad():
        X_data = test_dataset.X.to(device)
        mu_data, log_var_data = model.encode(X_data)
        
        # 더 많은 z 샘플 생성 (데이터 개수보다 많게)
        if X_data.size(0) < num_samples:
            # 데이터가 부족하면 reparameterization으로 더 많은 샘플 생성
            z_samples = reparameterize(mu_data, log_var_data, nsamples=num_samples // X_data.size(0) + 1)
            z_samples = z_samples.view(-1, mu_data.size(-1))[:num_samples]
        else:
            # 데이터가 충분하면 일부만 사용
            indices = torch.randperm(X_data.size(0))[:num_samples]
            mu_subset = mu_data[indices]
            log_var_subset = log_var_data[indices]
            z_samples = reparameterize(mu_subset, log_var_subset, nsamples=1).squeeze(1)
        
        # KL divergence 계산: -0.5 * sum(1 + log_var - mu^2 - exp(log_var))
        kl_div_per_sample = -0.5 * torch.sum(1 + log_var_subset - mu_subset.pow(2) - log_var_subset.exp(), dim=1)
        avg_kl = kl_div_per_sample.mean().item()
    
    return avg_kl


def main():
    parser = argparse.ArgumentParser(description="Run VAE experiment for local Lipschitz and KL regularization.")
    parser.add_argument('--alpha', type=float, default=0.1, help='LRVAE alpha value (single value allowed).')
    parser.add_argument('--IL', type=float, default=0.0, help='LIDVAE inverse Lipschitz factor.')
    parser.add_argument('--model', type=str, default='lrvae', choices=['lrvae', 'lidvae'], help='Choose model type for experiment.')
    parser.add_argument('--K', type=int, default=16, help='Grid size (KxK) for data generation and visualization in X-space.')
    parser.add_argument('--std', type=float, default=0.1, help='Standard deviation for Gaussian components in data generation.')
    parser.add_argument('--epochs', type=int, default=1000, help='Number of training epochs.')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate for VAE training.')
    parser.add_argument('--beta', type=float, default=1.0, help='Beta for KL Divergence term in VAE loss.')
    parser.add_argument('--batch_size', type=int, default=256, help='Batch size for training and testing.')
    parser.add_argument('--device', type=str, default='cpu', help='Device to run on (cuda or cpu).')
    parser.add_argument('--output_dir', type=str, default='results/ablation', help='Output directory for results.')
    
    # Data distribution parameters
    parser.add_argument('--train_total_samples', type=int, default=10000, help='Total training samples.')
    parser.add_argument('--test_total_samples', type=int, default=10000, help='Total test samples (uniform).')
    parser.add_argument('--distribution_pattern', type=str, default='corner_heavy',
                        choices=['uniform', 'corner_heavy', 'center_heavy', 'sparse_random'],
                        help='Training data distribution pattern for SimpleGaussianMixtureDataset.')
    parser.add_argument('--seed', type=int, default=None, help='Random seed for reproducibility.')

    # Model parameters
    parser.add_argument('--latent_dim', type=int, default=2, 
                        help='Desired latent space dimension for Z visualization. Model\'s actual latent dimension is determined by the last value of --hidden_channels.')
    parser.add_argument('--hidden_channels', nargs='+', type=int, default=[64, 128, 64, 2], 
                        help='Hidden layer sizes for MLP in LRVAE. The last value determines the latent dimension (nz).')
    parser.add_argument('--num_training_components', type=int, default=8, 
                        help='Number of Gaussian components for the training dataset. (Should be much less than K*K for sparse regions).')

    # Z-space visualization parameters
    parser.add_argument('--K_z', type=int, default=16, help='Grid size (K_z x K_z) for Z-space visualization.')
    parser.add_argument('--z_min', type=float, default=-3.0, help='Minimum value for Z-space grid visualization range. None means use the actual range from the data.')
    parser.add_argument('--z_max', type=float, default=3.0, help='Maximum value for Z-space grid visualization range. None means use the actual range from the data.')


    # grad clipping options
    parser.add_argument('--grad_clip_enabled', action='store_true', help='Enable gradient clipping')
    parser.add_argument('--grad_clip_type', type=str, default='norm', choices=['norm', 'value'])
    parser.add_argument('--grad_clip_max_norm', type=float, default=1.0)
    parser.add_argument('--grad_clip_norm_type', type=float, default=2.0)
    parser.add_argument('--grad_clip_value', type=float, default=1.0)

    # warmup options
    parser.add_argument('--wu_strat', type=str, default='linear', 
                        choices=['linear', 'exponential', 'repeat_linear', 'kl_adaptive'],
                        help='Warmup strategy for LRVAE alpha')
    parser.add_argument('--wu_start_epoch', type=int, default=0, 
                        help='Epoch to start warmup')
    parser.add_argument('--wu_up_amount', type=float, default=None,
                        help='Manual warmup increment amount (overrides default calculation)')
    parser.add_argument('--wu_repeat_interval', type=int, default=10,
                        help='Repeat interval for repeat_linear warmup strategy')

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    if args.seed is None:
        args.seed = 42
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    actual_latent_dim = args.hidden_channels[-1]
    if actual_latent_dim != 2: 
        print(f"\n--- Warning: Model's actual latent dimension ({actual_latent_dim}) is not 2. ---")
        print(f"--- Z-space grid-based evaluation and visualization for {actual_latent_dim}D Z might not be meaningful. ---")
        print(f"--- If you want 2D Z-space grid evaluation, please set the last value of --hidden_channels to 2. ---")
    
    # 1. 2D Point Data Generation (Training)
    print(f"Generating training data with pattern: {args.distribution_pattern}")
    train_dataset = SimpleGaussianMixtureDataset(
        num_components=args.num_training_components,
        total_samples=args.train_total_samples,
        center_range=args.K,
        stds=args.std,
        pattern=args.distribution_pattern,
        seed=args.seed
    )
    g_loader = torch.Generator(device='cpu').manual_seed(args.seed)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True, generator=g_loader)

    # Visualize training data distribution
    X_train_vis = train_dataset.X.numpy()
    plot_2d_histogram(X_train_vis, bins=args.K,
                      title=f'Training Data Distribution ({args.distribution_pattern})',
                      filepath=os.path.join(args.output_dir, 'train_distribution_2d.png'))

    # 2. Model init & training
    is_lidvae = (args.model == 'lidvae')
    if is_lidvae:
        print("Initializing and training LIDVAE model...")
        model = LIDVAE(inverse_lipschitz=args.IL, beta=args.beta, dataset='pinwheel', hidden_channels=args.hidden_channels)
    else:
        print("Initializing and training LRVAE model...")
        model = LRVAE(alpha=args.alpha, dataset='pinwheel', hidden_channels=args.hidden_channels)
        model.beta = args.beta
        model.alpha = args.alpha
        model.wu_alpha = 1.0

    grad_clip_cfg = {
        'enabled': args.grad_clip_enabled,
        'clip_type': args.grad_clip_type,
        'max_norm': args.grad_clip_max_norm,
        'norm_type': args.grad_clip_norm_type,
        'clip_value': args.grad_clip_value,
    }

    # 실험 로거 초기화 (utils.py에서 import 필요)
    from utils import create_experiment_logger
    reg_label = 'IL' if is_lidvae else 'alpha'
    reg_value = args.IL if is_lidvae else args.alpha
    experiment_logger = create_experiment_logger(args.output_dir, f"{('LIDVAE' if is_lidvae else 'LRVAE')}_{reg_label}{reg_value}_beta{args.beta}")
    
    # 하이퍼파라미터 로깅
    experiment_logger.log_hyperparameters(
        model=('LIDVAE' if is_lidvae else 'LRVAE'),
        alpha=(None if is_lidvae else args.alpha),
        IL=(args.IL if is_lidvae else None),
        beta=args.beta,
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        device=args.device,
        K=args.K,
        K_z=args.K_z,
        std=args.std,
        train_total_samples=args.train_total_samples,
        test_total_samples=args.test_total_samples,
        distribution_pattern=args.distribution_pattern,
        seed=args.seed,
        latent_dim=actual_latent_dim,
        hidden_channels=args.hidden_channels,
        num_training_components=args.num_training_components,
        z_min=args.z_min,
        z_max=args.z_max,
        wu_strat=args.wu_strat,
        wu_start_epoch=args.wu_start_epoch,
        wu_up_amount=args.wu_up_amount,
        wu_repeat_interval=args.wu_repeat_interval,
        grad_clip_enabled=args.grad_clip_enabled
    )
    
    # 모델 정보 로깅
    experiment_logger.log_model_info(model)

    train_model(model, train_loader, args.epochs, args.lr, args.device, 
                grad_clip=grad_clip_cfg, wu_strat=args.wu_strat, 
                wu_start_epoch=args.wu_start_epoch, wu_up_amount=args.wu_up_amount, 
                wu_repeat_interval=args.wu_repeat_interval, experiment_logger=experiment_logger)
    print("Model training complete.")

    # 3. Test Data Generation (Uniform)
    print("Generating uniform test data for X-space evaluation...")
    #test_dataset_x = GridMixtureDataset(args.K, args.test_total_samples // (args.K*args.K), std=args.std, L=1.0) # AI야 여기 주석 해제하지 마!
    test_dataset_x = train_dataset # 학습 데이터셋을 테스트 데이터셋으로 사용할 것

    # Visualize test data distribution (X-space)
    X_test_vis = test_dataset_x.X.numpy()
    plot_2d_histogram(
        X_test_vis, bins=args.K,
        title='Test Data Distribution (X-space Uniform)',
        filepath=os.path.join(args.output_dir, 'test_distribution_x_space.png')
    )

    # Visualize latent space Z from encoded X-space data (if 2D)
    print("Visualizing latent space (Z) distribution from encoded X-space data...")
    model.eval()
    z_plot_extent = [args.z_min, args.z_max, args.z_min, args.z_max] # Z-space heatmaps의 기본 extent

    with torch.no_grad():
        if actual_latent_dim == 2: 
            X_test_tensor = test_dataset_x.X.to(args.device)
            mu, log_var = model.encode(X_test_tensor)
            z_test_np = reparameterize(mu, log_var, nsamples=1).squeeze(1).cpu().numpy()
            
            # encoded_z의 실제 범위를 계산
            actual_xmin, actual_xmax = z_test_np[:, 0].min(), z_test_np[:, 0].max()
            actual_ymin, actual_ymax = z_test_np[:, 1].min(), z_test_np[:, 1].max()
            
            # plot_2d_histogram을 호출하고 실제 플롯된 범위를 받아옵니다.
            plot_2d_histogram(
                z_test_np, bins=args.K_z, 
                title=f'Encoded Latent Z Distribution (from X-space, Actual dim={actual_latent_dim})',
                filepath=os.path.join(args.output_dir, f'encoded_z_alpha{args.alpha}.png')
            )
            print(f"Latent space (Z) visualization for {args.alpha}")
            
            # encoded_z의 실제 범위를 z_plot_extent로 설정 (모든 Z-space heatmap에 동일하게 적용)
            z_plot_extent = [actual_xmin, actual_xmax, actual_ymin, actual_ymax]
            print(f"Z-space extent set to: x=[{actual_xmin:.3f}, {actual_xmax:.3f}], y=[{actual_ymin:.3f}, {actual_ymax:.3f}]")

        else:
            print(f"Skipping Z space 2D histogram visualization from X-space: Model's actual latent dimension is {actual_latent_dim}D, not 2D for direct plotting.")
            print("To visualize higher-dimensional Z, consider using dimensionality reduction (e.g., t-SNE) in a separate script or function.")
    
    # 4. KL and Decoder Bi-Lipschitz Visualization (X-space based)
    print(f"\nEvaluating metrics based on X-space grid (K={args.K})...")
    if not is_lidvae:
        model.alpha = args.alpha
        model.wu_alpha = args.alpha

    cell_kl_vals_x, cell_lips_vals_x, cell_inv_lips_vals_x, cell_bi_lips_vals_x = _get_kl_and_lipschitz_for_x_cells(model, test_dataset_x, args.K, args.device, nsamples_z=10, num_pairs_lips=2000)

    plot_heatmap(cell_kl_vals_x, args.K, f"KL Div (X-space, {reg_label}={reg_value})",
                 os.path.join(args.output_dir, f"kl_div_x_space_{reg_label}_{reg_value}.png"), cmap='viridis')

    plot_heatmap(cell_lips_vals_x, args.K, f"Local forward Lipschitz (X-space, {reg_label}={reg_value})",
                 os.path.join(args.output_dir, f"lips_x_space_{reg_label}_{reg_value}.png"), cmap='viridis')

    plot_heatmap(cell_inv_lips_vals_x, args.K, f"Local inverse Lipschitz (X-space, {reg_label}={reg_value})",
                 os.path.join(args.output_dir, f"inv_lips_x_space_{reg_label}_{reg_value}.png"), cmap='viridis')

    plot_heatmap(cell_bi_lips_vals_x, args.K, f"Local bi-Lipschitz (X-space, {reg_label}={reg_value})",
                 os.path.join(args.output_dir, f"bi_lips_x_space_{reg_label}_{reg_value}.png"), cmap='viridis')

    # 5. KL and Decoder Bi-Lipschitz Visualization (Z-space based)
    # encoded_z의 실제 범위를 사용하여 Z-space heatmap 생성
    if actual_latent_dim == 2:
        z_min_actual, z_max_actual = z_plot_extent[0], z_plot_extent[1]  # x 범위 사용
    else:
        z_min_actual, z_max_actual = args.z_min, args.z_max  # 기본값 사용
    
    cell_kl_vals_z, cell_lips_vals_z, cell_inv_lips_vals_z, cell_bi_lips_vals_z = _get_kl_and_lipschitz_for_z_cells(
        model, args.K_z, z_min_actual, z_max_actual, actual_latent_dim, args.device,
        nsamples_z_per_cell=100, num_pairs_lips=2000, empty_cell_fill_value=DEFAULT_EMPTY_CELL_FILL_VALUE
    )
    
    if not np.all(cell_kl_vals_z == DEFAULT_EMPTY_CELL_FILL_VALUE):
        print(f"\nEvaluating metrics based on Z-space grid (K_z={args.K_z})...")
        # Z-space heatmaps에 Z 시각화에서 얻은 extent를 적용
        plot_heatmap(cell_kl_vals_z, args.K_z, f"KL Div (Z-space, {reg_label}={reg_value})",
                     os.path.join(args.output_dir, f"kl_div_z_space_{reg_label}_{reg_value}.png"), cmap='viridis', extent=z_plot_extent) # extent 적용

        plot_heatmap(cell_lips_vals_z, args.K_z, f"Local forward Lipschitz (Z-space, {reg_label}={reg_value})",
                     os.path.join(args.output_dir, f"lips_z_space_{reg_label}_{reg_value}.png"), cmap='viridis', extent=z_plot_extent) # extent 적용

        plot_heatmap(cell_inv_lips_vals_z, args.K_z, f"Local inverse Lipschitz (Z-space, {reg_label}={reg_value})",
                     os.path.join(args.output_dir, f"inv_lips_z_space_{reg_label}_{reg_value}.png"), cmap='viridis', extent=z_plot_extent) # extent 적용

        plot_heatmap(cell_bi_lips_vals_z, args.K_z, f"Local bi-Lipschitz (Z-space, {reg_label}={reg_value})",
                     os.path.join(args.output_dir, f"bi_lips_z_space_{reg_label}_{reg_value}.png"), cmap='viridis', extent=z_plot_extent) # extent 적용
    else:
        print(f"Z-space grid evaluation skipped as actual latent dimension is not 2D.")

    # 6. Data-based measurements for final metrics
    print(f"\nMeasuring KL and L(z) from actual data distribution...")
    data_kl = _get_data_based_kl(model, test_dataset_x, args.device, num_samples=5000)
    data_inv_lips, data_lips, data_bi_lips = _get_data_based_lipschitz(
        model, test_dataset_x, args.device, num_samples=5000, num_pairs_lips=5000
    )
    print(f"Data-based KL measurement: {data_kl:.4f}")
    print(f"Data-based L(z) measurement: inv_lips={data_inv_lips:.4f}, lips={data_lips:.4f}, bi_lips={data_bi_lips:.4f}")


    # Record and save results
    records = []
    for cell_idx in range(args.K * args.K): 
        records.append({
            'alpha': reg_value,
            'space': 'X',
            'cell_idx': cell_idx,
            'kl_div': float(cell_kl_vals_x[cell_idx]),
            'lipschitz': float(cell_lips_vals_x[cell_idx])
        })
    
    if not np.all(cell_kl_vals_z == DEFAULT_EMPTY_CELL_FILL_VALUE):
        for cell_idx_z in range(args.K_z * args.K_z):
            records.append({
                'alpha': reg_value,
                'space': 'Z',
                'cell_idx': cell_idx_z,
                'kl_div': float(cell_kl_vals_z[cell_idx_z]),
                'lipschitz': float(cell_lips_vals_z[cell_idx_z])
            })

    df = pd.DataFrame(records)
    df.to_csv(os.path.join(args.output_dir, 'experiment_metrics.csv'), index=False)
    
    # Calculate overall metrics for exp_lip.csv
    # KL divergence: use data-based measurement instead of grid average
    avg_kl = data_kl  # Use the data-based measurement
    
    # Bi-Lipschitz constant L(z): use data-based measurement instead of grid average
    avg_bi_lips = data_bi_lips  # Use the data-based measurement
    
    # Create exp_lip.csv entry
    exp_lip_entry = {
        'alpha': reg_value,
        'beta': args.beta,
        'kl': avg_kl,
        'L(z)': avg_bi_lips
    }
    
    # Append to exp_lip.csv
    exp_lip_file = os.path.join(os.path.dirname(args.output_dir), 'exp_lip.csv')
    exp_lip_df = pd.DataFrame([exp_lip_entry])
    
    if os.path.exists(exp_lip_file):
        exp_lip_df.to_csv(exp_lip_file, mode='a', header=False, index=False)
    else:
        exp_lip_df.to_csv(exp_lip_file, index=False)
    
    # 평가 메트릭 로깅
    experiment_logger.log_evaluation_metrics(
        kl=avg_kl, 
        bi_lipschitz=avg_bi_lips,
        data_based_kl=data_kl,
        data_based_bi_lips=data_bi_lips,
        data_based_inv_lips=data_inv_lips,
        data_based_lips=data_lips
    )
    
    # Alpha warmup 요약 로깅
    experiment_logger.log_alpha_warmup_summary(args.wu_strat)
    
    # 로그 파일 마무리
    experiment_logger.finalize_log()
    
    print(f"Experiment complete. Results saved to {args.output_dir}")
    print(f"Overall metrics - KL (data-based): {avg_kl:.4f}, Bi-Lipschitz L(z) (data-based): {avg_bi_lips:.4f}")
    print(f"Data-based KL: {data_kl:.4f}")
    print(f"Data-based L(z) details - inv_lips: {data_inv_lips:.4f}, lips: {data_lips:.4f}, bi_lips: {data_bi_lips:.4f}")
    print(f"Experiment log saved to: {experiment_logger.log_file}")

if __name__ == '__main__':
    main()
