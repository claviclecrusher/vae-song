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
from model import LRVAE
from utils import compute_local_reg, estimate_local_lipschitz, plot_heatmap, plot_2d_histogram, reparameterize
import time

# --- Constants ---
DEFAULT_EMPTY_CELL_FILL_VALUE = -5.0 # Default value to fill empty cells in heatmaps
# --- Constants End ---

# --- train_model function definition ---
def train_model(model, loader, epochs, lr, device):
    model.to(device).train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    for epoch in tqdm(range(epochs), desc="Training Model"):
        for X, _ in loader:
            X = X.to(device)
            optimizer.zero_grad()
            recon, mu, log_var, input_z_stack, z_recon_stack = model(X)
            total_loss, _, _, _ = model.loss(X, recon, mu, log_var, input_z_stack, z_recon_stack)
            total_loss.backward()
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
            inv_lips, lips, bi_lips = estimate_local_lipschitz(model.decode, z_samples_for_cell, num_pairs=num_pairs_lips)
            lips_vals[cell_idx] = lips
            inv_lips_vals[cell_idx] = inv_lips
            bi_lips_vals[cell_idx] = bi_lips
            
    return kl_vals, lips_vals, inv_lips_vals, bi_lips_vals

# Calculate KL and Lipschitz for each grid cell (Z-space)
def _get_kl_and_lipschitz_for_z_cells(model, K_z, z_min, z_max, actual_latent_dim, device, nsamples_z_per_cell=100, num_pairs_lips=100, empty_cell_fill_value=DEFAULT_EMPTY_CELL_FILL_VALUE):
    # Z-space grid evaluation is only meaningful for 2D latent spaces.
    if actual_latent_dim != 2:
        print(f"Skipping Z-space grid evaluation: Model's actual latent dimension is {actual_latent_dim}D, not 2D.")
        return np.full(K_z * K_z, empty_cell_fill_value, dtype=np.float32), np.full(K_z * K_z, empty_cell_fill_value, dtype=np.float32)

    kl_vals_z = np.full(K_z * K_z, empty_cell_fill_value, dtype=np.float32)
    lips_vals_z = np.full(K_z * K_z, empty_cell_fill_value, dtype=np.float32)
    inv_lips_vals_z = np.full(K_z * K_z, empty_cell_fill_value, dtype=np.float32)
    bi_lips_vals_z = np.full(K_z * K_z, empty_cell_fill_value, dtype=np.float32)
    
    model.eval()
    with torch.no_grad():
        # Define Z-space grid centers
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
            x_recon = model.decode(z_samples)
            mu_re, log_var_re = model.encode(x_recon)
            # KL Divergence for re-encoded Z (vs. standard normal prior p(z) = N(0,I))
            kl_div_per_sample = -0.5 * torch.sum(1 + log_var_re - mu_re.pow(2) - log_var_re.exp(), dim=1)
            kl_vals_z[cell_idx] = kl_div_per_sample.mean().item()

            # --- Decoder Lipschitz Calculation (Z -> X_recon) ---
            if z_samples.size(0) < 2:
                continue
            inv_lips, lips, bi_lips = estimate_local_lipschitz(model.decode, z_samples, num_pairs=num_pairs_lips)
            lips_vals_z[cell_idx] = lips
            inv_lips_vals_z[cell_idx] = inv_lips
            bi_lips_vals_z[cell_idx] = bi_lips
            
    return kl_vals_z, lips_vals_z, inv_lips_vals_z, bi_lips_vals_z


def main():
    parser = argparse.ArgumentParser(description="Run VAE experiment for local Lipschitz and KL regularization.")
    parser.add_argument('--alpha', type=float, default=0.1, help='LRVAE alpha value (single value allowed).')
    parser.add_argument('--K', type=int, default=16, help='Grid size (KxK) for data generation and visualization in X-space.')
    parser.add_argument('--std', type=float, default=0.1, help='Standard deviation for Gaussian components in data generation.')
    parser.add_argument('--epochs', type=int, default=1000, help='Number of training epochs.')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate for VAE training.')
    parser.add_argument('--beta', type=float, default=1.0, help='Beta for KL Divergence term in VAE loss.')
    parser.add_argument('--batch_size', type=int, default=256, help='Batch size for training and testing.')
    parser.add_argument('--device', type=str, default='cpu', help='Device to run on (cuda or cpu).')
    parser.add_argument('--output_dir', type=str, default='results/exp_new_setup', help='Output directory for results.')
    
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
    parser.add_argument('--z_min', type=float, default=-3.0, help='Minimum value for Z-space grid visualization range.')
    parser.add_argument('--z_max', type=float, default=3.0, help='Maximum value for Z-space grid visualization range.')


    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    if args.seed is None:
        args.seed = int(time.time())
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

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
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)

    # Visualize training data distribution
    X_train_vis = train_dataset.X.numpy()
    plot_2d_histogram(X_train_vis, bins=args.K,
                      title=f'Training Data Distribution ({args.distribution_pattern})',
                      filepath=os.path.join(args.output_dir, 'train_distribution_2d.png'))

    # 2. LRVAE Model Training
    print("Initializing and training LRVAE model...")
    model = LRVAE(alpha=args.alpha, dataset='pinwheel', hidden_channels=args.hidden_channels)
    model.beta = args.beta
    model.alpha = args.alpha
    model.wu_alpha = 1.0

    train_model(model, train_loader, args.epochs, args.lr, args.device)
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
            
            # plot_2d_histogram을 호출하고 실제 플롯된 범위를 받아옵니다.
            actual_xmin, actual_xmax, actual_ymin, actual_ymax = plot_2d_histogram(
                z_test_np, bins=args.K_z, 
                title=f'Encoded Latent Z Distribution (from X-space, Actual dim={actual_latent_dim})',
                filepath=os.path.join(args.output_dir, f'encoded_z_alpha{args.alpha}.png')
            )
            print(f"Latent space (Z) visualization for {args.alpha}")
            
            # `--z_min`과 `--z_max`가 기본값일 경우, 실제 플롯된 Z 범위로 extent를 업데이트합니다.
            # 약간의 여유를 두거나, 정확히 플롯된 경계를 사용합니다.
            # 여기서는 플롯된 경계를 직접 사용합니다.
            if args.z_min == -3.0 and args.z_max == 3.0: # 기본값 사용 여부 확인
                 z_plot_extent = [actual_xmin, actual_xmax, actual_ymin, actual_ymax]
                 # 필요하다면 경계에 약간의 여유를 줄 수 있습니다. (예: * 1.1)
                 # z_plot_extent = [actual_xmin * 1.1 if actual_xmin < 0 else actual_xmin * 0.9, 
                 #                  actual_xmax * 1.1 if actual_xmax > 0 else actual_xmax * 0.9,
                 #                  actual_ymin * 1.1 if actual_ymin < 0 else actual_ymin * 0.9,
                 #                  actual_ymax * 1.1 if actual_ymax > 0 else actual_ymax * 0.9]
            else:
                 # 사용자가 z_min, z_max를 지정한 경우 해당 범위 사용
                 z_plot_extent = [args.z_min, args.z_max, args.z_min, args.z_max]

        else:
            print(f"Skipping Z space 2D histogram visualization from X-space: Model's actual latent dimension is {actual_latent_dim}D, not 2D for direct plotting.")
            print("To visualize higher-dimensional Z, consider using dimensionality reduction (e.g., t-SNE) in a separate script or function.")
    
    # 4. KL and Decoder Bi-Lipschitz Visualization (X-space based)
    print(f"\nEvaluating metrics based on X-space grid (K={args.K})...")
    model.alpha = args.alpha
    model.wu_alpha = args.alpha 

    cell_kl_vals_x, cell_lips_vals_x, cell_inv_lips_vals_x, cell_bi_lips_vals_x = _get_kl_and_lipschitz_for_x_cells(model, test_dataset_x, args.K, args.device)

    plot_heatmap(cell_kl_vals_x, args.K, f"KL Div (X-space, alpha={args.alpha})",
                 os.path.join(args.output_dir, f"kl_div_x_space_alpha_{args.alpha}.png"), cmap='viridis')

    plot_heatmap(cell_lips_vals_x, args.K, f"Local forward Lipschitz (X-space, alpha={args.alpha})",
                 os.path.join(args.output_dir, f"lips_x_space_alpha_{args.alpha}.png"), cmap='viridis')

    plot_heatmap(cell_inv_lips_vals_x, args.K, f"Local inverse Lipschitz (X-space, alpha={args.alpha})",
                 os.path.join(args.output_dir, f"inv_lips_x_space_alpha_{args.alpha}.png"), cmap='viridis')

    plot_heatmap(cell_bi_lips_vals_x, args.K, f"Local bi-Lipschitz (X-space, alpha={args.alpha})",
                 os.path.join(args.output_dir, f"bi_lips_x_space_alpha_{args.alpha}.png"), cmap='viridis')

    # 5. KL and Decoder Bi-Lipschitz Visualization (Z-space based)
    cell_kl_vals_z, cell_lips_vals_z, cell_inv_lips_vals_z, cell_bi_lips_vals_z = _get_kl_and_lipschitz_for_z_cells(
        model, args.K_z, args.z_min, args.z_max, actual_latent_dim, args.device,
        nsamples_z_per_cell=100, num_pairs_lips=100, empty_cell_fill_value=DEFAULT_EMPTY_CELL_FILL_VALUE
    )
    
    if not np.all(cell_kl_vals_z == DEFAULT_EMPTY_CELL_FILL_VALUE):
        print(f"\nEvaluating metrics based on Z-space grid (K_z={args.K_z})...")
        # Z-space heatmaps에 Z 시각화에서 얻은 extent를 적용
        plot_heatmap(cell_kl_vals_z, args.K_z, f"KL Div (Z-space, alpha={args.alpha})",
                     os.path.join(args.output_dir, f"kl_div_z_space_alpha_{args.alpha}.png"), cmap='viridis', extent=z_plot_extent) # extent 적용

        plot_heatmap(cell_lips_vals_z, args.K_z, f"Local forward Lipschitz (Z-space, alpha={args.alpha})",
                     os.path.join(args.output_dir, f"lips_z_space_alpha_{args.alpha}.png"), cmap='viridis', extent=z_plot_extent) # extent 적용

        plot_heatmap(cell_inv_lips_vals_z, args.K_z, f"Local inverse Lipschitz (Z-space, alpha={args.alpha})",
                     os.path.join(args.output_dir, f"inv_lips_z_space_alpha_{args.alpha}.png"), cmap='viridis', extent=z_plot_extent) # extent 적용

        plot_heatmap(cell_bi_lips_vals_z, args.K_z, f"Local bi-Lipschitz (Z-space, alpha={args.alpha})",
                     os.path.join(args.output_dir, f"bi_lips_z_space_alpha_{args.alpha}.png"), cmap='viridis', extent=z_plot_extent) # extent 적용
    else:
        print(f"Z-space grid evaluation skipped as actual latent dimension is not 2D.")


    # Record and save results
    records = []
    for cell_idx in range(args.K * args.K): 
        records.append({
            'alpha': args.alpha,
            'space': 'X',
            'cell_idx': cell_idx,
            'kl_div': float(cell_kl_vals_x[cell_idx]),
            'lipschitz': float(cell_lips_vals_x[cell_idx])
        })
    
    if not np.all(cell_kl_vals_z == DEFAULT_EMPTY_CELL_FILL_VALUE):
        for cell_idx_z in range(args.K_z * args.K_z):
            records.append({
                'alpha': args.alpha,
                'space': 'Z',
                'cell_idx': cell_idx_z,
                'kl_div': float(cell_kl_vals_z[cell_idx_z]),
                'lipschitz': float(cell_lips_vals_z[cell_idx_z])
            })

    df = pd.DataFrame(records)
    df.to_csv(os.path.join(args.output_dir, 'experiment_metrics.csv'), index=False)
    print(f"Experiment complete. Results saved to {args.output_dir}")

if __name__ == '__main__':
    main()
