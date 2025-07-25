import argparse
import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
from dataset import GridMixtureDataset, WeightedGridMixtureDataset, RandomGaussianMixtureDataset
from model import LRVAE
from utils import compute_local_reg, estimate_local_lipschitz, plot_heatmap


def train_model(model, loader, epochs, lr, device):
    model.to(device).train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    for _ in tqdm(range(epochs)):
        for X, _ in loader:
            X = X.to(device)
            optimizer.zero_grad()
            recon, mu, log_var, input_z_stack, z_recon_stack = model(X)
            loss, _, _, _ = model.loss(X, recon, mu, log_var, input_z_stack, z_recon_stack)
            loss.backward()
            optimizer.step()
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--alphas', nargs='+', type=float, required=True)
    parser.add_argument('--K', type=int, default=16)
    parser.add_argument('--N0', type=int, default=500)
    parser.add_argument('--std', type=float, default=0.1)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--beta', type=float, default=1.0)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--output_dir', type=str, default='results/alpha_exp')
    parser.add_argument('--train_weights', nargs='+', type=float, default=None, help='훈련용 셀별 가중치 (길이 K*K)')
    parser.add_argument('--train_total', type=int, default=None, help='훈련용 전체 샘플 수')
    parser.add_argument('--test_N0', type=int, default=None, help='테스트용 셀당 샘플 개수')
    parser.add_argument('--auto_weights', action='store_true', help='훈련용 가중치를 자동으로 생성 (불균일)')
    parser.add_argument('--seed', type=int, default=None, help='auto_weights seed')
    parser.add_argument('--random_mixture', action='store_true', help='Use random Gaussian mixture dataset')
    parser.add_argument('--num_components', type=int, default=None, help='Number of Gaussian components')
    parser.add_argument('--rgm_weights', nargs='+', type=float, default=None, help='Weights for Gaussian components')
    parser.add_argument('--rgm_total', type=int, default=None, help='Total samples for Gaussian mixture')
    parser.add_argument('--rgm_std', type=float, default=None, help='Std for Gaussian components')
    parser.add_argument('--rgm_L', type=float, default=None, help='Range L for Gaussian centers')
    parser.add_argument('--rgm_seed', type=int, default=None, help='Seed for Gaussian mixture')
    parser.add_argument('--grid_model', action='store_true', help='Use deeper network architecture for grid data')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # 훈련용 데이터 분포 설정: 명시적 weights 또는 auto_weights가 설정되면 불균일, 아니면 균일
    if args.random_mixture:
        assert args.num_components is not None, '--num_components is required for random_mixture'
        total = args.rgm_total if args.rgm_total is not None else args.K * args.K * args.N0
        Lval = args.rgm_L if args.rgm_L is not None else 1.0
        stdval = args.rgm_std if args.rgm_std is not None else args.std
        dataset = RandomGaussianMixtureDataset(
            args.num_components, total,
            weights=args.rgm_weights,
            std=stdval, L=Lval,
            seed=args.rgm_seed)
    elif args.train_weights is not None or args.auto_weights:
        # weights 결정
        if args.train_weights is not None:
            train_weights = args.train_weights
        else:
            # sparse한 불균일 분포 생성을 위해 exponential 분포 사용
            if args.seed is not None:
                np.random.seed(args.seed)
            w = np.random.exponential(scale=1.0, size=(args.K * args.K,))
            w = w / w.sum()
            train_weights = w.tolist()
        # total 샘플 수 결정
        train_total = args.train_total if args.train_total is not None else args.K * args.K * args.N0
        dataset = WeightedGridMixtureDataset(args.K, train_weights, train_total, std=args.std, L=1.0)
    else:
        dataset = GridMixtureDataset(args.K, args.N0, std=args.std, L=1.0)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    # Training data distribution visualization (grayscale heatmap with actual data points)
    counts = np.bincount(dataset.y.cpu().numpy(), minlength=dataset.K * dataset.K)
    arr = counts.reshape(dataset.K, dataset.K)
    plt.figure()
    # heatmap with continuous extent and blocky cells, 반투명 배경
    extent = [0, dataset.L, 0, dataset.L]
    plt.imshow(arr, cmap='gray', origin='lower', extent=extent,
               interpolation='nearest', alpha=0.5)
    plt.colorbar()
    plt.title('Training Data Distribution')
    # actual data points overlay (less 크고 투명하게)
    X = dataset.X.cpu().numpy()
    plt.scatter(X[:, 0], X[:, 1], c='red', s=1, alpha=0.3)
    plt.savefig(os.path.join(args.output_dir, 'train_distribution.png'))
    plt.close()

    # Test dataset (uniform)
    test_N0 = args.test_N0 if args.test_N0 is not None else args.N0
    test_dataset = GridMixtureDataset(args.K, test_N0, std=args.std, L=1.0)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    records = []
    for alpha in args.alphas:
        print(f"Running experiment for alpha = {alpha}")
        # choose hidden_channels: deeper network for grid if requested
        hidden = None
        if args.grid_model:
            # deeper MLP channels for grid data
            hidden = [64, 128, 256, 512, 256, 128, 64]
        model = LRVAE(alpha=alpha, dataset='pinwheel', hidden_channels=hidden)
        model.beta = args.beta
        model.alpha = alpha
        model.wu_alpha = 1.0

        train_model(model, loader, args.epochs, args.lr, device)

        # Local metrics on uniform test data
        reg_vals = compute_local_reg(model, test_loader, args.K)

        lips_vals = []
        for cell in range(args.K * args.K):
            X_cell = test_dataset.X[test_dataset.y == cell].to(device)
            lips = estimate_local_lipschitz(model.decode, X_cell, num_pairs=100)
            lips_vals.append(lips)

        # KL Reg heatmap (기존 컬러맵 유지)
        plot_heatmap(reg_vals, args.K, f"KL Reg (alpha={alpha})", os.path.join(args.output_dir, f"reg_alpha_{alpha}.png"))
        # Local Lipschitz heatmap (red 계열)
        plot_heatmap(np.array(lips_vals), args.K, f"Local Lipschitz (alpha={alpha})", os.path.join(args.output_dir, f"lips_alpha_{alpha}.png"), cmap='Reds')

        for cell, (r, l) in enumerate(zip(reg_vals, lips_vals)):
            records.append({'alpha': alpha, 'cell': cell, 'reg': float(r), 'lips': float(l)})

    df = pd.DataFrame(records)
    df.to_csv(os.path.join(args.output_dir, 'alpha_local_metrics.csv'), index=False)

if __name__ == '__main__':
    main() 