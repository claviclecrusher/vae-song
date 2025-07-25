import argparse
import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from tqdm import tqdm
from dataset import GridMixtureDataset
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
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    dataset = GridMixtureDataset(args.K, args.N0, std=args.std, L=1.0)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    records = []
    for alpha in args.alphas:
        print(f"Running experiment for alpha = {alpha}")
        model = LRVAE(alpha=alpha, dataset='pinwheel', hidden_channels=None)
        model.beta = args.beta
        model.alpha = alpha
        model.wu_alpha = 1.0

        train_model(model, loader, args.epochs, args.lr, device)

        test_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
        reg_vals = compute_local_reg(model, test_loader, args.K)

        lips_vals = []
        for cell in range(args.K * args.K):
            X_cell = dataset.X[dataset.y == cell].to(device)
            lips = estimate_local_lipschitz(model.decode, X_cell, num_pairs=100)
            lips_vals.append(lips)

        plot_heatmap(reg_vals, args.K, f"KL Reg (alpha={alpha})", os.path.join(args.output_dir, f"reg_alpha_{alpha}.png"))
        plot_heatmap(np.array(lips_vals), args.K, f"Local Lipschitz (alpha={alpha})", os.path.join(args.output_dir, f"lips_alpha_{alpha}.png"))

        for cell, (r, l) in enumerate(zip(reg_vals, lips_vals)):
            records.append({'alpha': alpha, 'cell': cell, 'reg': float(r), 'lips': float(l)})

    df = pd.DataFrame(records)
    df.to_csv(os.path.join(args.output_dir, 'alpha_local_metrics.csv'), index=False)

if __name__ == '__main__':
    main() 