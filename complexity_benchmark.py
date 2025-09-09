import argparse
import os
import time
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from tqdm import tqdm
from datetime import datetime
from torchvision.utils import save_image, make_grid

import model as Model
import dataset as Dataset


# -------------------- Memory helpers --------------------
def get_memory_usage_mb():
    import resource
    # ru_maxrss is KB on Linux; convert to MB
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0

def get_gpu_memory_usage_mb():
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / (1024.0 ** 2)
    return 0.0


# -------------------- IO helpers --------------------
def save_model_weights(model: torch.nn.Module, output_dir: str, model_name: str):
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, f"{model_name}.pth")
    torch.save(model.state_dict(), path)


def sample_and_save_grids(model: torch.nn.Module, device: str, output_dir: str, model_name: str,
                          num_grids: int = 4, grid_n: int = 8):
    os.makedirs(output_dir, exist_ok=True)
    model.eval()

    latent_dim = getattr(model, 'latent_channel', 2)
    use_grad = type(model).__name__ == 'LIDVAE'

    for i in range(num_grids):
        z = torch.randn(grid_n * grid_n, latent_dim, device=device, requires_grad=use_grad)
        if use_grad:
            with torch.enable_grad():
                x = model.decode(z)
        else:
            with torch.no_grad():
                x = model.decode(z)
        # x can be (N, C, H, W) or flattened
        if x.dim() == 2:
            # assume MNIST 1x28x28
            if x.size(1) == 28 * 28:
                x = x.view(-1, 1, 28, 28)
            else:
                side = int(round(x.size(1) ** 0.5))
                x = x.view(-1, 1, side, side)
        x = x.clamp(0.0, 1.0)
        grid = make_grid(x, nrow=grid_n, padding=2)
        save_image(grid, os.path.join(output_dir, f"{model_name}_samples_grid_{i+1}.png"))


# -------------------- Train/Eval (from main.py rules) --------------------
def evaluate(model: Model.VAE, loader_test: DataLoader, device: str):
    model.eval()
    loss_total = 0.0
    loss_recon_total = 0.0
    loss_reg_total = 0.0
    loss_lr_total = 0.0

    # LIDVAE needs grad during decode
    use_grad = type(model).__name__ == 'LIDVAE'
    ctx = torch.enable_grad() if use_grad else torch.no_grad()

    with ctx:
        for x, y in tqdm(loader_test, leave=False, desc="Evaluate"):
            x = x.to(device)
            y = y.to(device)
            result = model(x)
            loss, loss_recon, loss_reg, loss_lr = model.loss(x, *result)
            loss_total += float(loss)
            loss_recon_total += float(loss_recon)
            loss_reg_total += float(loss_reg)
            loss_lr_total += float(loss_lr)

    n = max(1, len(loader_test))
    return (
        loss_total / n,
        loss_recon_total / n,
        loss_reg_total / n,
        loss_lr_total / n,
    )


def train_one_model(model: Model.VAE, loader_train: DataLoader, loader_test: DataLoader,
                    epochs: int, batch_size: int, device: str, num_mc_samples: int = 1):
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs * len(loader_train))

    # measure training
    train_mem_start = get_memory_usage_mb()
    train_gpu_mem_start = get_gpu_memory_usage_mb()
    t0 = time.time()

    for epoch in tqdm(range(epochs), desc=type(model).__name__):
        model.train()
        model.warmup(epoch, epochs)
        for x, y in tqdm(loader_train, leave=False, desc="Train"):
            x = x.to(device)
            y = y.to(device)

            result = model(x, L=num_mc_samples)
            loss, loss_recon, loss_reg, loss_lr = model.loss(x, *result)

            optimizer.zero_grad()
            # staged backward as in main.py
            loss_lr.backward(retain_graph=True)
            lam = 0.0001
            for p in model.encoder.parameters():
                if p.grad is not None:
                    p.grad *= lam
            loss_reg.backward(retain_graph=True)
            loss_recon.backward()
            optimizer.step()
            scheduler.step()

    train_time = time.time() - t0
    train_mem_used = max(0.0, get_memory_usage_mb() - train_mem_start)
    train_gpu_mem_used = max(0.0, get_gpu_memory_usage_mb() - train_gpu_mem_start)

    # measure evaluation
    eval_mem_start = get_memory_usage_mb()
    eval_gpu_mem_start = get_gpu_memory_usage_mb()
    t1 = time.time()
    eval_losses = evaluate(model, loader_test, device)
    eval_time = time.time() - t1
    eval_mem_used = max(0.0, get_memory_usage_mb() - eval_mem_start)
    eval_gpu_mem_used = max(0.0, get_gpu_memory_usage_mb() - eval_gpu_mem_start)

    return {
        'train_time_sec': train_time,
        'eval_time_sec': eval_time,
        'train_memory_mb': train_mem_used,
        'eval_memory_mb': eval_mem_used,
        'train_gpu_memory_mb': train_gpu_mem_used,
        'eval_gpu_memory_mb': eval_gpu_mem_used,
        'eval_losses': eval_losses,
    }


# -------------------- Benchmark Runner --------------------
def run_complexity_benchmark():
    parser = argparse.ArgumentParser(description="Complexity benchmark on MNIST (using model.py and main.py rules)")
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--output_dir', type=str, default='results/complexity_benchmark')
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--num_mc_samples', type=int, default=1)
    parser.add_argument('--alpha', type=float, default=0.1)
    parser.add_argument('--beta', type=float, default=1.0)
    parser.add_argument('--inverse_lipschitz', type=float, default=0.0)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Data: MNIST
    train_dataset, test_dataset = Dataset.load_dataset('mnist')
    loader_train = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=8,
        drop_last=True,
        pin_memory=True,
    )
    loader_test = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=8,
        drop_last=True,
        pin_memory=True,
    )

    device = args.device

    models_to_test = [
        ('VanillaVAE', lambda: Model.VanillaVAE(
            beta=args.beta,
            dataset='mnist',
            hidden_channels=None,
            encoder_type='conv',
            decoder_type='mlp',
            fixed_var=False,
            residual_connection=False
        )),
        ('LIDVAE', lambda: Model.LIDVAE(
            inverse_lipschitz=args.inverse_lipschitz,
            beta=args.beta,
            dataset='mnist',
            hidden_channels=None
        )),
        ('LRVAE', lambda: Model.LRVAE(
            beta=args.beta,
            alpha=args.alpha,
            z_source='Ex',
            dataset='mnist',
            hidden_channels=None,
            pwise_reg=False,
            encoder_type='conv',
            decoder_type='mlp',
            residual_connection=False
        )),
    ]

    results = []

    for model_name, factory in models_to_test:
        print(f"\n=== Testing {model_name} on MNIST ===")
        model = factory()

        # param count & size
        param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
        param_size = sum(p.nelement() * p.element_size() for p in model.parameters())
        buffer_size = sum(b.nelement() * b.element_size() for b in model.buffers())
        model_size_mb = (param_size + buffer_size) / (1024.0 ** 2)

        metrics = train_one_model(model, loader_train, loader_test, args.epochs, args.batch_size, device, args.num_mc_samples)

        # save weights and samples
        save_model_weights(model, os.path.join(args.output_dir, 'weights'), model_name)
        sample_and_save_grids(model, device, os.path.join(args.output_dir, 'samples'), model_name, num_grids=4, grid_n=8)

        results.append({
            'model': model_name,
            'parameters': param_count,
            'model_size_mb': model_size_mb,
            'train_time_sec': metrics['train_time_sec'],
            'eval_time_sec': metrics['eval_time_sec'],
            'train_memory_mb': metrics['train_memory_mb'],
            'eval_memory_mb': metrics['eval_memory_mb'],
            'train_gpu_memory_mb': metrics['train_gpu_memory_mb'],
            'eval_gpu_memory_mb': metrics['eval_gpu_memory_mb'],
            'alpha': args.alpha if model_name == 'LRVAE' else None,
            'beta': args.beta if model_name in ['VanillaVAE', 'LIDVAE', 'LRVAE'] else None,
            'inverse_lipschitz': args.inverse_lipschitz if model_name == 'LIDVAE' else None,
        })

    # Save CSV and log
    df = pd.DataFrame(results)
    df.to_csv(os.path.join(args.output_dir, 'complexity_results.csv'), index=False)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(args.output_dir, f'complexity_benchmark_log_{timestamp}.txt')
    with open(log_file, 'w') as f:
        f.write(f"Complexity Benchmark Results - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 80 + "\n")
        for r in results:
            f.write(str(r) + "\n")

    print(f"\nBenchmark complete. Results saved to {args.output_dir}")
    print(f"CSV: {os.path.join(args.output_dir, 'complexity_results.csv')}")


if __name__ == '__main__':
    run_complexity_benchmark()