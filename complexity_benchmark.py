import argparse
import os
import time
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
import torch.nn.functional as F
import platform
from datetime import datetime
import module  # ICNN 등 참고
from torchvision import datasets, transforms
from torchvision.utils import save_image, make_grid

# --- Constants ---
DEFAULT_EMPTY_CELL_FILL_VALUE = -5.0

# --- Memory Usage Functions ---
def get_memory_usage():
    """현재 메모리 사용량을 반환합니다."""
    import resource
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024  # MB 단위

def get_gpu_memory_usage():
    """GPU 메모리 사용량을 반환합니다."""
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / (1024**2)  # MB 단위
    return 0

# --- Common Encoder (모든 모델이 동일한 인코더 사용) ---
class CommonEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dims, latent_dim):
        super(CommonEncoder, self).__init__()
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.LeakyReLU(0.2)
            ])
            prev_dim = hidden_dim
        
        # Output layer (mu and log_var)
        layers.append(nn.Linear(prev_dim, latent_dim * 2))
        
        self.encoder = nn.Sequential(*layers)
    
    def forward(self, x):
        h = self.encoder(x)
        mu, log_var = h.chunk(2, dim=1)
        return mu, log_var

# --- Common Decoder (VanillaVAE용) ---
class CommonDecoder(nn.Module):
    def __init__(self, latent_dim, hidden_dims, output_dim):
        super(CommonDecoder, self).__init__()
        
        layers = []
        prev_dim = latent_dim
        
        for hidden_dim in reversed(hidden_dims):
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.LeakyReLU(0.2)
            ])
            prev_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(prev_dim, output_dim))
        layers.append(nn.Sigmoid())
        
        self.decoder = nn.Sequential(*layers)
    
    def forward(self, z):
        return self.decoder(z)

# --- ICNN Decoder (LIDVAE용) ---
class ICNNDecoder(nn.Module):
    def __init__(self, latent_dim, hidden_dims, output_dim, inverse_lipschitz=0.0):
        super(ICNNDecoder, self).__init__()
        # model.py의 LIDVAE 방식 참고: 두 개의 ICNN과 브레니에 맵의 gradient를 사용
        self.icnn1 = module.ICNN(latent_dim, hidden_channel=hidden_dims[-1], num_layers=2)
        self.icnn2 = module.ICNN(output_dim, hidden_channel=hidden_dims[-1], num_layers=2)
        self.register_buffer("B", torch.eye(output_dim, latent_dim, requires_grad=False))
        self.il_factor = inverse_lipschitz / 2.0

    def forward(self, z):
        z = z.requires_grad_(True)
        # 첫 번째 ICNN + inverse-lipschitz 제곱항
        x_potential = self.icnn1(z) + self.il_factor * z.pow(2).sum(1, keepdim=True)
        # 브레니에 맵: gradient_z phi(z)
        x = torch.autograd.grad(x_potential, [z], torch.ones_like(x_potential), create_graph=True)[0]
        # Beta 선형변환 (여기서는 항등행렬)
        x = torch.nn.functional.linear(x, self.B)
        x = x.requires_grad_(True)
        # 두 번째 ICNN + inverse-lipschitz 제곱항
        y_potential = self.icnn2(x) + self.il_factor * x.pow(2).sum(1, keepdim=True)
        # 최종 브레니에 맵: gradient_x psi(x)
        y = torch.autograd.grad(y_potential, [x], torch.ones_like(y_potential), create_graph=True)[0]
        return y

# --- Base VAE Class ---
class BaseVAE(nn.Module):
    def __init__(self, input_dim, latent_dim, hidden_dims, beta=1.0):
        super(BaseVAE, self).__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.hidden_dims = hidden_dims
        self.beta = beta
        
        self.encoder = CommonEncoder(input_dim, hidden_dims, latent_dim)
        self.encoder_post = CommonEncoder(input_dim, hidden_dims, latent_dim)  # LRVAE latent recon 용

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def forward(self, x):
        mu, log_var = self.encoder(x)
        z = self.reparameterize(mu, log_var)
        recon = self.decoder(z)
        # LRVAE를 위해 재인코딩 결과도 함께 반환 (다른 모델은 사용 안 함)
        mu_re, log_var_re = self.encoder_post(recon.detach())
        z_recon = self.reparameterize(mu_re, log_var_re)
        return recon, mu, log_var, z, z_recon
    
    def loss(self, x, recon, mu, log_var, z, z_recon):
        # Reconstruction loss
        recon_loss = F.mse_loss(recon, x, reduction='sum')
        
        # KL divergence
        kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
        
        total_loss = recon_loss + self.beta * kl_loss
        return total_loss, recon_loss, kl_loss, 0.0

# --- VanillaVAE (Beta-VAE) ---
class VanillaVAE(BaseVAE):
    def __init__(self, input_dim, latent_dim, hidden_dims, beta=1.0):
        super(VanillaVAE, self).__init__(input_dim, latent_dim, hidden_dims, beta)
        self.decoder = CommonDecoder(latent_dim, hidden_dims, input_dim)

# --- LIDVAE ---
class LIDVAE(BaseVAE):
    def __init__(self, input_dim, latent_dim, hidden_dims, beta=1.0, inverse_lipschitz=0.0):
        super(LIDVAE, self).__init__(input_dim, latent_dim, hidden_dims, beta)
        self.inverse_lipschitz = inverse_lipschitz
        self.decoder = ICNNDecoder(latent_dim, hidden_dims, input_dim, inverse_lipschitz=inverse_lipschitz)
    
    def loss(self, x, recon, mu, log_var, z, z_recon):
        # Reconstruction loss
        recon_loss = F.mse_loss(recon, x, reduction='sum')
        
        # KL divergence
        kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
        
        # Inverse Lipschitz regularization: ||x||^2, ||y||^2에 대응하여 이미 디코더에 포함되었음
        lipschitz_loss = 0.0
        
        total_loss = recon_loss + self.beta * kl_loss + lipschitz_loss
        return total_loss, recon_loss, kl_loss, lipschitz_loss

# --- LRVAE ---
class LRVAE(BaseVAE):
    def __init__(self, input_dim, latent_dim, hidden_dims, alpha=0.1, beta=1.0):
        super(LRVAE, self).__init__(input_dim, latent_dim, hidden_dims, beta)
        self.alpha = alpha
        self.decoder = CommonDecoder(latent_dim, hidden_dims, input_dim)
    
    def loss(self, x, recon, mu, log_var, z, z_recon):
        # Reconstruction loss
        recon_loss = F.mse_loss(recon, x, reduction='sum')
        
        # KL divergence
        kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
        
        # Latent Reconstruction regularization (LRVAE): ||z - z_recon||^2
        local_reg = 0.0
        if self.alpha > 0:
            local_reg = self.alpha * ((z - z_recon).pow(2).sum(1)).mean()
        
        total_loss = recon_loss + self.beta * kl_loss + local_reg
        return total_loss, recon_loss, kl_loss, local_reg

# --- Model Training Functions ---
def train_model_with_timing(model, loader, epochs, lr, device, model_name):
    """모델 훈련 시간과 메모리를 측정합니다."""
    model.to(device).train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    # 메모리 측정 시작
    start_memory = get_memory_usage()
    start_gpu_memory = get_gpu_memory_usage()
    
    # 훈련 시간 측정
    start_time = time.time()
    
    for epoch in tqdm(range(epochs), desc=f"Training {model_name}"):
        for X, _ in loader:
            X = X.to(device)
            optimizer.zero_grad()
            
            # 모든 모델이 동일한 forward 시그니처 사용
            recon, mu, log_var, z, z_recon = model(X)
            total_loss, _, _, _ = model.loss(X, recon, mu, log_var, z, z_recon)
            
            total_loss.backward()
            optimizer.step()
    
    end_time = time.time()
    end_memory = get_memory_usage()
    end_gpu_memory = get_gpu_memory_usage()
    
    training_time = end_time - start_time
    memory_used = end_memory - start_memory
    gpu_memory_used = end_gpu_memory - start_gpu_memory
    
    return training_time, memory_used, gpu_memory_used

def evaluate_model_complexity(model, test_loader, device, model_name):
    """모델 평가 시간과 메모리를 측정합니다."""
    model.eval()
    
    # 메모리 측정 시작
    start_memory = get_memory_usage()
    start_gpu_memory = get_gpu_memory_usage()
    
    start_time = time.time()
    
    # LIDVAE는 디코더 내부에서 autograd.grad를 사용하므로 no_grad 금지
    if model_name == 'LIDVAE':
        ctx = torch.enable_grad()
    else:
        ctx = torch.no_grad()

    with ctx:
        for X, _ in test_loader:
            X = X.to(device)
            recon, mu, log_var, z, z_recon = model(X)
    
    end_time = time.time()
    end_memory = get_memory_usage()
    end_gpu_memory = get_gpu_memory_usage()
    
    eval_time = end_time - start_time
    memory_used = end_memory - start_memory
    gpu_memory_used = end_gpu_memory - start_gpu_memory
    
    return eval_time, memory_used, gpu_memory_used

def count_parameters(model):
    """모델의 파라미터 수를 계산합니다."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def get_model_size_mb(model):
    """모델 크기를 MB 단위로 계산합니다."""
    param_size = 0
    buffer_size = 0
    
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    
    size_all_mb = (param_size + buffer_size) / (1024**2)
    return size_all_mb

# --- Sampling & Saving Utilities ---
CIFAR_MEAN = torch.tensor([0.4914, 0.4822, 0.4465]).view(1, 3, 1, 1)
CIFAR_STD = torch.tensor([0.2470, 0.2435, 0.2616]).view(1, 3, 1, 1)

def denormalize_cifar(x):
    # x: (N, 3, 32, 32) in normalized space
    return (x * CIFAR_STD.to(x.device)) + CIFAR_MEAN.to(x.device)

def sample_and_save_grids(model, device, output_dir, model_name, latent_dim, num_grids=4, grid_n=8):
    os.makedirs(output_dir, exist_ok=True)
    model.eval()
    for i in range(num_grids):
        z = torch.randn(grid_n * grid_n, latent_dim, device=device)
        # LIDVAE 디코더는 autograd.grad를 사용하므로 grad 활성화 필요
        with torch.enable_grad():
            imgs_vec = model.decoder(z)
        imgs = imgs_vec.view(-1, 3, 32, 32)
        imgs = denormalize_cifar(imgs)
        imgs = imgs.clamp(0.0, 1.0)
        grid = make_grid(imgs, nrow=grid_n, padding=2)
        save_path = os.path.join(output_dir, f"{model_name}_samples_grid_{i+1}.png")
        save_image(grid, save_path)

def save_model_weights(model, output_dir, model_name):
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, f"{model_name}.pth")
    torch.save(model.state_dict(), save_path)

# --- Main Experiment Function ---
def run_complexity_benchmark():
    parser = argparse.ArgumentParser(description="Complexity benchmark for VAE models.")
    
    # Common parameters
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs.')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate.')
    parser.add_argument('--batch_size', type=int, default=256, help='Batch size.')
    parser.add_argument('--device', type=str, default='cpu', help='Device to run on (cuda or cpu).')
    parser.add_argument('--output_dir', type=str, default='results/complexity_benchmark', help='Output directory.')
    
    # Data parameters
    parser.add_argument('--train_samples', type=int, default=50000, help='Number of training samples to use (<=50000).')
    parser.add_argument('--test_samples', type=int, default=10000, help='Number of test samples to use (<=10000).')
    parser.add_argument('--input_dim', type=int, default=3072, help='Input dimension (CIFAR-10: 32x32x3=3072).')
    parser.add_argument('--latent_dim', type=int, default=2, help='Latent dimension.')
    
    # Model parameters
    parser.add_argument('--hidden_dims', nargs='+', type=int, default=[512, 256, 128], help='Hidden dimensions.')
    parser.add_argument('--alpha', type=float, default=0.1, help='Alpha for LRVAE.')
    parser.add_argument('--beta', type=float, default=1.0, help='Beta for beta-VAE.')
    parser.add_argument('--inverse_lipschitz', type=float, default=0.0, help='Inverse Lipschitz for LIDVAE.')
    
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # CIFAR-10 실제 데이터 로딩
    print("Loading CIFAR-10 dataset...")
    np.random.seed(42)
    torch.manual_seed(42)

    transform = transforms.Compose([
        transforms.ToTensor(),  # [0,1]
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
        transforms.Lambda(lambda x: x.view(-1))  # flatten to 3072
    ])

    train_full = datasets.CIFAR10(root=os.path.join(args.output_dir, 'data'), train=True, download=True, transform=transform)
    test_full = datasets.CIFAR10(root=os.path.join(args.output_dir, 'data'), train=False, download=True, transform=transform)

    # Subset if user requests fewer samples
    if args.train_samples < len(train_full):
        train_indices = list(range(args.train_samples))
        train_dataset = Subset(train_full, train_indices)
    else:
        train_dataset = train_full

    if args.test_samples < len(test_full):
        test_indices = list(range(args.test_samples))
        test_dataset = Subset(test_full, test_indices)
    else:
        test_dataset = test_full

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True)
    
    # 실험 결과 저장
    results = []
    log_entries = []
    
    # 공통 조건 로그
    common_conditions = f"""
=== COMMON CONDITIONS ===
Epochs: {args.epochs}
Learning Rate: {args.lr}
Batch Size: {args.batch_size}
Device: {args.device}
Training Samples: {args.train_samples}
Test Samples: {args.test_samples}
Input Dimension: {args.input_dim} (CIFAR-10-like)
Latent Dimension: {args.latent_dim}
Hidden Dimensions: {args.hidden_dims}
Python Version: {platform.python_version()}
PyTorch Version: {torch.__version__}
"""
    log_entries.append(common_conditions)
    
    # 모델별 실험 - 모든 모델이 동일한 인코더 사용
    models_to_test = [
        ('VanillaVAE', lambda: VanillaVAE(input_dim=args.input_dim, latent_dim=args.latent_dim, 
                                         hidden_dims=args.hidden_dims, beta=args.beta)),
        ('LIDVAE', lambda: LIDVAE(input_dim=args.input_dim, latent_dim=args.latent_dim, 
                                 hidden_dims=args.hidden_dims, beta=args.beta, inverse_lipschitz=args.inverse_lipschitz)),
        ('LRVAE', lambda: LRVAE(input_dim=args.input_dim, latent_dim=args.latent_dim, 
                               hidden_dims=args.hidden_dims, alpha=args.alpha, beta=args.beta))
    ]
    
    for model_name, model_factory in models_to_test:
        print(f"\n=== Testing {model_name} ===")
        
        try:
            # 모델 생성
            model = model_factory()
            
            # 모델 정보
            param_count = count_parameters(model)
            model_size_mb = get_model_size_mb(model)
            
            print(f"Parameters: {param_count:,}")
            print(f"Model Size: {model_size_mb:.2f} MB")
            
            # 훈련 시간 및 메모리 측정
            print("Training...")
            train_time, train_memory, train_gpu_memory = train_model_with_timing(
                model, train_loader, args.epochs, args.lr, args.device, model_name
            )
            
            # 평가 시간 및 메모리 측정
            print("Evaluating...")
            eval_time, eval_memory, eval_gpu_memory = evaluate_model_complexity(
                model, test_loader, args.device, model_name
            )

            # 학습 완료 후 가중치 저장
            weights_dir = os.path.join(args.output_dir, 'weights')
            save_model_weights(model, weights_dir, model_name)

            # 샘플 이미지 저장 (64개, 8x8) 4장
            samples_dir = os.path.join(args.output_dir, 'samples')
            sample_and_save_grids(model, args.device, samples_dir, model_name, args.latent_dim, num_grids=4, grid_n=8)
            
            # 결과 저장
            result = {
                'model': model_name,
                'parameters': param_count,
                'model_size_mb': model_size_mb,
                'train_time_sec': train_time,
                'eval_time_sec': eval_time,
                'train_memory_mb': train_memory,
                'eval_memory_mb': eval_memory,
                'train_gpu_memory_mb': train_gpu_memory,
                'eval_gpu_memory_mb': eval_gpu_memory,
                'alpha': args.alpha if model_name == 'LRVAE' else None,
                'beta': args.beta if model_name in ['VanillaVAE', 'LIDVAE'] else None,
                'inverse_lipschitz': args.inverse_lipschitz if model_name == 'LIDVAE' else None
            }
            results.append(result)
            
            # 로그 생성
            model_log = f"""
=== {model_name} RESULTS ===
Parameters: {param_count:,}
Model Size: {model_size_mb:.2f} MB
Training Time: {train_time:.2f} seconds
Evaluation Time: {eval_time:.2f} seconds
Training Memory Usage: {train_memory:.2f} MB
Evaluation Memory Usage: {eval_memory:.2f} MB
Training GPU Memory Usage: {train_gpu_memory:.2f} MB
Evaluation GPU Memory Usage: {eval_gpu_memory:.2f} MB
"""
            if model_name == 'LRVAE':
                model_log += f"Alpha: {args.alpha}\n"
            if model_name in ['VanillaVAE', 'LIDVAE']:
                model_log += f"Beta: {args.beta}\n"
            if model_name == 'LIDVAE':
                model_log += f"Inverse Lipschitz: {args.inverse_lipschitz}\n"
            
            log_entries.append(model_log)
            
        except Exception as e:
            error_log = f"""
=== {model_name} ERROR ===
Error: {str(e)}
"""
            log_entries.append(error_log)
            print(f"Error testing {model_name}: {e}")
    
    # 결과 저장
    df = pd.DataFrame(results)
    df.to_csv(os.path.join(args.output_dir, 'complexity_results.csv'), index=False)
    
    # 로그 파일 저장
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(args.output_dir, f'complexity_benchmark_log_{timestamp}.txt')
    
    with open(log_file, 'w') as f:
        f.write(f"Complexity Benchmark Results - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 80 + "\n")
        for entry in log_entries:
            f.write(entry)
            f.write("\n")
    
    print(f"\nBenchmark complete. Results saved to {args.output_dir}")
    print(f"Log file: {log_file}")
    
    # 요약 출력
    print("\n=== SUMMARY ===")
    for result in results:
        print(f"{result['model']}: {result['train_time_sec']:.2f}s train, {result['eval_time_sec']:.2f}s eval, "
              f"{result['parameters']:,} params, {result['model_size_mb']:.2f} MB")

if __name__ == '__main__':
    run_complexity_benchmark()