import torch
import torchvision
import os
import csv
import math
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader
from sklearn.manifold import TSNE

def reparameterize(mu, logvar, nsamples=1):
    """(Wang et al.) sample from posterior Gaussian family"""
    batch_size, nz = mu.size()
    std = logvar.mul(0.5).exp()
    mu_expd = mu.unsqueeze(1).expand(batch_size, nsamples, nz)
    std_expd = std.unsqueeze(1).expand(batch_size, nsamples, nz)
    eps = torch.zeros_like(std_expd).normal_()
    return mu_expd + torch.mul(eps, std_expd) # (batch, nsamples, nz)

def calc_au_per_batch(z, eps=0.01):
    return (torch.mean((z-z.mean(dim=0, keepdim=True))**2, dim=0) >= eps).type(torch.FloatTensor).mean().detach().item()

def calc_au(model, loader, device, delta=0.01): # from wang et al.
    cnt = 0
    for data in loader:
        mean, _ = model.encode(data[0].to(device))
        if cnt == 0:
            means_sum = mean.sum(dim=0, keepdim=True)
        else:
            means_sum = means_sum + mean.sum(dim=0, keepdim=True)
        cnt += mean.size(0)
    mean_mean = means_sum / cnt
    cnt = 0
    for data in loader:
        mean, _ = model.encode(data[0].to(device))
        if cnt == 0:
            var_sum = ((mean - mean_mean) ** 2).sum(dim=0)
        else:
            var_sum = var_sum + ((mean - mean_mean) ** 2).sum(dim=0)
        cnt += mean.size(0)
    au_var = var_sum / (cnt - 1)
    return (au_var >= delta).sum().item()/mean.size(1), au_var

def log_sum_exp(value, dim=None, keepdim=False):
    """(Wang et al.) Numerically stable implementation of the operation
    value.exp().sum(dim, keepdim).log()    """
    if dim is not None:
        m, _ = torch.max(value, dim=dim, keepdim=True)
        value0 = value - m
        if keepdim is False:
            m = m.squeeze(dim)
        return m + torch.log(torch.sum(torch.exp(value0), dim=dim, keepdim=keepdim))
    else:
        m = torch.max(value)
        sum_exp = torch.sum(torch.exp(value - m))
        return m + torch.log(sum_exp)
        
def calc_mi(mu, logvar):
    """(Wang et al.) Approximate the mutual information between x and z
    I(x, z) = E_xE_{q(z|x)}log(q(z|x)) - E_xE_{q(z|x)}log(q(z))
    Returns: Float """
    x_batch, nz = mu.size()
    # E_{q(z|x)}log(q(z|x)) = -0.5*nz*log(2*\pi) - 0.5*(1+logvar).sum(-1)
    neg_entropy = (-0.5 * nz * math.log(2 * math.pi)- 0.5 * (1 + logvar).sum(-1)).mean()
    # [z_batch, 1, nz]
    z_samples = reparameterize(mu, logvar, 1)
    # [1, x_batch, nz]
    mu, logvar = mu.unsqueeze(0), logvar.unsqueeze(0)
    var = logvar.exp()
    # (z_batch, x_batch, nz)
    dev = z_samples - mu
    # (z_batch, x_batch)
    log_density = -0.5 * ((dev ** 2) / var).sum(dim=-1) - \
        0.5 * (nz * math.log(2 * math.pi) + logvar.sum(-1))
    # log q(z): aggregate posterior
    # [z_batch]
    log_qz = log_sum_exp(log_density, dim=1) - math.log(x_batch)
    return (neg_entropy - log_qz.mean(-1)).item()

def nll_iw(mu, log_var, loss_rec, nsamples=100):
    """(Wang et al.) compute the importance weighting estimate of the log-likelihood"""
    # [batch, ns, nz]
    # param is the parameters required to evaluate q(z|x)
    z = reparameterize(mu, log_var, nsamples)
    #z, param = self.encoder.sample(x, ns) # param = [mu, logvar]
    # [batch, ns]
    log_comp_ll = log_prob(z.device, z.size()).log_prob(z).sum(dim=-1) - loss_rec # log p(z,x) = log p(z) + log p(x|z) = log p(x|z)p(z)
    log_infer_ll = eval_inference_dist(mu, log_var, z) # log q(z|x)
    tmp= (log_comp_ll - log_infer_ll)
    ll_iw = log_sum_exp(tmp) - math.log(nsamples) # p(x) = log p(z,x) - log q(z|x) - log(ns) = log p(z|x)p(x)/q(z|x)(ns)
    return -ll_iw.detach().item()

def log_prob(device, nz):
    loc = torch.zeros(nz, device=device)
    scale = torch.ones(nz, device=device)
    return torch.distributions.normal.Normal(loc, scale)

def eval_inference_dist(mu, logvar, z):
    """this function computes log q(z | x)"""
    nz = z.size(2)
    # (batch_size, 1, nz)
    mu, logvar = mu.unsqueeze(1), logvar.unsqueeze(1)
    var = logvar.exp()
    # (batch_size, nsamples, nz)
    dev = z - mu
    # (batch_size, nsamples)
    log_density = -0.5 * ((dev ** 2) / var).sum(dim=-1) - \
        0.5 * (nz * math.log(2 * math.pi) + logvar.sum(-1))
    return log_density

def kld(mu, log_var):
    return (-0.5 * (1 + log_var - mu**2 - log_var.exp())).mean(dim=0).sum().detach().item()


def measure_pc_runmodel(model, loader, device):
    au_sum = 0
    kl_sum = 0
    mi_sum = 0
    nll_sum = 0
    var_sum = 0
    for i, data in enumerate(loader):
        res_list = model(data[0].to(device)) 
        recon  = res_list[0]
        mu  = res_list[1]
        log_var = res_list[2]
        z_input = res_list[3] if len(res_list)>3 else None
        z_recon = res_list[4] if len(res_list)>4 else None
        loss, loss_rec, loss_reg, loss_lr = model.loss(data[0].to(device), recon, mu, log_var, z_input=z_input, z_recon=z_recon)
        au_sum += calc_au_per_batch(mu) # unstable result
        kl_sum += kld(mu, log_var)
        mi_sum += calc_mi(mu, log_var)
        nll_sum += nll_iw(mu, log_var, loss_rec)
        if torch.is_tensor(log_var): # except naive ae
            var_sum += log_var.exp().sum().detach().item()
    #au, au_var = calc_au(model, loader, device, delta=0.01)
    return au_sum/(i+1), kl_sum/(i+1), mi_sum/(i+1), nll_sum/(i+1), var_sum/(i+1)



def log_unified(path, list_elements, list_names, logfilename='unified_log.csv'):
    os.makedirs(path, exist_ok=True)
    if not os.path.isfile(os.path.join(path, logfilename)):
        with open(os.path.join(path, logfilename), mode='a') as file:
            writer = csv.writer(file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            writer.writerow(list_names)
    with open(os.path.join(path, logfilename), mode='a') as file:
        writer = csv.writer(file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        writer.writerow(list_elements)

def log_unified_dict(path, dict_elements, logfilename='unified_log.csv'):
    os.makedirs(path, exist_ok=True)
    if not os.path.isfile(os.path.join(path, logfilename)):
        with open(os.path.join(path, logfilename), mode='a') as file:
            writer = csv.writer(file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            writer.writerow(dict_elements.keys())
    with open(os.path.join(path, logfilename), mode='a') as file:
        writer = csv.writer(file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        writer.writerow(dict_elements.values())


def logscale_plt_color_map(original_cmap_name):
    """Create a new colormap with log scale"""
    origin = matplotlib.cm.get_cmap(original_cmap_name, 256)
    newcolors = origin(np.logspace(0, 1, 256) / 10)
    return matplotlib.colors.ListedColormap(newcolors)


def pca_calculation(x):
    # Compute covariance matrix
    x_mean = np.mean(x, axis=0)
    x_centered = x - x_mean
    cov_matrix = np.dot(x_centered.T, x_centered) / (x_centered.shape[0] - 1)

    # Eigen decomposition
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

    # Sort eigenvalues and eigenvectors
    sorted_indices = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[sorted_indices]
    eigenvectors = eigenvectors[:, sorted_indices]

    x_pca = np.dot(x_centered, eigenvectors)
    x_pca_min, x_pca_max = x_pca.min(), x_pca.max()
    x_min, x_max = x.min(), x.max()
    #print(f" var min: {v_min}, var max: {v_max}, var mean: {var.mean()}")
    return x_pca, x_pca_min, x_pca_max, x_min, x_max


def pca_plot(x, x_pca, x_pca_min, x_pca_max, x_min, x_max, v_min, v_max, y, epoch, resultname, name, variablename='?', var=0.0, cmapc='viridis'):
    # Plot 1D scatter for each principal component
    MAX_1D_PLOT_ITER = 32
    zero_array = np.zeros_like(x_pca[:, 0])
    num_components = min(x_pca.shape[1], MAX_1D_PLOT_ITER)
    fig, axes = plt.subplots(num_components, 1, figsize=(15, 10), sharex=True)
    plt.yticks([])
    for i in range(num_components):
        axes[i].scatter(x_pca[:, i], zero_array, c=var[:,i], cmap=cmapc, vmin=0, vmax=1.0, marker='|')
        axes[i].get_yaxis().set_visible(False)  # Hide the y-axis
        axes[i].set_xlim([x_pca_min, x_pca_max])
    plt.savefig(f"./results/{resultname}/{name}/pca/{epoch}_pca_all_{variablename}.png")
    plt.close()

    # Plot 1D scatter for each actual channel
    MAX_1D_PLOT_ITER = 32
    zero_array = np.zeros_like(x[:, 0])
    num_components = min(x.shape[1], MAX_1D_PLOT_ITER)
    fig, axes = plt.subplots(num_components, 1, figsize=(15, 10), sharex=True)
    plt.yticks([])
    for i in range(num_components):
        #axes[i].scatter(mu[:, i], zero_array, c=var[:,i], cmap=logscale_plt_color_map('viridis'), vmin=0, vmax=1.0)
        axes[i].scatter(x[:, i], zero_array, c=var[:,i], cmap=cmapc, vmin=v_min, vmax=v_max, marker='|')
        axes[i].get_yaxis().set_visible(False)  # Hide the y-axis
        axes[i].set_xlim([x_min, x_max])
    #plt.colorbar(axes, label='Average Variance')
    plt.savefig(f"./results/{resultname}/{name}/pca/{epoch}_channels_all_{variablename}.png")
    plt.close()

    # Plot 2D scatter for the first two principal components
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(x_pca[:, 0], x_pca[:, 1], c=var.max(1), cmap=cmapc, vmin=v_min, vmax=v_max)
    plt.colorbar(scatter, label='Maximum Variance')
    #plt.xlim([-4, 4])
    #plt.ylim([-4, 4])
    plt.savefig(f"./results/{resultname}/{name}/pca/{epoch}_pca_v_{variablename}.png")
    plt.close()

    try:
        tsne = TSNE(n_components=2, random_state=0)
        mu_tsne = tsne.fit_transform(x)
        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(mu_tsne[:, 0], mu_tsne[:, 1], c=y, cmap='tab10')
        plt.colorbar(scatter, label='Class')
        plt.xlim([-50, 50])
        plt.ylim([-50, 50])
        plt.savefig(f"./results/{resultname}/{name}/pca/{epoch}_tsne_c.png")
        plt.close()
    except Exception as e:
        print(f"Error in tsne: {e}")
        exit()

    return


def pca_visualization(model, loader_test, device, epoch, name, resultname):

    os.makedirs("./results/"+resultname+"/" + name + "/pca", exist_ok=True)
    model.eval()
    loader_test = DataLoader(
        loader_test.dataset,
        batch_size=1000,
        shuffle=False,
        num_workers=8,
        drop_last=True,
        pin_memory=True,
    )
    x, y = next(iter(loader_test))
    x = x.to(device)
    #result = model(x)
    mu, var = model.encode(x)
    z = reparameterize(mu, var).squeeze()

    mu = mu.cpu().detach().numpy()
    z = z.cpu().detach().numpy()
    if torch.is_tensor(var):
        var = var.cpu().detach().numpy()
    else:
        var = np.zeros_like(mu)

    v_min, v_max = var.min(), var.max()

    mu_pca, mu_pca_min, mu_pca_max, mu_min, mu_max = pca_calculation(mu)
    z_pca, z_pca_min, z_pca_max, z_min, z_max = pca_calculation(z)


    # Plot 2D scatter for prior distribution
    if epoch == 0:
        zpz = np.random.randn(*mu.shape)
        zpz_pca, _, _, _, _ = pca_calculation(zpz)
        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(zpz_pca[:, 0], zpz_pca[:, 1], c=var.mean(1), cmap='coolwarm', vmin=0, vmax=1.0)
        #plt.colorbar(scatter, label='Average Variance')
        #plt.xlim([-4, 4])
        #plt.ylim([-4, 4])
        plt.savefig(f"./results/{resultname}/{name}/pca/prior.png")
        plt.close()

    pca_plot(mu, mu_pca, mu_pca_min, mu_pca_max, mu_min, mu_max, v_min, v_max, y, epoch, resultname, name, variablename='mu', var=var)
    pca_plot(z, z_pca, z_pca_min, z_pca_max, z_min, z_max, v_min, v_max, y, epoch, resultname, name, variablename='z', var=np.zeros_like(mu), cmapc='coolwarm')

    return


# ./results 아래에 있는 모든 결과들을 불러와서, 각각의 reconstruction loss와 latent reconstruction loss를 scatter plot으로 그려주는 함수
# 각각의 결과는 다른 색으로 표시되며, 각각의 결과는 그 결과가 나온 디렉토리 이름으로 표시됨
# dataset_name은 mnist, celeba, fashionmnist 중 하나를 입력받음
# ./results 하위 디렉토리 이름 가장 마지막은 _dataset_name으로 끝나야 함
def rec_lr_scatter_visualization(models, dataset_name, device):
    l_rec = []
    l_lr = []
    colors = []
    labels = []
    color_labels = []

    if dataset_name == 'mnist':
        transforms = torchvision.transforms.Compose(
            [
                torchvision.transforms.RandomRotation(20),
                torchvision.transforms.RandomResizedCrop((28, 28), (0.9, 1), (0.9, 1.1)),
                torchvision.transforms.ToTensor(),
            ]
        )
        test_dataset = torchvision.datasets.MNIST(root="dataset/", transform=transforms, train=False)
    elif dataset_name == 'celeba':
        transforms = torchvision.transforms.Compose(
            [
                torchvision.transforms.RandomHorizontalFlip(),
                torchvision.transforms.CenterCrop(148),
                torchvision.transforms.Resize(64),
                torchvision.transforms.ToTensor(),
            ]
        )
        test_dataset = torchvision.datasets.CelebA(root="dataset/", transform=transforms, split="test")
    elif dataset_name == 'fashionmnist':
        transforms = torchvision.transforms.Compose(
            [
                torchvision.transforms.Resize(28),
                torchvision.transforms.ToTensor(),
            ]
        )
        test_dataset = torchvision.datasets.FashionMNIST(root="dataset/", transform=transforms, train=False)
    else:
        print(dataset_name, "is not implemented")
        exit()

    count_points = 0
    for root, dirs, files in os.walk("./results/"):
        for file in files:
            if file == "model_99.pt" and root.startswith("./results/result_") and root.find(dataset_name) != -1:
                model_path = os.path.join(root, file)
                key = root.split('/')[2].split('_')[1]
                try:
                    model = models[key]
                except Exception as e:
                    print(f"Error finding key: {e}")
                    continue
                try:
                    model.load_state_dict(torch.load(model_path, map_location=device))
                except Exception as e:
                    print(f"Error loading model: {e}")
                    continue
                model.to(device).eval()
                
                test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False, num_workers=8, drop_last=True, pin_memory=True)
                x, _ = next(iter(test_loader))
                x = x.to(device)
                
                res_list = model(x, latent_recon=True) # to get lr loss for test
                recon = res_list[0]
                mu = res_list[1]
                log_var = res_list[2]
                z_input = res_list[3] if len(res_list) > 3 else None
                z_recon = res_list[4] if len(res_list) > 4 else None
                loss_rec = ((x - recon) ** 2).mean(dim=0).sum()
                loss_lr = ((z_input - z_recon) ** 2).mean(dim=0).sum()

                if isinstance(loss_rec, torch.Tensor):
                    loss_rec = loss_rec.cpu().detach()
                if isinstance(loss_lr, torch.Tensor):
                    loss_lr = loss_lr.cpu().detach()
                
                l_rec.append(loss_rec)
                l_lr.append(loss_lr)
                labels.append(root.split('/')[3])

                if root.split('/')[3].split(' ')[0] not in color_labels:
                    color_labels.append(root.split('/')[3].split(' ')[0])

                color_index = len(color_labels) - 1
                colors.append(color_index)
                
                count_points += 1

    plt.figure(figsize=(10, 8))
    plt.title('Reconstruction Loss vs Latent Reconstruction Loss:'+ dataset_name)
    scatter = plt.scatter(l_lr, l_rec, c=colors, cmap='tab10')
    for i, label_name in enumerate(labels):
        plt.annotate(label_name, (l_lr[i], l_rec[i]), fontsize=8, alpha=0.7, rotation=0)
    plt.yscale('log')
    plt.xscale('log')
    plt.xlabel('Latent Reconstruction Loss')
    plt.ylabel('Reconstruction Loss')
    
    os.makedirs("./results/rec_scatter", exist_ok=True)
    plt.savefig("./results/rec_scatter/loss_scatter_plot.png")
    plt.close()

    print(count_points, "points plotted")


def visualize_2c_points_on_image(tensor, label, resultname, name, epoch, tensor_name='recon'):
    # 텐서를 numpy 배열로 변환
    if torch.is_tensor(tensor):
        tensor = tensor.cpu().detach().numpy()
        label = label.cpu().detach().numpy()
    
    # 텐서의 크기를 확인
    assert tensor.shape[1] == 2, f"Tensor must have shape [N, 2] but given shape is {tensor.shape}"
    
    # 점을 시각화
    plt.figure(figsize=(8, 8))
    plt.scatter(tensor[:, 0], tensor[:, 1], c=label, cmap='tab10', marker='o')
    plt.title(f'{tensor_name} 2D scatter plot')
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.grid(True)
    os.makedirs(f"./results/{resultname}/{name}/scatter2d/", exist_ok=True)
    plt.savefig(f"./results/{resultname}/{name}/scatter2d/{epoch}_{tensor_name}.png")


def visualize_flows(input, mu, z, output, resultname, name, epoch, num_flows=8):

    # Flatten the input tensor to [N, C] where C is the product of all dimensions except the first one
    input = input.reshape(input.shape[0], -1)
    mu = mu.reshape(mu.shape[0], -1)
    z = z.reshape(z.shape[0], -1)
    output = output.reshape(output.shape[0], -1)

    # Split the tensors into [num_flows, C]
    input = input[:num_flows]
    mu = mu[:num_flows]
    z = z[:num_flows]
    output = output[:num_flows]

    # Convert tensors to numpy arrays if they are torch tensors
    if torch.is_tensor(input):
        input = input.cpu().detach().numpy()
    if torch.is_tensor(mu):
        mu = mu.cpu().detach().numpy()
    if torch.is_tensor(z):
        z = z.cpu().detach().numpy()
    if torch.is_tensor(output):
        output = output.cpu().detach().numpy()

    # Create a figure and axis
    fig, ax = plt.subplots(figsize=(10, 6))

    # Generate a color for each point
    colors = plt.cm.Spectral(np.linspace(0, 1, len(input)))

    # Plot the points with different colors
    dummy = [np.ones_like(input[0])*i for i in range(4)] # x-axis dummy values
    for i in range(len(input)):
        ax.scatter(dummy[0], input[i], color=colors[i], label='input' if i == 0 else "")
        ax.scatter(dummy[1], mu[i], color=colors[i], label='mu' if i == 0 else "")
        ax.scatter(dummy[2], z[i], color=colors[i], label='z' if i == 0 else "")
        ax.scatter(dummy[3], output[i], color=colors[i], label='recon' if i == 0 else "")

        # Draw lines between corresponding points
        ax.plot([0, 1], [input[i], mu[i]], color=colors[i], linestyle='-')
        ax.plot([1, 2], [mu[i], z[i]], color=colors[i], linestyle='-')
        ax.plot([2, 3], [z[i], output[i]], color=colors[i], linestyle='-')

    # Set labels and title
    ax.set_xticks([0, 1, 2, 3])
    ax.set_xticklabels(['input', 'mu', 'z', 'recon'])
    ax.set_title('Visualized flows')
    #ax.legend()

    # Save the plot
    os.makedirs(f"./results/{resultname}/{name}/visualize_flows/", exist_ok=True)
    plt.savefig(f"./results/{resultname}/{name}/visualize_flows/{epoch}_flows.png")
    plt.close()