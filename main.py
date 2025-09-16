from tqdm import tqdm
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from torchvision.utils import save_image
import torch
import torchvision
import os
import model as Model
from utils import pca_visualization
from utils import apply_grad_clip
import utils
from absl import flags
import sys
 
# dataset config forwarding
DATASET_PARAMS = {}
import dataset
import yaml
import random
import numpy as np

# For point cloud visualization
try:
    import open3d as o3d
    HAS_OPEN3D = True
except ImportError:
    HAS_OPEN3D = False

# 재현성을 위한 시드 고정
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

def load_config(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def save_point_cloud(points, filepath):
    """Save point cloud in .ply format."""
    if HAS_OPEN3D:
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        o3d.io.write_point_cloud(filepath, pcd)
    else:
        print(f"Warning: open3d not available. Skipping {filepath}")

def save_set_samples(model, loader_test, device, output_dir, name, epoch, n_samples=4):
    """Save reconstruction and prior samples for SetVAE/SetLRVAE models."""
    if not HAS_OPEN3D:
        print("Warning: open3d not available. Skipping point cloud visualization.")
        return
    
    model.eval()
    os.makedirs(output_dir, exist_ok=True)
    
    with torch.no_grad():
        # 1. Save reconstruction samples (4 samples)
        print(f"Saving {n_samples} reconstruction samples...")
        for i, (x, _) in enumerate(loader_test):
            if i >= n_samples:
                break
            x = x.to(device)
            result = model(x, latent_rand_sampling=False)  # Use mean for reconstruction
            recon_points = result[0].cpu().numpy()  # [1, N, 3]
            
            # Save reconstruction
            recon_path = os.path.join(output_dir, f"{name}_epoch{epoch}_recon_{i:02d}.ply")
            save_point_cloud(recon_points[0], recon_path)
            
            # Save original for comparison
            orig_path = os.path.join(output_dir, f"{name}_epoch{epoch}_orig_{i:02d}.ply")
            save_point_cloud(x.cpu().numpy()[0], orig_path)
        
        # 2. Save prior samples (4 samples)
        print(f"Saving {n_samples} prior samples...")
        for i in range(n_samples):
            z = torch.randn(1, model.latent_channel, device=device)
            sample_points = model.decode(z).cpu().numpy()  # [1, N, 3]
            
            # Save prior sample
            sample_path = os.path.join(output_dir, f"{name}_epoch{epoch}_prior_{i:02d}.ply")
            save_point_cloud(sample_points[0], sample_path)
    
    print(f"Point cloud samples saved to: {output_dir}")

def eval(model: Model.VAE, loader_test, device, epoch, name, resultname, save_img=True, visualize=True, data_type='2d'):
    model.eval() #
    loss_total = 0.0
    loss_recon_total = 0.0
    loss_reg_total = 0.0
    loss_lr_total = 0.0

    # Validation loop
    for x, y in tqdm(loader_test, leave=False, desc="Evaluate"):
        x = x.to(device)
        y = y.to(device)

        result = model(x)
        loss, loss_recon, loss_reg, loss_lr = model.loss(x, *result)
        loss_total += float(loss)
        loss_recon_total += float(loss_recon)
        loss_reg_total += float(loss_reg)
        loss_lr_total += float(loss_lr)

    if visualize and (data_type == '1d'):
        utils.visualize_2c_points_on_image(x, y, resultname, name, epoch, "input")
        utils.visualize_2c_points_on_image(result[1], y, resultname, name, epoch, "mu") # only work on latent channel==2
        utils.visualize_2c_points_on_image(result[3], y, resultname, name, epoch, "z") # only work on latent channel==2
        utils.visualize_2c_points_on_image(result[0], y, resultname, name, epoch, "recon")
        xpx = (
                torch.randn((x.shape[0], model.latent_channel))
                .requires_grad_(False)
                .to(device)
            )
        sample_res = model.decode(xpx)
        utils.visualize_2c_points_on_image(sample_res, y, resultname, name, epoch, "sample")

    #if visualize:
    #    utils.visualize_flows(x, result[1], result[3], result[0], resultname, name, epoch)

    if save_img: # == True) and (((epoch & (epoch - 1)) == 0) or (epoch % 100 == 99)):
        os.makedirs("./results/"+resultname+"/" + name + "/valontr", exist_ok=True)
        for _ in tqdm(range(1), leave=False, desc="Test"):
            x, _ = next(iter(loader_test))
            x = x.requires_grad_(True).to(device)
            result = model(x)
            result_wo_sampling = model(x, latent_rand_sampling=False)

            # Save reconstruction example
            save_image(
                x[:256],
                "./results/"+resultname+"/" + name + "/valontr/" + str(epoch) + "_origin.png",
                normalize=True,
                nrow=16,
            )
            save_image(
                result[0][:256].clip(0, 1),
                "./results/"+resultname+"/" + name + "/valontr/" + str(epoch) + "_recon.png",
                normalize=True,
                nrow=16,
            )
            save_image(
                result_wo_sampling[0][:256].clip(0, 1),
                "./results/"+resultname+"/" + name + "/valontr/" + str(epoch) + "_recon_wos.png",
                normalize=True,
                nrow=16,
            )

            # Save sampled example
            x = (
                torch.randn((x.shape[0], model.latent_channel))
                .requires_grad_(False)
                .to(device)
            )
            result = model.decode(x)
            save_image(
                result[:256].clip(0, 1),
                "./results/"+resultname+"/" + name + "/valontr/" + str(epoch) + "_sample.png",
                normalize=True,
                nrow=16,
            )

    # PCA visualization
    if visualize: # and ((epoch & (epoch - 1)) == 0 or epoch % 100 == 99):
        pca_visualization(model, loader_test, device, epoch, name, resultname)
        
    return loss_total / len(loader_test), loss_recon_total / len(loader_test), loss_reg_total / len(loader_test), loss_lr_total / len(loader_test)

def train_and_test(model: Model.VAE, epochs=100, batch_size=128, device="cuda", dataset_name='mnist', logfilename='log.csv', resultname='res', pt_param=None, num_mc_samples=1, grad_clip=None):
    # 데이터 타입 기본값 설정 (2d: 이미지), 필요 시 덮어쓰기
    data_type = '2d'
    
    train_dataset, test_dataset = dataset.load_dataset(dataset_name, **DATASET_PARAMS)

    test_shuffle = True if (dataset_name == 'pinwheel' or dataset_name == 'chessboard') else False
    
    loader_train = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=8,
        drop_last=True,
        pin_memory=True,
    )
    loader_test = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=test_shuffle,
        num_workers=8,
        drop_last=True,
        pin_memory=True,
    )

    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, epochs * len(loader_train)
    )

    if (pt_param != None) and os.path.exists(pt_param):
        model.load_state_dict(torch.load(pt_param))
    elif (pt_param != None):
        print('No such file:', pt_param)
        exit()

    name = type(model).__name__ + datetime.now().strftime(" %m%d%H%M")
    if not type(model).__name__.startswith("NaiveAE"):
        name += "_b=" + str(float(model.beta))
    if type(model).__name__.startswith("LR"):
        name += "_a="+str(model.alpha)
    if model.is_log_mse == True:
        name += "_logmse"  # https://proceedings.mlr.press/v139/rybkin21a.html
    if type(model).__name__ == "LIDVAE":
        name += "_il=" + str(float(model.il_factor))

    writer = SummaryWriter(log_dir="runs/" + name)    
    os.makedirs(f"./results/{resultname}/{name}/params/", exist_ok=True)

    # Main train loop
    for epoch in tqdm(range(epochs), desc=name):
        model.train()
        model.warmup(epoch, epochs)
        loss_total = 0.0
        loss_recon_total = 0.0
        loss_reg_total = 0.0
        loss_lr_total = 0.0

        # Train loop
        for x, y in tqdm(loader_train, leave=False, desc="Train"):
            x = x.to(device)
            y = y.to(device)

            result = model(x, L=num_mc_samples)
            loss, loss_recon, loss_reg, loss_lr = model.loss(x, *result)

            optimizer.zero_grad()
            did_backward = False
            # latent recon term (may be detached for some models like SetVAE)
            if hasattr(loss_lr, 'requires_grad') and loss_lr.requires_grad:
                loss_lr.backward(retain_graph=True)
                did_backward = True
                # encoder의 gradient에 weight lam 곱하기
                lam = 0.0001 # 1: 기존 방법(oversmoothing), 0: encoder에는 lr 적용 안함(collapsed)
                for param in model.encoder.parameters():
                    if param.grad is not None:
                        param.grad *= lam
            # regularization term (KL); may be detached depending on model
            if hasattr(loss_reg, 'requires_grad') and loss_reg.requires_grad:
                loss_reg.backward(retain_graph=True)
                did_backward = True
            # reconstruction term; for set models, this is chamfer and requires grad
            if hasattr(loss_recon, 'requires_grad') and loss_recon.requires_grad:
                loss_recon.backward()
                did_backward = True
            # Fallback: if none required grad, backprop total loss once
            if not did_backward:
                loss.backward()
            #loss.backward()
            apply_grad_clip(model, grad_clip)
            optimizer.step()
            scheduler.step()

            loss_total += float(loss)
            loss_recon_total += float(loss_recon)
            loss_reg_total += float(loss_reg)
            loss_lr_total += float(loss_lr)
        
        writer.add_scalar("loss/train", loss_total / len(loader_train), epoch) # divide by 0 인 경우 batch size 살펴보기
        writer.add_scalar("recon/train", loss_recon_total / len(loader_train), epoch)
        writer.add_scalar("reg/train", loss_reg_total / len(loader_train), epoch)
        
        if dataset_name == 'pinwheel' or dataset_name == 'chessboard':
            data_type = '1d'
        # Set 데이터(setvae, setlrvae)는 2D 이미지 시각화/저장 비활성화
        is_set_model = getattr(model, 'data_type', None) == 'set'
        visualize, save_img = (True, True) if ((epoch == (epochs-1)) and not is_set_model) else (False, False)
        loss_total, loss_recon_total, loss_reg_total, loss_lr_total = eval(model, loader_test, device, epoch, name, resultname, save_img=save_img, visualize=visualize, data_type=data_type)

        # eval() 반환은 이미 배치 평균이므로, 추가 나눗셈 없이 그대로 기록
        writer.add_scalar("loss/test", loss_total, epoch)
        if epoch == (epochs-1):
            torch.save(
                model.state_dict(), "./results/" + resultname + "/" + name + "/params/model_" + str(epoch) + ".pt"
            )
            
            # Save point cloud samples for SetVAE/SetLRVAE models
            if is_set_model:
                point_cloud_dir = os.path.join("./results", resultname, name, "point_clouds")
                save_set_samples(model, loader_test, device, point_cloud_dir, name, epoch, n_samples=4)

    epoch = epochs
    #loss_total, loss_recon_total, loss_reg_total, loss_lr_total = eval(model, loader_test, device, epoch, name, resultname, visualize=True)

    # Generate samples to calculate FID score
    # We cannot use no_grad, since LIDVAE requires calculation of gradient
    # with torch.no_grad():
    if epochs < 0: # run only test
        os.makedirs("./results/"+resultname+"/" + name + "/generation", exist_ok=True)

        SAMPLE_ITERATION = 50
        for i in tqdm(range(SAMPLE_ITERATION), leave=False, desc="Generate"):
            x = (
                torch.randn((batch_size, model.latent_channel))
                .requires_grad_(True)
                .to(device)
            )
            x = model.decode(x).clip(0, 1)

            for j in range(batch_size):
                save_image(
                    x[j],
                    "./results/"+resultname+"/"
                    + name
                    + "/generation/"
                    + str(i * batch_size + j)
                    + ".png",
                    normalize=True,
                    nrow=1,
                )

    writer.close()

    # Calculate FID via `pytorch_fid` lib
    fid = -1
    if epochs < 0: # run only test
        fid = "None"
        try:
            import pytorch_fid
            fid = os.popen(
                f'python -m pytorch_fid ../mnist/ "./results/{resultname}/{name}/generation/" --device cuda:0'
            ).read()
            print('fid:',fid)
        except ModuleNotFoundError:
            print("Please install `pytorch_fid` to show FID score")

    # Calculate NLL, AU, KL
    loader_eval = DataLoader(
        test_dataset,
        batch_size=50,
        shuffle=True,
        num_workers=8,
        drop_last=True,
        pin_memory=True,
    )
    au, kl, mi, nll, mvar = utils.measure_pc_runmodel(model, loader_eval, device)
    print('| au:', au, '| kl:', kl, '| mi:', mi, '| nll:', nll)

    utils.log_unified_dict("./log/", {'name':name, 'dataset_name':dataset_name, 'epoch':epoch+1, 'fid':fid, 'au':au, 'kl':kl, 'mi':mi, 'nll':nll, 'vloss':loss_total/len(loader_test), \
                                    'vlrec':loss_recon_total/len(loader_test), 'vlreg':loss_reg_total/len(loader_test), 'vllr':loss_lr_total/len(loader_test), \
                                    'mean_var':mvar},
                                    logfilename=logfilename)

def run_experiment(config_path):
    config = load_config(config_path)
    
    exp_type = config['experiment_type']
    common_params = config['common_params']
    model_params = config['model_params']


    if model_params['residual_connection']:
        str_res='_res'
    else:
        str_res=''
    exp_config_str = f"{common_params['exp_data']}_{exp_type}{str_res}_depth{len(model_params['hchans'])}_mc{model_params['num_mc_samples']}"

    if common_params['logfilename'] is None:
        logfilename=f"log_{exp_config_str}.csv"
    else:
        logfilename=common_params['logfilename']
        
    if common_params['resultname'] is None:
        resultname=f"result_{exp_config_str}"
    else:
        resultname=common_params['resultname']
    
    global DATASET_PARAMS
    DATASET_PARAMS = common_params.get('dataset_params', {})

    if exp_type == 'lidvae':
        for beta in model_params['beta_list']:
            for il in model_params['il_list']:
                for i in range(common_params['niter']):
                    model = Model.LIDVAE(
                        is_log_mse=model_params.get('log_mse', False),
                        inverse_lipschitz=il,
                        beta=beta,
                        dataset=common_params['exp_data'],
                        hidden_channels=model_params.get('hchans', None)
                    )
                    train_and_test(
                        model,
                        epochs=common_params['exp_epochs'],
                        batch_size=common_params['batch_size'],
                        dataset_name=common_params['exp_data'],
                        logfilename=logfilename,
                        resultname=resultname,
                        pt_param=common_params.get('pt_param', None),
                        num_mc_samples=model_params.get('num_mc_samples', 1),
                        grad_clip=common_params.get('grad_clip', None)
                    )
                    
    elif exp_type == 'vae':
        for beta in model_params['beta_list']:
            for i in range(common_params['niter']):
                model = Model.VanillaVAE(
                    beta=beta,
                    dataset=common_params['exp_data'],
                    hidden_channels=model_params.get('hchans', None),
                    encoder_type=model_params.get('encoder_type', 'conv'),
                    decoder_type=model_params.get('decoder_type', 'mlp'),
                    fixed_var=model_params.get('fixed_var', False),
                    residual_connection=model_params.get('residual_connection', False)
                )
                train_and_test(
                    model,
                    epochs=common_params['exp_epochs'],
                    batch_size=common_params['batch_size'],
                    dataset_name=common_params['exp_data'],
                    logfilename=logfilename,
                    resultname=resultname,
                    pt_param=common_params.get('pt_param', None),
                    num_mc_samples=model_params.get('num_mc_samples', 1),
                    grad_clip=common_params.get('grad_clip', None)
                )
                
    elif exp_type == 'nae':
        for i in range(common_params['niter']):
            model = Model.NaiveAE(
                dataset=common_params['exp_data'],
                hidden_channels=model_params.get('hchans', None),
                encoder_type=model_params.get('encoder_type', 'conv'),
                decoder_type=model_params.get('decoder_type', 'mlp')
            )
            train_and_test(
                model,
                epochs=common_params['exp_epochs'],
                batch_size=common_params['batch_size'],
                dataset_name=common_params['exp_data'],
                logfilename=logfilename,
                resultname=resultname,
                pt_param=common_params.get('pt_param', None),
                num_mc_samples=model_params.get('num_mc_samples', 1),
                grad_clip=common_params.get('grad_clip', None)
            )

    elif exp_type == 'lrvae':
        for alpha in model_params['alpha_list']:
            for beta in model_params['beta_list']:
                for i in range(common_params['niter']):
                    model = Model.LRVAE(
                        beta=beta,
                        alpha=alpha,
                        z_source=model_params.get('z_source', 'Ex'),
                        dataset=common_params['exp_data'],
                        hidden_channels=model_params.get('hchans', None),
                        pwise_reg=model_params.get('pwise_reg', False),
                        encoder_type=model_params.get('encoder_type', 'conv'),
                        decoder_type=model_params.get('decoder_type', 'mlp'),
                        residual_connection=model_params.get('residual_connection', False)
                    )
                    train_and_test(
                        model,
                        epochs=common_params['exp_epochs'],
                        batch_size=common_params['batch_size'],
                        dataset_name=common_params['exp_data'],
                        logfilename=logfilename,
                        resultname=resultname,
                        pt_param=common_params.get('pt_param', None),
                        num_mc_samples=model_params.get('num_mc_samples', 1),
                        grad_clip=common_params.get('grad_clip', None)
                    )

    elif exp_type == 'setvae':
        for beta in model_params.get('beta_list', [1.0]):
            for i in range(common_params['niter']):
                model = Model.SetVAE(
                    beta=beta,
                    latent_channel=model_params.get('latent_channel', 128),
                    num_points=model_params.get('num_points', 2048),
                    encoder_hidden=model_params.get('encoder_hidden', [128,256,512]),
                    decoder_hidden=model_params.get('decoder_hidden', [512,256,128]),
                    dataset='shapenet',
                    pool_type=model_params.get('pool_type', 'max'),
                )
                train_and_test(
                    model,
                    epochs=common_params['exp_epochs'],
                    batch_size=common_params['batch_size'],
                    dataset_name=common_params.get('exp_data', 'shapenet'),
                    logfilename=logfilename,
                    resultname=resultname,
                    pt_param=common_params.get('pt_param', None),
                    num_mc_samples=model_params.get('num_mc_samples', 1),
                    grad_clip=common_params.get('grad_clip', None)
                )

    elif exp_type == 'setlrvae':
        for alpha in model_params.get('alpha_list', [0.01]):
            for beta in model_params.get('beta_list', [1.0]):
                for i in range(common_params['niter']):
                    model = Model.SetLRVAE(
                        alpha=alpha,
                        beta=beta,
                        latent_channel=model_params.get('latent_channel', 128),
                        num_points=model_params.get('num_points', 2048),
                        encoder_hidden=model_params.get('encoder_hidden', [128,256,512]),
                        decoder_hidden=model_params.get('decoder_hidden', [512,256,128]),
                        dataset='shapenet',
                        pool_type=model_params.get('pool_type', 'max'),
                    )
                    train_and_test(
                        model,
                        epochs=common_params['exp_epochs'],
                        batch_size=common_params['batch_size'],
                        dataset_name=common_params.get('exp_data', 'shapenet'),
                        logfilename=logfilename,
                        resultname=resultname,
                        pt_param=common_params.get('pt_param', None),
                        num_mc_samples=model_params.get('num_mc_samples', 1),
                        grad_clip=common_params.get('grad_clip', None)
                    )

if __name__ == "__main__":

    FLAGS = flags.FLAGS
    flags.DEFINE_string('config', './configs/config_shapenet_setvae.yaml', 'config file path')

    if not FLAGS.is_parsed():
        FLAGS(sys.argv)

    run_experiment(FLAGS.config)
