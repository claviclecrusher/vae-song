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
import utils
import dataset
import yaml
import random
import numpy as np

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

def train_and_test(model: Model.VAE, epochs=100, batch_size=128, device="cuda", dataset_name='mnist', logfilename='log.csv', resultname='res', pt_param=None, num_mc_samples=1):
    # 데이터 타입 기본값 설정 (2d: 이미지), 필요 시 덮어쓰기
    data_type = '2d'
    
    train_dataset, test_dataset = dataset.load_dataset(dataset_name)

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
            loss_lr.backward(retain_graph=True)
            # encoder의 gradient에 weight lam 곱하기
            lam = 0.0001 # 1: 기존 방법(oversmoothing), 0: encoder에는 lr 적용 안함(collapsed)
            for param in model.encoder.parameters():
                if param.grad is not None:
                    param.grad *= lam
            loss_reg.backward(retain_graph=True)
            loss_recon.backward()
            #loss.backward()
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
        #visualize, save_img = (True, True) if ((epoch == (epochs-1)) or ((epoch & (epoch - 1)) == 0)) else (False, False)
        visualize, save_img = (True, True) if (epoch == (epochs-1)) else (False, False)
        loss_total, loss_recon_total, loss_reg_total, loss_lr_total = eval(model, loader_test, device, epoch, name, resultname, save_img=save_img, visualize=visualize, data_type=data_type)

        # eval() 반환은 이미 배치 평균이므로, 추가 나눗셈 없이 그대로 기록
        writer.add_scalar("loss/test", loss_total, epoch)
        if epoch == (epochs-1):
            torch.save(
                model.state_dict(), "./results/" + resultname + "/" + name + "/params/model_" + str(epoch) + ".pt"
            )

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
                        num_mc_samples=model_params.get('num_mc_samples', 1)
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
                    num_mc_samples=model_params.get('num_mc_samples', 1)
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
                num_mc_samples=model_params.get('num_mc_samples', 1)
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
                        num_mc_samples=model_params.get('num_mc_samples', 1)
                    )

if __name__ == "__main__":
    config_path = "./configs/config_pinwheel.yaml"
    run_experiment(config_path)
