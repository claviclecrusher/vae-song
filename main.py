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

def eval(model: Model.VAE, loader_test, device, epoch, name, resultname, save_img=True, pca=True):
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

    if save_img == True and (epoch & (epoch - 1)) == 0:
        os.makedirs("./results/"+resultname+"/" + name + "/valontr", exist_ok=True)
        for _ in tqdm(range(1), leave=False, desc="Test"):
            x, _ = next(iter(loader_test))
            x = x.requires_grad_(True).to(device)
            result = model(x)

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

            # Save sampled example
            x = (
                torch.randn((x.shape[0], model.latent_channel))
                .requires_grad_(True)
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
    if pca == True:
        pca_visualization(model, loader_test, device, epoch, name, resultname)
        
    return loss_total / len(loader_test), loss_recon_total / len(loader_test), loss_reg_total / len(loader_test), loss_lr_total / len(loader_test)



def train_and_test(model: Model.VAE, epochs=100, batch_size=128, device="cuda", dataset_name='mnist', logfilename='log.csv', resultname='res', pt_param=None):
    if dataset_name == 'mnist':
        transforms = torchvision.transforms.Compose(
            [
                torchvision.transforms.RandomRotation(20),
                torchvision.transforms.RandomResizedCrop((28, 28), (0.9, 1), (0.9, 1.1)),
                torchvision.transforms.ToTensor(),
            ]
        )
        train_dataset = torchvision.datasets.MNIST(root="dataset/", transform=transforms, download=True)
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
        train_dataset = torchvision.datasets.CelebA(root="dataset/", transform=transforms, download=True)
        test_dataset = torchvision.datasets.CelebA(root="dataset/", transform=transforms, split="test")
    elif dataset_name == 'fashionmnist':
        transforms = torchvision.transforms.Compose(
            [
                torchvision.transforms.Resize(28),
                torchvision.transforms.ToTensor(),
            ]
        )
        train_dataset = torchvision.datasets.FashionMNIST(root="dataset/", transform=transforms, download=True)
        test_dataset = torchvision.datasets.FashionMNIST(root="dataset/", transform=transforms, train=False)
    else:
        print(dataset_name, "is not implemented")
        exit()
    
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
        shuffle=False,
        num_workers=8,
        drop_last=True,
        pin_memory=True,
    )


    #if torch.cuda.device_count() > 1:
    #    model = torch.nn.DataParallel(model) # 쓰면 model attribute 직접 접근 어려워짐. 코드 수정 필요.
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
    name += "_b=" + str(float(model.beta))
    if type(model).__name__.startswith("LR"):
        name += "_a="+str(model.alpha)
    if model.is_log_mse == True:
        name += "_logmse"  # https://proceedings.mlr.press/v139/rybkin21a.html
    if type(model).__name__ == "LIDVAE":
        name += "_il=" + str(float(model.il_factor))

    writer = SummaryWriter(log_dir="runs/" + name)    
    os.makedirs("./results/"+resultname+"/" + name + "/params/", exist_ok=True)
    os.makedirs("./results/"+resultname+"/" + name + "/params/", exist_ok=True)

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

            result = model(x)
            loss, loss_recon, loss_reg, loss_lr = model.loss(x, *result)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            loss_total += float(loss)
            loss_recon_total += float(loss_recon)
            loss_reg_total += float(loss_reg)
            loss_lr_total += float(loss_lr)

        writer.add_scalar("loss/train", loss_total / len(loader_train), epoch)
        writer.add_scalar("recon/train", loss_recon_total / len(loader_train), epoch)
        writer.add_scalar("reg/train", loss_reg_total / len(loader_train), epoch)
        
        do_pca = ((epoch & (epoch - 1)) == 0) or (epoch % 100 == 0) # do pca if epoch is power of 2
        loss_total, loss_recon_total, loss_reg_total, loss_lr_total = eval(model, loader_test, device, epoch, name, resultname, pca=do_pca)

        writer.add_scalar("loss/test", loss_total / len(loader_test), epoch)
        if epoch % 100 == 1 or (epoch & (epoch - 1)) == 0:
            torch.save(
                model.state_dict(), "./results/" + resultname + "/" + name + "/params/model_" + str(epoch) + ".pt"
            )

    epoch = epochs
    loss_total, loss_recon_total, loss_reg_total, loss_lr_total = eval(model, loader_test, device, epoch, name, resultname, pca=True)

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
    au, kl, mi, nll = utils.measure_pc(model, loader_eval, device)
    print('| au:', au, '| kl:', kl, '| mi:', mi, '| nll:', nll)

    utils.log_unified("./log/", [name, dataset_name, epoch+1, fid, au, kl, mi, nll, loss_total/len(loader_test), \
                                loss_recon_total/len(loader_test), loss_reg_total/len(loader_test), loss_lr_total/len(loader_test)],
                                ['name', 'dataset_name', 'epoch', 'fid', 'au', 'kl', 'mi', 'nll', 'vloss', 'vlrec', 'vlreg', 'vllr'],
                                logfilename=logfilename)


def exp_lidvae(niter=1, exp_epochs=100, batch_size=128, exp_data='mnist', beta_list=[0.4], log_mse=False,
               il_list=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0], logfilename='log_lidvae.csv', resultname='res_lidvae', pt_param=None):
    for b in beta_list:
        for il in il_list:
            for i in range(niter):
                train_and_test(Model.LIDVAE(is_log_mse=log_mse, inverse_lipschitz=il, beta=b, dataset=exp_data), epochs=exp_epochs, batch_size=batch_size,
                               dataset_name=exp_data, logfilename=logfilename, resultname=resultname, pt_param=pt_param)
    return


def exp_vae(niter=1, exp_epochs=100, batch_size=128, exp_data='mnist', beta_list=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0], logfilename='log_vae.csv', resultname='res_vae', pt_param=None):
    for b in beta_list:
        for i in range(niter):
            train_and_test(Model.VanillaVAE(beta=b, dataset=exp_data), epochs=exp_epochs, batch_size=batch_size,
                           dataset_name=exp_data, logfilename=logfilename, resultname=resultname, pt_param=pt_param)
    return

def exp_lrvae(niter=1, exp_epochs=100, batch_size=128, exp_data='mnist', beta_list=[0.1], alpha_list=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0], pz=True, logfilename='log_lrvae.csv', resultname='res_lrvae', pt_param=None):
    for a in alpha_list:
        for b in beta_list:
            for i in range(niter):
                train_and_test(Model.LRVAE(beta=b, alpha=a, from_pz=pz, dataset=exp_data), epochs=exp_epochs, batch_size=batch_size,
                               dataset_name=exp_data, logfilename=logfilename, resultname=resultname, pt_param=pt_param)
    return

def exp_vae_conv(niter=1, exp_epochs=100, batch_size=128, exp_data='mnist', beta_list=[0.5, 1.0, 2.0, 4.0, 8.0, 16.0], 
                 logfilename='log_conv.csv', resultname='res_conv', pt_param=None):
    for b in beta_list:
        for i in range(niter):
            train_and_test(Model.ConvVAE(is_log_mse=False, beta=b, dataset=exp_data), epochs=exp_epochs, batch_size=batch_size,
                           dataset_name=exp_data, logfilename=logfilename, resultname=resultname, pt_param=pt_param)
    return



if __name__ == "__main__":
    exp_epochs = 100 # 0 for only testing
    exp_data = 'mnist'
    model_dict = {"vae": Model.VanillaVAE(dataset=exp_data), "lrvae": Model.LRVAE(dataset=exp_data), "lidvae":Model.LIDVAE(dataset=exp_data)}

    # Train
    #exp_vae(1, exp_epochs=1, batch_size=128, exp_data=exp_data, beta_list=[1.0], logfilename='TEST.csv', resultname='TEST')
    #exp_vae(1, exp_epochs=exp_epochs, batch_size=128, exp_data=exp_data, beta_list=[3.0], logfilename='log_vae_mnist.csv', resultname='result_vae_mnist')
    exp_lrvae(1, exp_epochs=exp_epochs, batch_size=128, exp_data=exp_data, beta_list=[0.0, 0.1, 0.2, 0.4, 1.0, 2.0], alpha_list=[0.0, 0.01, 0.1, 0.2, 0.4, 1.0], logfilename='log_lrvae_mnist.csv', resultname='result_lrvae_mnist')
    #exp_lidvae(1, exp_epochs=exp_epochs, batch_size=128, exp_data=exp_data, beta_list=[10.0], il_list=[0.0], logfilename='log_lidvae_lm_mnist.csv', resultname='result_lidvae_lm_mnist', log_mse=True)

    # Test    
    #exp_vae(1, exp_epochs=0, batch_size=128, exp_data=exp_data, beta_list=[1.0], logfilename='TEST.csv', resultname='TEST', pt_param='./results/result_vae_mnist/VanillaVAE 11071505_b=1.0/model_99.pt')
    #exp_lrvae(1, exp_epochs=0, batch_size=128, exp_data=exp_data, beta_list=[0.1], alpha_list=[0.1], logfilename='TEST.csv', resultname='TEST', pt_param='./results/result_lrvae_mnist/LRVAE 11081131_b=3.0_a=0.1/model_99.pt')
    #exp_lidvae(1, exp_epochs=0, batch_size=128, exp_data=exp_data, beta_list=[10.0], il_list=[0.0], logfilename='TEST.csv', resultname='TEST', pt_param='./results/result_lidvae_lm_mnist/LIDVAE 11151324_b=10.0_logmse_il=0.0/model_99.pt')

    # Curve Scatter
    #rec_lr_scatter_visualization(model_dict, exp_data, "cuda")






    #exp_vae_ep(4, exp_epochs, 128, 'mnist', beta_list=[0.1, 0.5, 1.0, 2.0, 4.0], logfilename='log_vaeep.csv', resultname='result_mnist_vaeep')
    #exp_vae_ep(4, exp_epochs, 128, 'fashionmnist', beta_list=[0.1, 0.5, 1.0, 2.0, 4.0], logfilename='log_vaeep.csv', resultname='result_fashionmnist_vaeep')
    #exp_vae_ep(4, exp_epochs, 64, 'celeba', beta_list=[0.1, 0.5, 1.0, 2.0, 4.0], logfilename='log_vaeep.csv', resultname='result_celeba_vaeep')

    #exp_lidvae(1, exp_epochs, 128, 'mnist', beta_list=[1.0, 3.0], il_list=[0.0, 0.5, 1.0, 1.5], logfilename='log_lidvae.csv', resultname='result_mnist_ilidvae')
    #exp_lidvae(4, exp_epochs, 128, 'celeba', beta_list=[0.1, 0.5, 1.0, 2.0, 4.0], il_list=[0.0, 0.1, 0.2, 0.4], logfilename='lidvae_celeba.csv', resultname='ilidvae_celeba')

    #exp_lrvae_ep(4, exp_epochs, 128, 'mnist', beta_list=[0.1, 0.5, 1.0, 2.0, 4.0], alpha_list=[0.01, 0.1, 0.4, 1.0],
    #           logfilename='log_lrvae.csv', resultname='result_mnist_lrvae')
    #exp_lrvae_ep(4, exp_epochs, 128, 'fashionmnist', beta_list=[0.1, 0.5, 1.0, 2.0, 4.0], alpha_list=[0.01, 0.1, 0.4, 1.0],
    #           logfilename='log_lrvae.csv', resultname='result_fashionmnist_lrvae')

    #exp_vae_conv(4, exp_epochs, 128, 'mnist', beta_list=[0.1, 0.5, 1.0, 2.0, 4.0], 
    #           logfilename='log_conv.csv', resultname='result_mnist_conv')
    #exp_vae_conv(4, exp_epochs, 128, 'fashionmnist', beta_list=[0.1, 0.5, 1.0, 2.0, 4.0], 
    #           logfilename='log_conv.csv', resultname='result_fashionmnist_conv')
    #exp_vae_conv(4, exp_epochs, 64, 'celeba', beta_list=[0.1, 0.5, 1.0, 2.0, 4.0], 
    #           logfilename='log_conv.csv', resultname='result_celeba_conv')

    #exp_vae(beta_list=[1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0])
    #exp_lrvae(niter=2, beta_list=[0.4], alpha_list=[1.0], pz=False, logfilename='log_lrvae_mnist_qzx_ba.csv', resultname='lrvae_mnist_qzx_ba')
    #exp_lidvae()
    #exp_lidvae(1, 1, exp_data, beta_list=[0.5, 1.0, 2.0, 4.0], il_list=[0.0, 0.5, 1.0, 1.5], logfilename='log_mnist_lidvae.csv', resultname='result_mnist_ilid')
    #train_and_test(Model.LRVAE_EP(is_log_mse=False, beta=1.0, alpha=0.01), epochs=2, dataset_name=exp_data, logfilename='TEST.csv', resultname='TEST')
    #train_and_test(Model.VanillaVAE(beta=0.1, dataset=exp_data), epochs=exp_epochs, dataset_name=exp_data, logfilename='TEST.csv', resultname='TEST')
    #train_and_test(Model.VanillaVAE_EP(beta=0.1, dataset=exp_data), epochs=exp_epochs, dataset_name=exp_data, logfilename='TEST.csv', resultname='TEST')
    #train_and_test(Model.LRVAE(beta=0.1, dataset=exp_data, alpha=0.01)), epochs=exp_epochs, dataset_name=exp_data, logfilename='TEST.csv', resultname='TEST')
