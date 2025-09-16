import torch
import module
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import math


class VAE(torch.nn.Module):
    def __init__(self):
        super(VAE, self).__init__()
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d) or isinstance(m, torch.nn.Linear):
                init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, torch.nn.BatchNorm2d) or isinstance(m, torch.nn.BatchNorm1d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)

    def encode(self, input):
        raise NotImplementedError

    def decode(self, input):
        raise NotImplementedError

    def forward(self, input, latent_rand_sampling=True):
        raise NotImplementedError

    def loss(self, *args):
        raise NotImplementedError

    def warmup(self, epoch, amount=None):
        return False





class FlexibleVAE(VAE):

    def __init__(
        self,
        in_channel=1,
        latent_channel=32,
        hidden_channels=None,
        icnn_channels=None,
        input_dim=28,
        beta=1.0,
        alpha=0.0,
        is_log_mse=False,
        dataset=None,
        z_source='Ex',
        bal_alpha=True,
        pwise_reg=False,
        variational=True,
        encoder_type='mlp',
        decoder_type='mlp',
        residual_connection=False,
        fixed_var=False,
    ):
        """
        VAE with residual-conv encoder and MLP decoder, for image dataset.
        """
        if dataset == "celeba":
            in_channel = 3
            latent_channel = 128
            hidden_channels = [32, 64, 128, 256] if hidden_channels == None else hidden_channels
            input_dim = 64
        elif (dataset == "mnist") or (dataset == "fashionmnist"):
            in_channel = 1
            latent_channel = 28
            hidden_channels = [32, 64, 128] if hidden_channels == None else hidden_channels
            input_dim = 28
        elif dataset == "cifar10":
            in_channel = 3
            latent_channel = 128
            hidden_channels = [32, 64, 128, 256] if hidden_channels == None else hidden_channels
            input_dim = 32
        elif dataset == "omniglot":
            in_channel = 1
            latent_channel = 32
            hidden_channels = [32, 64, 128, 256] if hidden_channels == None else hidden_channels
            input_dim = 28
        elif (dataset == "pinwheel") or (dataset == "chessboard"):
            in_channel = 2
            latent_channel = 2
            hidden_channels = [2, 2, 2, 2] if hidden_channels == None else hidden_channels
            input_dim = 1
        else:
            raise ValueError(f"Invalid dataset: {dataset}")

        super(VAE, self).__init__()

        self.variational = variational # if False, it is just an autoencoder
        self.latent_channel = latent_channel
        self.beta = beta
        self.alpha = alpha
        self.z_source = z_source
        self.wu_alpha = 0.0
        self.is_log_mse = is_log_mse
        self.balanced_alpha = bal_alpha
        self.pwise_reg = pwise_reg
        self.fixed_var = fixed_var
        self.data_type = '2d' # for image dataset
        self.residual_connection = residual_connection

        if dataset in ['pinwheel', 'chessboard']:
            self.data_type = '1d'
            
        fc_dim = input_dim
        transpose_padding = []
        for _ in range(len(hidden_channels)):
            transpose_padding.append((fc_dim + 1) % 2)
            fc_dim = (fc_dim - 1) // 2 + 1
        transpose_padding.reverse()


        # Make encoder
        if self.data_type == '1d' and encoder_type == 'mlp':
            if self.residual_connection:
                self.encoder = self.make_encoder_residual_mlp_1d(hidden_channels, in_channel, latent_channel)
            else:   
                self.encoder = self.make_encoder_mlp_1d(hidden_channels, in_channel, latent_channel)
        elif encoder_type == 'mlp':
            self.encoder = self.make_encoder_mlp_2d(hidden_channels, in_channel, latent_channel)
        elif encoder_type == 'conv':
            self.encoder = self.make_encoder_conv_2d(hidden_channels, in_channel, latent_channel, fc_dim)
        else:
            print(f'Invalid encoder type: {self.data_type} {encoder_type}')
            exit()

        # Make decoder
        if self.data_type == '1d' and decoder_type == 'mlp':
            if self.residual_connection:
                self.decoder = self.make_decoder_residual_mlp_1d(in_channel, latent_channel, list(reversed(hidden_channels)))
            else:
                self.decoder = self.make_decoder_mlp_1d(in_channel, latent_channel, list(reversed(hidden_channels)))
        elif decoder_type == 'mlp':
            self.decoder = self.make_decoder_mlp_2d(in_channel, latent_channel, input_dim)
        elif decoder_type == 'conv':
            self.decoder = self.make_decoder_conv_2d(in_channel, latent_channel, list(reversed(hidden_channels)), fc_dim, transpose_padding)
        else:
            print(f'Invalid decoder type: {self.data_type} {decoder_type}')
            exit()

        return

    # --- 공통 MLP 블록 생성 헬퍼 ---
    def _build_mlp(self, hidden_channels, in_ch, out_ch, block_fn):
        """
        hidden_channels: list of int, in_ch: int, out_ch: int
        block_fn: function(in_dim, out_dim) -> nn.Module
        """
        layers = []
        last = in_ch
        for ch in hidden_channels:
            layers.append(block_fn(last, ch))
            last = ch
        layers.append(block_fn(last, out_ch))
        return nn.Sequential(*layers)

    def make_encoder_mlp_1d(self, hidden_channels, in_channel, latent_channel):
        # MLP 인코더: Linear + BN + LeakyReLU 블록을 생성
        return self._build_mlp(
            hidden_channels,
            in_channel,
            latent_channel * 2,
            lambda in_dim, out_dim: nn.Sequential(
                nn.Linear(in_dim, out_dim),
                nn.BatchNorm1d(out_dim),
                nn.LeakyReLU(),
            ),
        )

    def make_encoder_residual_mlp_1d(self, hidden_channels, in_channel, latent_channel):
        # Residual MLP 블록 기반 인코더 생성
        return self._build_mlp(
            hidden_channels,
            in_channel,
            latent_channel * 2,
            lambda in_dim, out_dim: module.ResidualMLPBlock(in_dim, out_dim),
        )
    
    def make_encoder_mlp_2d(self, hidden_channels, in_channel, latent_channel):
        last_channel = in_channel
        encoder = []
        for channel in hidden_channels:
            encoder.append(
                torch.nn.Sequential(
                    torch.nn.Flatten(),
                    torch.nn.Linear(last_channel, channel),
                    torch.nn.BatchNorm1d(channel),
                    torch.nn.LeakyReLU(),
                )
            )
            last_channel = channel

        encoder.append(
            torch.nn.Sequential(
                torch.nn.Linear(last_channel, latent_channel * 2),
                torch.nn.BatchNorm1d(latent_channel * 2),
                torch.nn.LeakyReLU(),
                torch.nn.Linear(latent_channel * 2, latent_channel * 2),
            )
        )
        encoder = torch.nn.Sequential(*encoder)
        return encoder

    def make_encoder_conv_2d(self, hidden_channels, in_channel, latent_channel, fc_dim):
        last_channel = in_channel
        encoder = []
        for channel in hidden_channels:
            encoder.append(
                torch.nn.Sequential(
                    module.ResidualConvBlock(last_channel, channel, 2),
                    module.ResidualConvBlock(channel, channel, 1),
                )
            )
            last_channel = channel

        encoder.append(
            torch.nn.Sequential(
                torch.nn.Flatten(),
                torch.nn.Linear(last_channel * (fc_dim**2), latent_channel * 2),
                torch.nn.BatchNorm1d(latent_channel * 2),
                torch.nn.LeakyReLU(),
                torch.nn.Linear(latent_channel * 2, latent_channel * 2),
            )
        )
        encoder = torch.nn.Sequential(*encoder)
        return encoder
    
    def make_decoder_mlp_1d(self, in_channel, latent_channel, hidden_channels=[]):
        # First layer: half of final dimension
        decoder = []
        last_channel = latent_channel
        channel = in_channel

        for channel in hidden_channels:
            decoder.append(
                torch.nn.Sequential(
                    torch.nn.Linear(last_channel, channel),
                    torch.nn.BatchNorm1d(channel),
                    torch.nn.LeakyReLU(),
                )
            )
            last_channel = channel

        # Second and last layer: full dimension
        last_channel = channel
        channel = in_channel
        decoder.append(
            torch.nn.Sequential(
                torch.nn.Linear(last_channel, channel),
                #torch.nn.BatchNorm1d(channel),
                #torch.nn.LeakyReLU(),
                #torch.nn.Linear(channel, channel),
            )
        )

        # Unflatten to shape of 1d data
        #decoder.append(torch.nn.Unflatten(1, (in_channel)))

        # Note that there is no range mapping!
        # Use Clip or Sigmoid if you want
        decoder = torch.nn.Sequential(*decoder)
        return decoder


    def make_decoder_residual_mlp_1d(self, in_channel, latent_channel, hidden_channels=[]):
        # First layer: half of final dimension
        decoder = []
        last_channel = latent_channel
        channel = in_channel

        for channel in hidden_channels:
            decoder.append(
                torch.nn.Sequential(
                    module.ResidualMLPBlock(last_channel, channel),
                )
            )
            last_channel = channel

        # Second and last layer: full dimension
        last_channel = channel
        channel = in_channel
        decoder.append(
            torch.nn.Sequential(
                module.ResidualMLPBlock(last_channel, channel),
            )
        )

        # Note that there is no range mapping!
        # Use Clip or Sigmoid if you want
        decoder = torch.nn.Sequential(*decoder)
        return decoder



    def make_decoder_mlp_2d(self, in_channel, latent_channel, input_dim):
        # First layer: half of final dimension
        decoder = []
        last_channel = latent_channel
        channel = (input_dim**2) * in_channel // 2
        decoder.append(
            torch.nn.Sequential(
                torch.nn.Linear(last_channel, channel),
                torch.nn.BatchNorm1d(channel),
                torch.nn.LeakyReLU(),
                torch.nn.Linear(channel, channel),
                torch.nn.BatchNorm1d(channel),
                torch.nn.LeakyReLU(),
            )
        )

        # Second and last layer: full dimension
        last_channel = channel
        channel = (input_dim**2) * in_channel
        decoder.append(
            torch.nn.Sequential(
                torch.nn.Linear(last_channel, channel),
                torch.nn.BatchNorm1d(channel),
                torch.nn.LeakyReLU(),
                torch.nn.Linear(channel, channel),
            )
        )

        # Unflatten to shape of image
        decoder.append(torch.nn.Unflatten(1, (in_channel, input_dim, input_dim)))

        # Note that there is no range mapping!
        # Use Clip or Sigmoid if you want
        decoder = torch.nn.Sequential(*decoder)
        return decoder
    

    def make_decoder_conv_2d(self, in_channel, latent_channel, hidden_channels, fc_dim, transpose_padding):
        decoder = []
        last_channel = hidden_channels[0]

        decoder.append(
            torch.nn.Sequential(
                torch.nn.Linear(latent_channel, last_channel * (fc_dim**2)),
                torch.nn.BatchNorm1d(last_channel * (fc_dim**2)),
                torch.nn.LeakyReLU(),
                torch.nn.Unflatten(1, (last_channel, fc_dim, fc_dim)),
                module.ResidualConvBlock(last_channel, last_channel, 1),
            )
        )

        for channel, pad in zip(hidden_channels[1:], transpose_padding[:-1]):
            decoder.append(
                torch.nn.Sequential(
                    torch.nn.ConvTranspose2d(last_channel, channel, 3, 2, 1, pad),
                    torch.nn.BatchNorm2d(channel),
                    torch.nn.LeakyReLU(),
                )
            )
            last_channel = channel

        decoder.append(
            torch.nn.Sequential(
                torch.nn.ConvTranspose2d(
                    last_channel, last_channel, 3, 2, 1, transpose_padding[-1]
                ),
                torch.nn.BatchNorm2d(last_channel),
                torch.nn.LeakyReLU(),
                torch.nn.ConvTranspose2d(last_channel, in_channel, 3, 1, 1),
            )
        )
        # Note that there is no range mapping!
        # Use Clip or Sigmoid if you want
        decoder = torch.nn.Sequential(*decoder)
        return decoder
    

    def encode(self, input):
        ret = self.encoder(input)
        #return ret.split(ret.shape[1] // 2, 1)
        mu, log_var = ret.split(ret.shape[1] // 2, 1)
        #return mu, F.softplus(var) # prevent var from being negative
        return mu, log_var

    def decode(self, input):
        return self.decoder(input)


    def forward(self, input, latent_rand_sampling=True, L=1):  # L: number of samples for MC
        mu, log_var = self.encode(input)

        # 샘플 z 생성 (첫 인코더까지는 recon 용도로 경사 허용)
        if latent_rand_sampling:
            eps = torch.randn(L, *mu.shape, device=mu.device)
            input_z_stack = mu.unsqueeze(0) + eps * torch.exp(log_var * 0.5).unsqueeze(0)  # [L, B, D]
        else:
            input_z_stack = mu.unsqueeze(0)

        B = input.shape[0]
        z_flat = input_z_stack.view(-1, input_z_stack.shape[-1])  # [L*B, D]

        # 1) 재구성 경로: detach 없이 디코더로 → recon은 완전 연결 그래프 유지
        recon_flat_attached = self.decode(z_flat)  # grad → decoder, encoder1(통해 z)

        # 2) 잠재복원 경로: 첫 인코더 출력 z만 detach → grad는 decoder, encoder2로만
        z_flat_detached = z_flat.detach()
        recon_flat_lr = self.decode(z_flat_detached)             # grad → decoder
        z_recon_flat_lr, _ = self.encode(recon_flat_lr)          # grad → encoder2

        # 모양 되돌리기
        recon_stack = recon_flat_attached.view(L, B, *recon_flat_attached.shape[1:])
        z_recon_stack = z_recon_flat_lr.view(L, B, *z_recon_flat_lr.shape[1:])
        recon = recon_stack.mean(dim=0)

        # latent recon 손실 경로로는 z를 detach 해서 돌려줌
        input_z_stack_detached = input_z_stack.detach()

        return recon, mu, log_var, input_z_stack_detached, z_recon_stack


    def forward_regacy(self, input, latent_recon=True, latent_rand_sampling=True, L=1):
        if self.variational == False:
            return self.forward_ae(input)
        if self.z_source == 'pz':
            return self.forward_pz(input, latent_rand_sampling=latent_rand_sampling, L=L)
        elif self.z_source == 'qzx':
            return self.forward_qzx(input, latent_rand_sampling=latent_rand_sampling, L=L)
        elif self.z_source == 'Ex':
            return self.forward_Ex(input, latent_rand_sampling=latent_rand_sampling, L=L)
        else:
            print('Invalid z_source')
            exit(1)

    def forward_ae(self, input):
        z, _ = self.encode(input)
        return self.decode(z), z, 0.0, z, 0.0
    
    def forward_Ex(self, input, latent_rand_sampling=True, L=1): # Latent reconstruction, z is encoded from x
        mu, log_var = self.encode(input)
        if self.fixed_var != False:
            log_var = torch.log(torch.ones_like(log_var) * self.fixed_var)
        if latent_rand_sampling:
            z = mu + torch.randn_like(mu) * torch.exp(log_var * 0.5)
        else:
            z = mu
        recon = self.decode(z)
        z_recon, _ = self.encode(recon)
        return recon, mu, log_var, z, z_recon
    
    def forward_qzx(self, input, latent_rand_sampling=True, L=1): # Latent reconstruction, z is encoded from x, z is reconstructed to mu
        mu, log_var = self.encode(input)
        if self.fixed_var != False:
            log_var = torch.log(torch.ones_like(log_var) * self.fixed_var)
        if latent_rand_sampling:
            z = mu + torch.randn_like(mu) * torch.exp(log_var * 0.5)
        else:
            z = mu
        recon = self.decode(z)
        z_recon, _ = self.encode(recon)
        return recon, mu, log_var, mu, z_recon

    def forward_pz(self, input, latent_rand_sampling=True, L=1): # Latent reconstruction, z is sampled from p(z)
        mu, log_var = self.encode(input)
        if self.fixed_var != False:
            log_var = torch.log(torch.ones_like(log_var) * self.fixed_var)
        if latent_rand_sampling:
            z = mu + torch.randn_like(mu) * torch.exp(log_var * 0.5)
        else:
            z = mu
        z_input = torch.randn_like(mu) * torch.exp(torch.ones_like(log_var) * 0.5)
        z_recon, _ = self.encode(self.decode(z_input))
        return self.decode(z), mu, log_var, z_input, z_recon

    


class NaiveAE(FlexibleVAE):
    def __init__(self, **kwargs):
        kwargs['variational'] = False
        super(NaiveAE, self).__init__(**kwargs)

    def loss(self, input, output, mu, log_var, z_input=None, z_recon=None):
        loss_recon = (
            ((input - output) ** 2).mean(dim=0).sum()
            if not self.is_log_mse
            else (
                0.5
                * torch.ones_like(input[0]).sum()
                * (
                    (
                        2 * torch.pi * ((input - output) ** 2).mean(1).mean(1).mean(1)
                        + 1e-5  # To avoid log(0)
                    ).log()
                    + 1
                )
            ).mean()
        )

        return loss_recon, loss_recon.detach(), 0.0, 0.0

class VanillaVAE(FlexibleVAE):
    def __init__(self, **kwargs):
        super(VanillaVAE, self).__init__(**kwargs)

    #def loss(self, input, output, mu, log_var, z_input=None, z_recon=None, L=1):
    #    if L == 1:
    #        return self.loss_naive(input, output, mu, log_var, z_input, z_recon)
    #    else:
    #        return self.loss_mc(input, output, mu, log_var, z_input, z_recon)

    def loss(self, input, output, mu, log_var, z_input=None, z_recon=None):
        loss_recon = (
            ((input - output) ** 2).mean(dim=0).sum()
            if not self.is_log_mse
            else (0.5 * torch.ones_like(input[0]).sum()
                * (( 2 * torch.pi * ((input - output) ** 2).mean(1).mean(1).mean(1)
                        + 1e-5  # To avoid log(0)
                    ).log() + 1)
            ).mean()
        )
        loss_reg = (-0.5 * (1 + log_var - mu**2 - log_var.exp())).mean(dim=0).sum()
        loss_lr = ((z_input - z_recon) ** 2).mean(dim=0).sum()

        return loss_recon + loss_reg * self.beta, loss_recon.detach(), loss_reg.detach(), loss_lr.detach()


    #def loss_mc(self, input, output, mu, log_var, z_input, z_recon): # output:[L,B,C,H,W], z_input:[L,B,D], z_recon:[L,B,D]
    #    loss_recon = (
    #        ((input.unsqueeze(0) - output) ** 2).mean(dim=0).sum()
    #        if not self.is_log_mse
    #        else (0.5 * torch.ones_like(input[0]).sum()
    #            * (( 2 * torch.pi * ((input.unsqueeze(0) - output) ** 2).mean(2).mean(2).mean(2)
    #                    + 1e-5  # To avoid log(0)
    #                ).log() + 1)
    #        ).mean()
    #    )
    #    loss_reg = (-0.5 * (1 + log_var - mu**2 - log_var.exp())).mean(dim=0).sum()
    #    loss_lr = ((z_input - z_recon) ** 2).mean(dim=0).sum()

    #    return loss_recon + loss_reg * self.beta + loss_lr * self.alpha * self.wu_alpha, loss_recon.detach(), loss_reg.detach(), loss_lr.detach()



class LRVAE(FlexibleVAE):
    def __init__(self, alpha=0.01, **kwargs):
        super(LRVAE, self).__init__(**kwargs)
        self.alpha = alpha

    def warmup(self, epoch, max_epoch, wu_strat='linear', up_amount=None, start_epoch=0, repeat_interval=10):
        if wu_strat == 'linear':
            if epoch >= start_epoch:
                if up_amount == None:
                    self.wu_alpha = min(self.wu_alpha + 1.0/(max_epoch-start_epoch+1), 1.0)
                else:
                    self.wu_alpha = min(self.wu_alpha + up_amount, 1.0)
        elif wu_strat == 'exponential':
            if epoch >= start_epoch:
                if up_amount == None: # exponential function 0.0 at start_epoch and 1.0 at max_epoch
                    x = (epoch-start_epoch)*math.log(2)/(max_epoch-start_epoch)
                    self.wu_alpha = max(min(math.exp(x)-1.0, 1.0), 0.0)
                else:
                    x = up_amount*(epoch-start_epoch)
                    self.wu_alpha = max(min(math.exp(x)-1.0, 1.0), 0.0)
        elif wu_strat == 'repeat_linear':
            if epoch >= start_epoch:
                self.wu_alpha = min(1.0/((epoch%repeat_interval)+1), 1.0)
        return True


    #def loss(self, input, output, mu, log_var, z_input, z_recon, L=1):
    #    if L == 1:
    #        return self.loss_naive(input, output, mu, log_var, z_input, z_recon)
    #    else:
    #        return self.loss_mc(input, output, mu, log_var, z_input, z_recon)


    def loss(self, input, output, mu, log_var, z_input, z_recon):
        loss_recon = (
            ((input - output) ** 2).mean(dim=0).sum()
            if not self.is_log_mse
            else (
                0.5
                * torch.ones_like(input[0]).sum()
                * (
                    (
                        2 * torch.pi * ((input - output) ** 2).mean(1).mean(1).mean(1)
                        + 1e-5  # To avoid log(0)
                    ).log()
                    + 1
                )
            ).mean()
        )
        loss_lr = ((z_input - z_recon) ** 2).mean(dim=0).sum()

        
        loss_reg = (-0.5 * (1 + log_var - mu**2 - log_var.exp())).mean(dim=0).sum()

        if self.pwise_reg:
            mu_zp = z_input.mean(dim=1, keepdim=True)
            logvar_zp = torch.log(((z_input - mu_zp) ** 2).mean(dim=1))
            loss_reg = loss_reg/2.0 + (-0.5 * (1 + logvar_zp - mu_zp**2 - logvar_zp.exp())).mean(dim=1).sum()/2.0

        return loss_recon + loss_reg * self.beta + loss_lr * self.alpha * self.wu_alpha, loss_recon, loss_reg * self.beta, loss_lr * self.alpha * self.wu_alpha

    #def loss_mc(self, input, output, mu, log_var, z_input, z_recon):
        # output, z_input, z_recon are stacked tensors with shape [L, batch_size, ...]
        # loss_recon = (
        #     ((input.unsqueeze(0) - output) ** 2).mean(dim=1).sum()
        #     if not self.is_log_mse
        #     else (0.5 * torch.ones_like(input[0]).sum()
        #         * (( 2 * torch.pi * ((input.unsqueeze(0) - output) ** 2).mean(2).mean(2).mean(2)
        #                 + 1e-5  # To avoid log(0)
        #             ).log() + 1)
        #     ).mean()
        # ).mean(0)  # Average over L samples

        # loss_reg = (-0.5 * (1 + log_var - mu**2 - log_var.exp())).mean(dim=0).sum()
        # loss_lr = ((z_input - z_recon) ** 2).mean(dim=1).sum().mean(0)  # Average over L samples

        # return loss_recon + loss_reg * self.beta + loss_lr * self.alpha * self.wu_alpha, loss_recon.detach(), loss_reg.detach(), loss_lr.detach()



class LIDVAE(VAE):

    def __init__(
        self,
        in_channel=1,
        latent_channel=32,
        hidden_channels=None,
        icnn_channels=[512, 1024],
        input_dim=28,
        inverse_lipschitz=0.0,
        beta=1.0,
        is_log_mse=False,
        dataset=None,
    ):
        """
        LIDVAE with residual-conv encoder and Brenier map decoder, for image dataset.
        Decoder consists of 2 ICNN, so 2-length array is expected for hidden channels of ICNNs.
        See Wang et al. for details on Brenier map.
        Inverse Lipschitz, Beta, and logMSE features are ready-to-use, but are disabled by default.
        """
        if len(icnn_channels) != 2:
            raise ValueError("2-length array was expected for `icnn_channels`")

        if dataset == "celeba":
            in_channel = 3
            latent_channel = 64
            hidden_channels = [32, 64, 128, 256] if hidden_channels == None else hidden_channels
            input_dim = 64
        elif (dataset == "mnist") or (dataset == "fashionmnist"):
            in_channel = 1
            latent_channel = 32
            hidden_channels = [32, 64, 128] if hidden_channels == None else hidden_channels
            input_dim = 28
        elif dataset == "cifar10":
            in_channel = 3
            latent_channel = 128
            hidden_channels = [32, 64, 128, 256] if hidden_channels == None else hidden_channels
            input_dim = 32
        elif dataset == "omniglot":
            in_channel = 1
            latent_channel = 32
            hidden_channels = [32, 64, 128] if hidden_channels == None else hidden_channels
            input_dim = 28
        elif dataset == "pinwheel" or dataset == "chessboard":
            in_channel = 2
            latent_channel = 2
            hidden_channels = [2, 2, 2, 2] if hidden_channels == None else hidden_channels
            input_dim = 1
            data_type = '1d'
        else:
            raise ValueError(f"Invalid dataset: {dataset}")

        super(VAE, self).__init__()

        self.latent_channel = latent_channel
        self.il_factor = inverse_lipschitz / 2.0
        self.beta = beta
        self.is_log_mse = is_log_mse

        fc_dim = input_dim
        transpose_padding = []
        for _ in range(len(hidden_channels)):
            transpose_padding.append((fc_dim + 1) % 2)
            fc_dim = (fc_dim - 1) // 2 + 1
        transpose_padding.reverse()

        if data_type == '1d':
            self.encoder = self.make_encoder_1d(hidden_channels, in_channel, latent_channel)
            self.decoder = self.make_decoder_1d(in_channel, latent_channel, icnn_channels, input_dim)
        else:
            self.encoder = self.make_encoder_2d(hidden_channels, in_channel, latent_channel, fc_dim)
            self.decoder = self.make_decoder_2d(in_channel, latent_channel, icnn_channels, input_dim)


    def make_encoder_1d(self, hidden_channels, in_channel, latent_channel):
        # Make encoder
        encoder = []
        last_channel = in_channel

        for channel in hidden_channels:
            encoder.append(
                torch.nn.Sequential(
                    torch.nn.Linear(last_channel, channel),
                    torch.nn.BatchNorm1d(channel),
                    torch.nn.LeakyReLU(),
                )
            )
            last_channel = channel

        encoder.append(
            torch.nn.Sequential(
                torch.nn.Linear(last_channel, latent_channel * 2),
                torch.nn.BatchNorm1d(latent_channel * 2),
                torch.nn.LeakyReLU(),
                torch.nn.Linear(latent_channel * 2, latent_channel * 2),
            )
        )
        return torch.nn.Sequential(*encoder)

    def make_encoder_2d(self, hidden_channels, in_channel, latent_channel, fc_dim):
        # Make encoder
        encoder = []
        last_channel = in_channel

        for channel in hidden_channels:
            encoder.append(
                torch.nn.Sequential(
                    module.ResidualConvBlock(last_channel, channel, 2),
                    module.ResidualConvBlock(channel, channel, 1),
                )
            )
            last_channel = channel

        encoder.append(
            torch.nn.Sequential(
                torch.nn.Flatten(),
                torch.nn.Linear(last_channel * (fc_dim**2), latent_channel * 2),
                torch.nn.BatchNorm1d(latent_channel * 2),
                torch.nn.LeakyReLU(),
                torch.nn.Linear(latent_channel * 2, latent_channel * 2),
            )
        )
        return torch.nn.Sequential(*encoder)
    
    def make_decoder_1d(self, in_channel, latent_channel, icnn_channels, input_dim):
        # Make decoder
        decoder = []

        # First layer: ICNN in latent channel
        decoder.append(module.ICNN(latent_channel, icnn_channels[0]))

        # In the original implmentation,
        # a trainable full-rank matrix is used as Beta via SVD (as in appendix)
        # Here, we use an identity matrix for injective map (as in main text)
        self.register_buffer(
            "B",
            torch.eye((input_dim) * in_channel, latent_channel, requires_grad=False),
        )

        # Second and last layer: ICNN in data dimension
        decoder.append(module.ICNN((input_dim) * in_channel, icnn_channels[1]))

        # dummy to fit index
        decoder.append(torch.nn.Identity())

        # Note that there is no range mapping!
        # Use Clip or Sigmoid if you want
        return torch.nn.ModuleList(decoder)

    def make_decoder_2d(self, in_channel, latent_channel, icnn_channels, input_dim):
        # Make decoder
        decoder = []

        # First layer: ICNN in latent channel
        decoder.append(module.ICNN(latent_channel, icnn_channels[0]))

        # In the original implmentation,
        # a trainable full-rank matrix is used as Beta via SVD (as in appendix)
        # Here, we use an identity matrix for injective map (as in main text)
        self.register_buffer(
            "B",
            torch.eye((input_dim**2) * in_channel, latent_channel, requires_grad=False),
        )

        # Second and last layer: ICNN in data dimension
        decoder.append(module.ICNN((input_dim**2) * in_channel, icnn_channels[1]))

        # Unflatten to shape of image
        decoder.append(torch.nn.Unflatten(1, (in_channel, input_dim, input_dim)))

        # Note that there is no range mapping!
        # Use Clip or Sigmoid if you want
        return torch.nn.ModuleList(decoder)


    def encode(self, input):
        ret = self.encoder(input)
        #return ret.split(ret.shape[1] // 2, 1)
        mu, var = ret.split(ret.shape[1] // 2, 1)
        return mu, F.softplus(var) # prevent var from being negative

    def decode(self, input):
        # x is result of first ICNN
        x = self.decoder[0](input) + self.il_factor * input.pow(2).sum(1, keepdim=True)
        # x is result of brenier map
        x = torch.autograd.grad(x, [input], torch.ones_like(x), create_graph=True)[0]
        # x is result of Beta (id mat)
        x = torch.nn.functional.linear(x, self.B)
        # y is result of second ICNN
        y = self.decoder[1](x) + self.il_factor * x.pow(2).sum(1, keepdim=True)
        # y is result of brenier map
        y = torch.autograd.grad(y, [x], torch.ones_like(y), create_graph=True)[0]

        return self.decoder[2](y)

   
    def forward(self, input, latent_recon=False, latent_rand_sampling=True):
        if latent_recon:
            return self.forward_Ex(input, latent_rand_sampling=latent_rand_sampling) # should be equal to lrvae source
            # return self.forward_Ex(input)
        else:
            return self.forward_vae(input, latent_rand_sampling=latent_rand_sampling)

    def forward_vae(self, input, latent_rand_sampling=True):
        mu, log_var = self.encode(input)
        if latent_rand_sampling:
            z = mu + torch.randn_like(mu) * torch.exp(log_var * 0.5)
        else:
            z = mu
        return self.decode(z), mu, log_var, z, None
    
    def forward_Ex(self, input, latent_rand_sampling=True): # Latent reconstruction, z is encoded from x
        mu, log_var = self.encode(input)
        if latent_rand_sampling:
            z = mu + torch.randn_like(mu) * torch.exp(log_var * 0.5)
        else:
            z = mu
        recon = self.decode(z)
        z_recon, _ = self.encode(recon)
        return recon, mu, log_var, z, z_recon
    
    def forward_qzx(self, input, latent_rand_sampling=True): # Latent reconstruction, z is encoded from x, z is reconstructed to mu
        mu, log_var = self.encode(input)
        if latent_rand_sampling:
            z = mu + torch.randn_like(mu) * torch.exp(log_var * 0.5)
        else:
            z = mu
        recon = self.decode(z)
        z_recon, _ = self.encode(recon)
        return recon, mu, log_var, mu, z_recon

    def loss(self, input, output, mu, log_var, z_input=None, z_recon=None):
        loss_recon = (
            ((input - output) ** 2).mean(dim=0).sum()
            if not self.is_log_mse
            else (
                0.5
                * torch.ones_like(input[0]).sum()
                * (
                    (
                        2 * torch.pi * ((input - output) ** 2).reshape(input.size(0), -1).mean(dim=1)
                        + 1e-5  # To avoid log(0)
                    ).log()
                    + 1
                )
            ).mean()
        )
        loss_reg = (-0.5 * (1 + log_var - mu**2 - log_var.exp())).mean(dim=0).sum()

        return loss_recon + loss_reg * self.beta, loss_recon.detach(), loss_reg.detach(), 0.0


########################################
# SetVAE for 3D point cloud (inspired by SetVAE)
# - Permutation-invariant encoder (DeepSets-style)
# - Fixed-length set decoder conditioned on latent code
# - Chamfer distance for reconstruction loss
########################################

def chamfer_distance(points_pred, points_gt):
    """
    Compute symmetric Chamfer distance between predicted and ground-truth point sets.
    points_pred: Tensor [B, Np, 3]
    points_gt:   Tensor [B, Ng, 3]
    Returns: scalar Tensor
    """
    # Use squared Euclidean distances
    # torch.cdist -> [B, Np, Ng]
    dist = torch.cdist(points_pred, points_gt, p=2)  # L2 distance
    dist2 = dist.pow(2)
    # For each point in pred, find closest in gt
    min_pred_to_gt, _ = dist2.min(dim=2)  # [B, Np]
    # For each point in gt, find closest in pred
    min_gt_to_pred, _ = dist2.min(dim=1)  # [B, Ng]
    cd = min_pred_to_gt.mean(dim=1) + min_gt_to_pred.mean(dim=1)  # [B]
    return cd.mean()


class SetEncoder(nn.Module):
    def __init__(self, point_dim=3, hidden_dims=[128, 256, 512], latent_dim=128, pool_type='max'):
        super().__init__()
        self.pool_type = pool_type
        layers = []
        last = point_dim
        for h in hidden_dims:
            layers.append(nn.Sequential(
                nn.Linear(last, h),
                nn.BatchNorm1d(h),
                nn.ReLU(),
            ))
            last = h
        self.phi = nn.ModuleList(layers)
        self.fc_mu = nn.Linear(last, latent_dim)
        self.fc_logvar = nn.Linear(last, latent_dim)

    def forward(self, points):
        # points: [B, N, 3]
        B, N, D = points.shape
        x = points.view(B * N, D)
        for layer in self.phi:
            x = layer(x)
        x = x.view(B, N, -1)
        if self.pool_type == 'mean':
            s = x.mean(dim=1)
        elif self.pool_type == 'sum':
            s = x.sum(dim=1)
        else:  # max
            s, _ = x.max(dim=1)
        mu = self.fc_mu(s)
        log_var = self.fc_logvar(s)
        return mu, log_var


class SetEncoderAttn(nn.Module):
    def __init__(self, point_dim=3, latent_dim=128, d_model=256, num_heads=4, num_layers=2, ff_dim=512, dropout=0.0):
        super().__init__()
        self.input_proj = nn.Linear(point_dim, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=num_heads, dim_feedforward=ff_dim, dropout=dropout, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.pool = nn.AdaptiveMaxPool1d(1)
        self.fc_mu = nn.Linear(d_model, latent_dim)
        self.fc_logvar = nn.Linear(d_model, latent_dim)

    def forward(self, points):
        # points: [B, N, 3]
        x = self.input_proj(points)  # [B, N, d_model]
        x = self.encoder(x)          # [B, N, d_model]
        # permutation-invariant pooling (max)
        x = x.transpose(1, 2)        # [B, d_model, N]
        s = self.pool(x).squeeze(-1) # [B, d_model]
        mu = self.fc_mu(s)
        log_var = self.fc_logvar(s)
        return mu, log_var


class SetDecoderAttn(nn.Module):
    def __init__(self, latent_dim=128, num_points=2048, d_model=256, num_heads=4, num_layers=2, ff_dim=512, dropout=0.0):
        super().__init__()
        self.num_points = num_points
        self.query_embed = nn.Parameter(torch.randn(num_points, d_model) * 0.02)
        self.latent_to_token = nn.Linear(latent_dim, d_model)
        decoder_layer = nn.TransformerDecoderLayer(d_model=d_model, nhead=num_heads, dim_feedforward=ff_dim, dropout=dropout, batch_first=True)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.output_proj = nn.Linear(d_model, 3)

    def forward(self, z):
        # z: [B, D]
        B, D = z.shape
        # make memory (context) token from latent
        memory = self.latent_to_token(z).unsqueeze(1)  # [B, 1, d_model]
        # queries
        queries = self.query_embed.unsqueeze(0).expand(B, -1, -1)  # [B, N, d_model]
        # transformer decode: queries attend to memory
        x = self.decoder(tgt=queries, memory=memory)  # [B, N, d_model]
        points = self.output_proj(x)  # [B, N, 3]
        return points

class SetDecoder(nn.Module):
    def __init__(self, latent_dim=128, num_points=2048, hidden_dims=[512, 256, 128], point_dim=3):
        super().__init__()
        self.num_points = num_points
        # Learnable per-point queries
        self.point_queries = nn.Parameter(torch.randn(num_points, 64) * 0.02)
        input_dim = latent_dim + 64

        layers = []
        last = input_dim
        for h in hidden_dims:
            layers.append(nn.Sequential(
                nn.Linear(last, h),
                nn.BatchNorm1d(h),
                nn.ReLU(),
            ))
            last = h
        layers.append(nn.Linear(last, point_dim))
        self.mlp = nn.ModuleList(layers)

    def forward(self, z):
        # z: [B, D]
        B, D = z.shape
        queries = self.point_queries.unsqueeze(0).expand(B, -1, -1)  # [B, N, 64]
        z_expanded = z.unsqueeze(1).expand(-1, self.num_points, -1)  # [B, N, D]
        x = torch.cat([z_expanded, queries], dim=-1)  # [B, N, D+64]
        x = x.reshape(B * self.num_points, -1)
        for layer in self.mlp[:-1]:
            x = layer(x)
        points = self.mlp[-1](x)
        points = points.view(B, self.num_points, -1)  # [B, N, 3]
        return points


class SetVAE(VAE):
    def __init__(
        self,
        latent_channel=128,
        num_points=2048,
        encoder_hidden=[128, 256, 512],
        decoder_hidden=[512, 256, 128],
        beta=1.0,
        is_log_mse=False,  # unused for set, kept for API compat
        dataset='shapenet',
        pool_type='max',
        use_attention=True,
        d_model=256,
        num_heads=4,
        num_encoder_layers=2,
        num_decoder_layers=2,
        ff_dim=512,
        attn_dropout=0.0,
    ):
        super(SetVAE, self).__init__()
        self.latent_channel = latent_channel
        self.beta = beta
        self.is_log_mse = is_log_mse
        self.num_points = num_points
        self.data_type = 'set'  # to disable 2D image visualization

        if use_attention:
            self.encoder = SetEncoderAttn(point_dim=3, latent_dim=latent_channel, d_model=d_model, num_heads=num_heads,
                                          num_layers=num_encoder_layers, ff_dim=ff_dim, dropout=attn_dropout)
            self.decoder = SetDecoderAttn(latent_dim=latent_channel, num_points=num_points, d_model=d_model, num_heads=num_heads,
                                          num_layers=num_decoder_layers, ff_dim=ff_dim, dropout=attn_dropout)
        else:
            self.encoder = SetEncoder(point_dim=3, hidden_dims=encoder_hidden, latent_dim=latent_channel, pool_type=pool_type)
            self.decoder = SetDecoder(latent_dim=latent_channel, num_points=num_points, hidden_dims=decoder_hidden, point_dim=3)

    def encode(self, input):
        return self.encoder(input)

    def decode(self, input):
        return self.decoder(input)

    def forward(self, input, latent_rand_sampling=True, L=1):
        # input: [B, N, 3]
        mu, log_var = self.encode(input)
        if latent_rand_sampling:
            z = mu + torch.randn_like(mu) * torch.exp(log_var * 0.5)
        else:
            z = mu
        recon = self.decode(z)
        # For set models, latent reconstruction path is optional; return None placeholders
        return recon, mu, log_var, z, None

    def loss(self, input, output, mu, log_var, z_input=None, z_recon=None):
        # Chamfer distance as reconstruction loss
        loss_recon = chamfer_distance(output, input)
        loss_reg = (-0.5 * (1 + log_var - mu**2 - log_var.exp())).mean(dim=0).sum()
        return loss_recon + self.beta * loss_reg, loss_recon.detach(), loss_reg.detach(), torch.tensor(0.0, device=input.device)


class SetLRVAE(SetVAE):
    def __init__(self, alpha=0.01, **kwargs):
        super().__init__(**kwargs)
        self.alpha = alpha
        self.wu_alpha = 0.0

    def warmup(self, epoch, max_epoch, wu_strat='linear', up_amount=None, start_epoch=0, repeat_interval=10):
        if wu_strat == 'linear':
            if epoch >= start_epoch:
                if up_amount is None:
                    self.wu_alpha = min(self.wu_alpha + 1.0 / (max_epoch - start_epoch + 1), 1.0)
                else:
                    self.wu_alpha = min(self.wu_alpha + up_amount, 1.0)
        elif wu_strat == 'exponential':
            if epoch >= start_epoch:
                if up_amount is None:
                    x = (epoch - start_epoch) * math.log(2) / (max_epoch - start_epoch)
                    self.wu_alpha = max(min(math.exp(x) - 1.0, 1.0), 0.0)
                else:
                    x = up_amount * (epoch - start_epoch)
                    self.wu_alpha = max(min(math.exp(x) - 1.0, 1.0), 0.0)
        elif wu_strat == 'repeat_linear':
            if epoch >= start_epoch:
                self.wu_alpha = min(1.0 / ((epoch % repeat_interval) + 1), 1.0)
        return True

    def forward(self, input, latent_rand_sampling=True, L=1):
        mu, log_var = self.encode(input)
        if latent_rand_sampling:
            z = mu + torch.randn_like(mu) * torch.exp(log_var * 0.5)
        else:
            z = mu
        recon = self.decode(z.detach())  # detach for latent recon pathway
        z_recon, _ = self.encode(recon)
        # return z (as z_input) and z_recon for latent recon loss
        return recon, mu, log_var, z, z_recon

    def loss(self, input, output, mu, log_var, z_input, z_recon):
        loss_recon = chamfer_distance(output, input)
        loss_reg = (-0.5 * (1 + log_var - mu**2 - log_var.exp())).mean(dim=0).sum()
        loss_lr = ((z_input - z_recon) ** 2).mean(dim=0).sum()
        total = loss_recon + self.beta * loss_reg + self.alpha * self.wu_alpha * loss_lr
        return total, loss_recon.detach(), (self.beta * loss_reg).detach(), (self.alpha * self.wu_alpha * loss_lr).detach()
