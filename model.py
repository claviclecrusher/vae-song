import torch
import module
import torch.nn.functional as F
import torch.nn.init as init


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

    def make_encoder_mlp_1d(self, hidden_channels, in_channel, latent_channel):
        last_channel = in_channel
        encoder = []
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
                #torch.nn.Linear(last_channel, latent_channel * 2),
                #torch.nn.BatchNorm1d(latent_channel * 2),
                #torch.nn.LeakyReLU(),
                torch.nn.Linear(last_channel, latent_channel * 2),
            )
        )
        encoder = torch.nn.Sequential(*encoder)
        return encoder

    def make_encoder_residual_mlp_1d(self, hidden_channels, in_channel, latent_channel):
        last_channel = in_channel
        encoder = []
        for channel in hidden_channels:
            encoder.append(
                torch.nn.Sequential(
                    module.ResidualMLPBlock(last_channel, channel),
                )
            )
            last_channel = channel

        encoder.append(
            torch.nn.Sequential(
                module.ResidualMLPBlock(last_channel, latent_channel * 2),
            )
        )
        encoder = torch.nn.Sequential(*encoder)
        return encoder
    
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
        mu, var = ret.split(ret.shape[1] // 2, 1)
        return mu, F.softplus(var) # prevent var from being negative

    def decode(self, input):
        return self.decoder(input)


    def forward(self, input, latent_rand_sampling=True, L=1): # L: number of samples for MC
        mu, log_var = self.encode(input)

        if latent_rand_sampling:
            # [L, B, D] 형태로 한번에 랜덤 샘플 생성
            eps = torch.randn(L, *mu.shape, device=mu.device)
            input_z_stack = mu.unsqueeze(0) + eps * torch.exp(log_var * 0.5).unsqueeze(0)
        else:
            input_z_stack = mu.unsqueeze(0)

        # [L, B, D] -> [L*B, D] 형태로 변환하여 한번에 디코딩
        B = input.shape[0]
        input_z_flat = input_z_stack.view(-1, input_z_stack.shape[-1])
        recon_flat = self.decode(input_z_flat)
        recon_stack = recon_flat.view(L, B, *recon_flat.shape[1:])

        # [L*B, D] -> [L, B, D] 형태로 변환하여 한번에 인코딩
        recon_flat_for_lr = self.decode(input_z_flat.detach()) # 첫 encoder에 영향을 주지 않기 위해서 detach
        z_recon_flat, _ = self.encode(recon_flat_for_lr)
        z_recon_stack = z_recon_flat.view(L, B, *z_recon_flat.shape[1:])

        #z_for_lr_stack = input_z_flat.detach().view(L, B, *input_z_flat.shape[1:]) # reshape to match z_recon_stack

        return recon_stack.mean(dim=0), mu, log_var, input_z_stack, z_recon_stack # recon은 평균값을 취하고 z에 대해서는 각각 계산


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

