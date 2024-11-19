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

    def forward(self, input):
        raise NotImplementedError

    def loss(self, *args):
        raise NotImplementedError

    def warmup(self, epoch, amount=None):
        return False


class VanillaVAE(VAE):

    def __init__(
        self,
        in_channel=1,
        latent_channel=32,
        hidden_channels=[32, 64, 128],
        input_dim=28,
        beta=1.0,
        is_log_mse=False,
        dataset=None,
    ):
        """
        Conventional VAE with residual-conv encoder and MLP decoder, for image dataset.
        Note that decoder is 4-layer MLP, to avoid using convolution and its transpose.
        Beta and logMSE features are ready-to-use, but are disabled by default.
        """
        if dataset == "celeba":
            in_channel = 3
            latent_channel = 64
            hidden_channels = [32, 64, 128, 256]
            input_dim = 64
        elif dataset == "mnist" or "fashionmnist":
            in_channel = 1
            latent_channel = 32
            hidden_channels = [32, 64, 128]
            input_dim = 28

        super(VAE, self).__init__()

        self.latent_channel = latent_channel
        self.beta = beta
        self.is_log_mse = is_log_mse

        fc_dim = input_dim
        transpose_padding = []
        for _ in range(len(hidden_channels)):
            transpose_padding.append((fc_dim + 1) % 2)
            fc_dim = (fc_dim - 1) // 2 + 1
        transpose_padding.reverse()

        # Make encoder
        self.encoder = []
        last_channel = in_channel

        for channel in hidden_channels:
            self.encoder.append(
                torch.nn.Sequential(
                    module.ResidualBlock(last_channel, channel, 2),
                    module.ResidualBlock(channel, channel, 1),
                )
            )
            last_channel = channel

        self.encoder.append(
            torch.nn.Sequential(
                torch.nn.Flatten(),
                torch.nn.Linear(last_channel * (fc_dim**2), latent_channel * 2),
                torch.nn.BatchNorm1d(latent_channel * 2),
                torch.nn.LeakyReLU(),
                torch.nn.Linear(latent_channel * 2, latent_channel * 2),
            )
        )
        self.encoder = torch.nn.Sequential(*self.encoder)

        # Make decoder
        self.decoder = []

        # First layer: half of final dimension
        last_channel = latent_channel
        channel = (input_dim**2) * in_channel // 2
        self.decoder.append(
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
        self.decoder.append(
            torch.nn.Sequential(
                torch.nn.Linear(last_channel, channel),
                torch.nn.BatchNorm1d(channel),
                torch.nn.LeakyReLU(),
                torch.nn.Linear(channel, channel),
            )
        )

        # Unflatten to shape of image
        self.decoder.append(torch.nn.Unflatten(1, (in_channel, input_dim, input_dim)))

        # Note that there is no range mapping!
        # Use Clip or Sigmoid if you want
        self.decoder = torch.nn.Sequential(*self.decoder)

    def encode(self, input):
        ret = self.encoder(input)
        #return ret.split(ret.shape[1] // 2, 1)
        mu, var = ret.split(ret.shape[1] // 2, 1)
        if self.beta == 0.0:
            return mu, torch.zeros_like(var)
        return mu, F.softplus(var) # prevent var from being negative

    def decode(self, input):
        return self.decoder(input)
    
    def forward(self, input, latent_recon=False):
        if latent_recon:
            return self.forward_qzx(input) # should be equal to lrvae source
            # return self.forward_Ex(input)
        else:
            return self.forward_vae(input)

    def forward_vae(self, input):
        mu, log_var = self.encode(input)
        z = mu + torch.randn_like(mu) * torch.exp(log_var * 0.5)
        return self.decode(z), mu, log_var
    
    def forward_Ex(self, input): # Latent reconstruction, z is encoded from x
        mu, log_var = self.encode(input)
        z = mu + torch.randn_like(mu) * torch.exp(log_var * 0.5)
        recon = self.decode(z)
        z_recon, _ = self.encode(recon)
        return recon, mu, log_var, z, z_recon
    
    def forward_qzx(self, input): # Latent reconstruction, z is encoded from x, z is reconstructed to mu
        mu, log_var = self.encode(input)
        z = mu + torch.randn_like(mu) * torch.exp(log_var * 0.5)
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
                        2 * torch.pi * ((input - output) ** 2).mean(1).mean(1).mean(1)
                        + 1e-5  # To avoid log(0)
                    ).log()
                    + 1
                )
            ).mean()
        )
        loss_reg = (-0.5 * (1 + log_var - mu**2 - log_var.exp())).mean(dim=0).sum()

        return loss_recon + loss_reg * self.beta, loss_recon.detach(), loss_reg.detach(), 0.0


class LRVAE(VAE):

    def __init__(
        self,
        in_channel=1,
        latent_channel=32,
        hidden_channels=[32, 64, 128],
        icnn_channels=[512, 1024],
        input_dim=28,
        beta=1.0,
        alpha=0.01,
        is_log_mse=False,
        dataset=None,
        z_source='qzx',
        bal_alpha=True
    ):
        """
        Latent Reconstruction VAE with residual-conv encoder and MLP decoder, for image dataset.
        """
        if dataset == "celeba":
            in_channel = 3
            latent_channel = 64
            hidden_channels = [32, 64, 128, 256]
            input_dim = 64
        elif dataset == "mnist" or "fashionmnist":
            in_channel = 1
            latent_channel = 32
            hidden_channels = [32, 64, 128]
            input_dim = 28

        super(VAE, self).__init__()

        self.latent_channel = latent_channel
        self.beta = beta
        self.alpha = alpha
        self.z_source = z_source
        self.wu_alpha = 0.0
        self.is_log_mse = is_log_mse
        self.balanced_alpha = bal_alpha

        fc_dim = input_dim
        transpose_padding = []
        for _ in range(len(hidden_channels)):
            transpose_padding.append((fc_dim + 1) % 2)
            fc_dim = (fc_dim - 1) // 2 + 1
        transpose_padding.reverse()

        # Make encoder
        self.encoder = []
        last_channel = in_channel

        for channel in hidden_channels:
            self.encoder.append(
                torch.nn.Sequential(
                    module.ResidualBlock(last_channel, channel, 2),
                    module.ResidualBlock(channel, channel, 1),
                )
            )
            last_channel = channel

        self.encoder.append(
            torch.nn.Sequential(
                torch.nn.Flatten(),
                torch.nn.Linear(last_channel * (fc_dim**2), latent_channel * 2),
                torch.nn.BatchNorm1d(latent_channel * 2),
                torch.nn.LeakyReLU(),
                torch.nn.Linear(latent_channel * 2, latent_channel * 2),
            )
        )
        self.encoder = torch.nn.Sequential(*self.encoder)

        # Make decoder
        self.decoder = []

        # First layer: half of final dimension
        last_channel = latent_channel
        channel = (input_dim**2) * in_channel // 2
        self.decoder.append(
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
        self.decoder.append(
            torch.nn.Sequential(
                torch.nn.Linear(last_channel, channel),
                torch.nn.BatchNorm1d(channel),
                torch.nn.LeakyReLU(),
                torch.nn.Linear(channel, channel),
            )
        )

        # Unflatten to shape of image
        self.decoder.append(torch.nn.Unflatten(1, (in_channel, input_dim, input_dim)))

        # Note that there is no range mapping!
        # Use Clip or Sigmoid if you want
        self.decoder = torch.nn.Sequential(*self.decoder)

    def encode(self, input):
        ret = self.encoder(input)
        #return ret.split(ret.shape[1] // 2, 1)
        mu, var = ret.split(ret.shape[1] // 2, 1)
        return mu, F.softplus(var) # prevent var from being negative

    def decode(self, input):
        return self.decoder(input)

    def forward(self, input, latent_recon=True):
        if self.z_source == 'pz':
            return self.forward_pz(input)
        elif self.z_source == 'qzx':
            return self.forward_qzx(input)
        elif self.z_source == 'Ex':
            return self.forward_Ex(input)
        else:
            print('Invalid z_source')
            exit(1)
    
    def forward_Ex(self, input): # Latent reconstruction, z is encoded from x
        mu, log_var = self.encode(input)
        z = mu + torch.randn_like(mu) * torch.exp(log_var * 0.5)
        recon = self.decode(z)
        z_recon, _ = self.encode(recon)
        return recon, mu, log_var, z, z_recon
    
    def forward_qzx(self, input): # Latent reconstruction, z is encoded from x, z is reconstructed to mu
        mu, log_var = self.encode(input)
        z = mu + torch.randn_like(mu) * torch.exp(log_var * 0.5)
        recon = self.decode(z)
        z_recon, _ = self.encode(recon)
        return recon, mu, log_var, mu, z_recon

    def forward_pz(self, input): # Latent reconstruction, z is sampled from p(z)
        mu, log_var = self.encode(input)
        z = mu + torch.randn_like(mu) * torch.exp(log_var * 0.5)
        z_input = torch.randn_like(mu) * torch.exp(torch.ones_like(log_var) * 0.5)
        z_recon, _ = self.encode(self.decode(z_input))
        return self.decode(z), mu, log_var, z_input, z_recon

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

        if self.balanced_alpha:
            return loss_recon*(1-self.alpha*self.wu_alpha) + loss_reg*self.beta + loss_lr*self.alpha*self.wu_alpha, loss_recon.detach(), loss_reg.detach(), loss_lr.detach()
        else:
            return loss_recon + loss_reg * self.beta + loss_lr * self.alpha * self.wu_alpha, loss_recon.detach(), loss_reg.detach(), loss_lr.detach()




class LIDVAE(VAE):

    def __init__(
        self,
        in_channel=1,
        latent_channel=32,
        hidden_channels=[32, 64, 128],
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
            hidden_channels = [32, 64, 128, 256]
            input_dim = 64
        elif dataset == "mnist" or "fashionmnist":
            in_channel = 1
            latent_channel = 32
            hidden_channels = [32, 64, 128]
            input_dim = 28

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

        # Make encoder
        self.encoder = []
        last_channel = in_channel

        for channel in hidden_channels:
            self.encoder.append(
                torch.nn.Sequential(
                    module.ResidualBlock(last_channel, channel, 2),
                    module.ResidualBlock(channel, channel, 1),
                )
            )
            last_channel = channel

        self.encoder.append(
            torch.nn.Sequential(
                torch.nn.Flatten(),
                torch.nn.Linear(last_channel * (fc_dim**2), latent_channel * 2),
                torch.nn.BatchNorm1d(latent_channel * 2),
                torch.nn.LeakyReLU(),
                torch.nn.Linear(latent_channel * 2, latent_channel * 2),
            )
        )
        self.encoder = torch.nn.Sequential(*self.encoder)

        # Make decoder
        self.decoder = []

        # First layer: ICNN in latent channel
        self.decoder.append(module.ICNN(latent_channel, icnn_channels[0]))

        # In the original implmentation,
        # a trainable full-rank matrix is used as Beta via SVD (as in appendix)
        # Here, we use an identity matrix for injective map (as in main text)
        self.register_buffer(
            "B",
            torch.eye((input_dim**2) * in_channel, latent_channel, requires_grad=False),
        )

        # Second and last layer: ICNN in data dimension
        self.decoder.append(module.ICNN((input_dim**2) * in_channel, icnn_channels[1]))

        # Unflatten to shape of image
        self.decoder.append(torch.nn.Unflatten(1, (in_channel, input_dim, input_dim)))

        # Note that there is no range mapping!
        # Use Clip or Sigmoid if you want
        self.decoder = torch.nn.ModuleList(self.decoder)

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

   
    def forward(self, input, latent_recon=False):
        if latent_recon:
            return self.forward_qzx(input) # should be equal to lrvae source
            # return self.forward_Ex(input)
        else:
            return self.forward_vae(input)

    def forward_vae(self, input):
        mu, log_var = self.encode(input)
        z = mu + torch.randn_like(mu) * torch.exp(log_var * 0.5)
        return self.decode(z), mu, log_var
    
    def forward_Ex(self, input): # Latent reconstruction, z is encoded from x
        mu, log_var = self.encode(input)
        z = mu + torch.randn_like(mu) * torch.exp(log_var * 0.5)
        recon = self.decode(z)
        z_recon, _ = self.encode(recon)
        return recon, mu, log_var, z, z_recon
    
    def forward_qzx(self, input): # Latent reconstruction, z is encoded from x, z is reconstructed to mu
        mu, log_var = self.encode(input)
        z = mu + torch.randn_like(mu) * torch.exp(log_var * 0.5)
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
                        2 * torch.pi * ((input - output) ** 2).mean(1).mean(1).mean(1)
                        + 1e-5  # To avoid log(0)
                    ).log()
                    + 1
                )
            ).mean()
        )
        loss_reg = (-0.5 * (1 + log_var - mu**2 - log_var.exp())).mean(dim=0).sum()

        return loss_recon + loss_reg * self.beta, loss_recon.detach(), loss_reg.detach(), 0.0


class ConvVAE(VAE):

    def __init__(
        self,
        in_channel=1,
        latent_channel=32,
        hidden_channels=[32, 64, 128],
        input_dim=28,
        beta=1.0,
        is_log_mse=False,
        dataset=None,
    ):
        """
        Conventional VAE with residual-convolution encoder and decoder, for image dataset.
        Note that decoder also consists of convolution and its transpose.
        Beta and logMSE features are ready-to-use, but are disabled by default.
        """
        if dataset == "celeba":
            in_channel = 3
            latent_channel = 64
            hidden_channels = [32, 64, 128, 256]
            input_dim = 64
        elif dataset == "mnist" or "fashionmnist":
            in_channel = 1
            latent_channel = 32
            hidden_channels = [32, 64, 128]
            input_dim = 28

        super(VAE, self).__init__()

        self.latent_channel = latent_channel
        self.beta = beta
        self.is_log_mse = is_log_mse

        fc_dim = input_dim
        transpose_padding = []
        for _ in range(len(hidden_channels)):
            transpose_padding.append((fc_dim + 1) % 2)
            fc_dim = (fc_dim - 1) // 2 + 1
        transpose_padding.reverse()

        # Make encoder
        self.encoder = []
        last_channel = in_channel

        for channel in hidden_channels:
            self.encoder.append(
                torch.nn.Sequential(
                    module.ResidualBlock(last_channel, channel, 2),
                    module.ResidualBlock(channel, channel, 1),
                )
            )
            last_channel = channel

        self.encoder.append(
            torch.nn.Sequential(
                torch.nn.Flatten(),
                torch.nn.Linear(last_channel * (fc_dim**2), latent_channel * 2),
                torch.nn.BatchNorm1d(latent_channel * 2),
                torch.nn.LeakyReLU(),
                torch.nn.Linear(latent_channel * 2, latent_channel * 2),
            )
        )
        self.encoder = torch.nn.Sequential(*self.encoder)

        # Make decoder
        hidden_channels.reverse()

        self.decoder = []
        last_channel = hidden_channels[0]

        self.decoder.append(
            torch.nn.Sequential(
                torch.nn.Linear(latent_channel, last_channel * (fc_dim**2)),
                torch.nn.BatchNorm1d(last_channel * (fc_dim**2)),
                torch.nn.LeakyReLU(),
                torch.nn.Unflatten(1, (last_channel, fc_dim, fc_dim)),
                module.ResidualBlock(last_channel, last_channel, 1),
            )
        )

        for channel, pad in zip(hidden_channels[1:], transpose_padding[:-1]):
            self.decoder.append(
                torch.nn.Sequential(
                    torch.nn.ConvTranspose2d(last_channel, channel, 3, 2, 1, pad),
                    torch.nn.BatchNorm2d(channel),
                    torch.nn.LeakyReLU(),
                )
            )
            last_channel = channel

        self.decoder.append(
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
        self.decoder = torch.nn.Sequential(*self.decoder)

        hidden_channels.reverse()

    def encode(self, input):
        ret = self.encoder(input)
        #return ret.split(ret.shape[1] // 2, 1)
        mu, var = ret.split(ret.shape[1] // 2, 1)
        return mu, F.softplus(var) # prevent var from being negative

    def decode(self, input):
        return self.decoder(input)

    def forward(self, input):
        mu, log_var = self.encode(input)
        z = mu + torch.randn_like(mu) * torch.exp(log_var * 0.5)
        return self.decode(z), mu, log_var

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
        loss_reg = (-0.5 * (1 + log_var - mu**2 - log_var.exp())).mean(dim=0).sum()

        return loss_recon + loss_reg * self.beta, loss_recon.detach(), loss_reg.detach(), 0.0
