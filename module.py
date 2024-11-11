import torch


class PlainConvolution(torch.nn.Module):
    def __init__(self, in_channel, out_channel, stride=1):
        super().__init__()

        self.activation = torch.nn.LeakyReLU()

        self.conv1 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channel, out_channel, 3, stride, 1),
            torch.nn.BatchNorm2d(out_channel),
            self.activation,
        )

        self.conv2 = torch.nn.Sequential(
            torch.nn.Conv2d(out_channel, out_channel, 3, 1, 1),
            torch.nn.BatchNorm2d(out_channel),
            self.activation,
        )

    def forward(self, input):
        output = self.conv1(input)
        output = self.conv2(output)

        return output


class ResidualBlock(torch.nn.Module):
    def __init__(self, in_channel, out_channel, stride=1):
        super().__init__()

        self.activation = torch.nn.LeakyReLU()

        self.conv1 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channel, out_channel, 3, stride, 1),
            torch.nn.BatchNorm2d(out_channel),
            self.activation,
        )

        self.conv2 = torch.nn.Sequential(
            torch.nn.Conv2d(out_channel, out_channel, 3, 1, 1),
            torch.nn.BatchNorm2d(out_channel),
        )

        if stride == 1 and in_channel == out_channel:
            self.identity = torch.nn.Identity()
        else:
            self.identity = torch.nn.Sequential(
                torch.nn.Conv2d(in_channel, out_channel, 1, stride, 0),
                torch.nn.BatchNorm2d(out_channel),
            )

    def forward(self, input):
        output = self.conv1(input)
        output = self.conv2(output)
        output = self.activation(output + self.identity(input))

        return output


class PositiveLinear(torch.nn.Module):
    def __init__(self, in_channel, out_channel, is_exp=True):
        super().__init__()

        self.param = torch.nn.Parameter(torch.Tensor(out_channel, in_channel))
        # The original implementation uses xavier
        # But here we follows the pytorch default
        torch.nn.init.kaiming_uniform_(self.param, a=2.2360679775)  # a=sqrt(5)

        self.is_exp = is_exp

    def forward(self, input):
        if self.is_exp:
            return torch.nn.functional.linear(input, self.param.exp())
        else:
            # IDK but the original implementation is like this!
            # So if you want it to be the same, set is_exp=False
            return torch.nn.functional.linear(input, self.param.clamp(min=1e-2))


class ICNN(torch.nn.Module):
    def __init__(self, in_channel, hidden_channel=128, num_layers=2):
        super().__init__()

        # Note that the paper uses alpha=0.2 as an example
        self.activation = torch.nn.LeakyReLU(0.2)

        self.W = []
        self.A = []
        for _ in range(num_layers - 1):
            self.W.append(PositiveLinear(hidden_channel, hidden_channel))
            self.A.append(torch.nn.Linear(in_channel, hidden_channel))
        self.W.append(PositiveLinear(hidden_channel, 1))
        self.A.append(torch.nn.Linear(in_channel, 1))

        self.W = torch.nn.Sequential(*self.W)
        self.A = torch.nn.Sequential(*self.A)

        # Note that actual layers are A_{0:num_layers}
        # We have (num_layers+1) layers;
        # specifically, there are (num_layers) hidden layers and output layer
        # Then z_{num_layers+1} is in R^1
        # This strange numbering follows the original implementation
        self.A0 = torch.nn.Linear(in_channel, hidden_channel)

    def forward(self, input):
        x = self.activation(self.A0(input)).pow(2)

        for w, a in zip(self.W, self.A):
            x = self.activation(w(x) + a(input))

        return x


class LinearModule_EP(torch.nn.Module):
    def __init__(self, in_channel, hidden_channel=128, num_layers=2):
        super().__init__()

        # Note that the paper uses alpha=0.2 as an example
        self.activation = torch.nn.LeakyReLU(0.2)

        self.W = []
        self.A = []
        for _ in range(num_layers - 1):
            self.W.append(torch.nn.Linear(hidden_channel, hidden_channel))
            self.A.append(torch.nn.Linear(in_channel, hidden_channel))
        self.W.append(torch.nn.Linear(hidden_channel, in_channel)) # NOT EQUAL PARAMS (origianl: (hidden_channel, 1))
        self.A.append(torch.nn.Linear(in_channel, 1))

        self.W = torch.nn.Sequential(*self.W)
        self.A = torch.nn.Sequential(*self.A)

        # Note that actual layers are A_{0:num_layers}
        # We have (num_layers+1) layers;
        # specifically, there are (num_layers) hidden layers and output layer
        # Then z_{num_layers+1} is in R^1
        # This strange numbering follows the original implementation
        self.A0 = torch.nn.Linear(in_channel, hidden_channel)

    def forward(self, input):
        x = self.activation(self.A0(input)).pow(2)

        for w, a in zip(self.W, self.A):
            x = self.activation(w(x) + a(input))

        return x