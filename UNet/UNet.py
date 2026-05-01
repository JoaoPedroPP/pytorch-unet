import torch
import torchvision.transforms.functional
from torch import nn

class DoubleConvolution(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()

        # Primeira camada convolucional 3x3
        self.first = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.act1 = nn.ReLU()
        # Segunda camada convolucional 3x3
        self.second = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.act2 = nn.ReLU()

    def forward(self, x: torch.Tensor):
        # Executa as duas primeiras camadas convolucionais
        x = self.first(x)
        x = self.act1(x)
        x = self.second(x)
        return self.act2(x)

class DownSample(nn.Module):
    def __init__(self):
        super().__init__()
        self.pool = nn.MaxPool2d(2)

    def forward(self, x: torch.Tensor):
        return self.pool(x)

class UpSample(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        # Convolucao apliando dimensao
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor):
        return self.up(x)

class CropAndConcatenate(nn.Module):
    def forward(self, x: torch.Tensor, contracting_x: torch.Tensor):
        # Ajusta o tamaho do feature map para o tamanho correto
        contracting_x = torchvision.transforms.functional.center_crop(contracting_x, [x.shape[2], x.shape[3]])
        # Concatena o feature map
        x = torch.cat([x, contracting_x], dim=1)

        return x

class UNet(nn.Module):
    def __init__(self, in_channels, out_channels, simple=False):
        super().__init__()

        # Original
        down_conv_sizes = [(in_channels, 64), (64, 128), (128, 256), (256, 512)]
        # Without the last layer
        if simple:
            down_conv_sizes = [(in_channels, 64), (64, 128), (128, 256)]
        self.down_conv = nn.ModuleList([DoubleConvolution(i, o) for i,o in down_conv_sizes])

        self.down_sample = nn.ModuleList([DownSample() for _ in range(len(down_conv_sizes))])

        # Original
        self.middle_conv = DoubleConvolution(512, 1024)
        # Without the last layer
        if simple:
            self.middle_conv = DoubleConvolution(256, 512)

        # Original
        upsample_sizes = [(1024, 512), (512, 256), (256, 128), (128, 64)]
        # Without the last layer
        if simple:
            upsample_sizes = [(512, 256), (256, 128), (128, 64)]

        self.up_sample = nn.ModuleList([UpSample(i, o) for i, o in upsample_sizes])

        # Original
        up_conv_sizes = [(1024, 512), (512, 256), (256, 128), (128, 64)]
        # Without the last layer
        if simple:
            up_conv_sizes = [(512, 256), (256, 128), (128, 64)]
        self.up_conv = nn.ModuleList([DoubleConvolution(i, o) for i, o in up_conv_sizes])

        self.concat = nn.ModuleList([CropAndConcatenate() for _ in range(len(up_conv_sizes))])

        self.final_conv = nn.Conv2d(64, out_channels, kernel_size=1)
        # self.final_activation = nn.Softmax(dim=out_channels)

    def forward(self, x: torch.Tensor):
        pass_trough = []
        for i in range(len(self.down_conv)):
            x = self.down_conv[i](x)
            pass_trough.append(x)
            x = self.down_sample[i](x)

        x = self.middle_conv(x)

        for i in range(len(self.up_conv)):
            x = self.up_sample[i](x)
            x = self.concat[i](x, pass_trough.pop())
            x = self.up_conv[i](x)

        out = self.final_conv(x)
        # out = self.final_activation(out)

        return out
