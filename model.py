import torch
import torch.nn as nn
import math
import torch.nn.functional as F
'''
The codes of Residual Block is heavily borrowed from:
https://github.com/clovaai/stargan-v2/blob/master/core/model.py
'''
class ResBlk(nn.Module):
    def __init__(self, dim_in, dim_out, actv=nn.LeakyReLU(0.2),
                 normalize=True, downsample=False):
        super().__init__()
        self.actv = actv
        self.normalize = normalize
        self.downsample = downsample
        self.learned_sc = dim_in != dim_out
        self._build_weights(dim_in, dim_out)

    def _build_weights(self, dim_in, dim_out):
        self.conv1 = nn.Conv2d(dim_in, dim_in, 3, 1, 1)
        self.conv2 = nn.Conv2d(dim_in, dim_out, 3, 1, 1)
        if self.normalize:
            self.norm1 = nn.InstanceNorm2d(dim_in, affine=True)
            self.norm2 = nn.InstanceNorm2d(dim_in, affine=True)
        if self.learned_sc:
            self.conv1x1 = nn.Conv2d(dim_in, dim_out, 1, 1, 0, bias=False)

    def _shortcut(self, x):
        if self.learned_sc:
            x = self.conv1x1(x)
        if self.downsample:
            x = F.avg_pool2d(x, 2)
        return x

    def _residual(self, x):
        if self.normalize:
            x = self.norm1(x)
        x = self.actv(x)
        x = self.conv1(x)
        if self.downsample:
            x = F.avg_pool2d(x, 2)
        if self.normalize:
            x = self.norm2(x)
        x = self.actv(x)
        x = self.conv2(x)
        return x

    def forward(self, x):
        x = self._shortcut(x) + self._residual(x)
        return x / math.sqrt(2)  # unit variance

class UpResBlk(nn.Module):
    def _build_weights(self, dim_in, dim_out):
        self.conv1 = nn.Conv2d(dim_in, dim_in, 3, 1, 1)
        self.conv2 = nn.Conv2d(dim_in, dim_out, 3, 1, 1)
        self.norm1 = nn.InstanceNorm2d(dim_in, affine=True)
        self.norm2 = nn.InstanceNorm2d(dim_in, affine=True)
        if self.learned_sc:
            self.conv1x1 = nn.Conv2d(dim_in, dim_out, 1, 1, 0, bias=False)

    def __init__(self, dim_in, dim_out, actv=nn.LeakyReLU(0.2)):
        super().__init__()
        self.learned_sc = dim_in != dim_out
        self._build_weights(dim_in, dim_out)
        self.actv = actv
    def _shortcut(self, x):
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        if self.learned_sc:
            x = self.conv1x1(x)
        return x
    def _residual(self, x):
        x = self.norm1(x)
        x = self.actv(x)

        x = F.interpolate(x, scale_factor=2, mode='nearest')
        x = self.conv1(x)
        x = self.norm2(x)
        x = self.actv(x)
        x = self.conv2(x)
        return x
    def forward(self, x):
        out = self._residual(x)
        out = (out + self._shortcut(x)) / math.sqrt(2)
        return out

class Discriminator(nn.Module):
    def btn_build(self):
        btn = list()
        for i in range(4):
            btn.append(ResBlk(1024, 1024, downsample=False))
        return nn.Sequential(*btn)
    def __init__(self):
        super().__init__()
        self.encoder128 = ResBlk(3, 64, downsample=True)
        self.encoder64 = ResBlk(64, 128, downsample=True)
        self.encoder32 = ResBlk(128, 256, downsample=True)
        self.encoder16 = ResBlk(256, 512, downsample=True)
        self.encoder8 = ResBlk(512, 1024, downsample=True)
        self.btn = self.btn_build()
        self.decoder16 = UpResBlk(1024, 512)
        self.decoder32 = UpResBlk(512, 256)
        self.decoder64 = UpResBlk(256, 128)
        self.decoder128 = UpResBlk(128, 64)
        self.decoder256 = nn.Sequential(
            UpResBlk(64, 1),
            nn.Sigmoid()
        )
    def forward(self, x):
        enc128 = self.encoder128(x)
        enc64 = self.encoder64(enc128)
        enc32 = self.encoder32(enc64)
        enc16 = self.encoder16(enc32)
        enc8 = self.encoder8(enc16)
        enc8 = self.btn(enc8)
        dec16 = self.decoder16(enc8)
        dec32 = self.decoder32( dec16 * enc16 + enc16 )
        dec64 = self.decoder64( dec32 * enc32 + enc32)
        dec128 = self.decoder128(dec64 * enc64 + enc64)
        dec256 = self.decoder256(dec128 * enc128 + enc128)
        return dec256


if __name__ == "__main__":
    toy_input = torch.randn((4, 3, 256, 256)).cuda()
    discriminator = Discriminator().cuda()
    p_map = discriminator(toy_input)
    print(p_map.size())