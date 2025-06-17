# -*- coding = utf-8 -*-
# @time : 2025/6/17 19:15
# @Author : 于玉
# @File : SFANet.py
# @software: PyCharm
import pywt
import pywt.data
import torch.nn as nn
import torch
import torchvision
import torch.nn.functional as F
from timm.layers import DropPath, trunc_normal_
from torchvision.models import resnet50


def create_wavelet_filter(wave, in_size, out_size, type=torch.float):
    w = pywt.Wavelet(wave)
    dec_hi = torch.tensor(w.dec_hi[::-1], dtype=type)
    dec_lo = torch.tensor(w.dec_lo[::-1], dtype=type)
    dec_filters = torch.stack([dec_lo.unsqueeze(0) * dec_lo.unsqueeze(1),
                               dec_lo.unsqueeze(0) * dec_hi.unsqueeze(1),
                               dec_hi.unsqueeze(0) * dec_lo.unsqueeze(1),
                               dec_hi.unsqueeze(0) * dec_hi.unsqueeze(1)], dim=0)

    dec_filters = dec_filters[:, None].repeat(in_size, 1, 1, 1)

    rec_hi = torch.tensor(w.rec_hi[::-1], dtype=type).flip(dims=[0])
    rec_lo = torch.tensor(w.rec_lo[::-1], dtype=type).flip(dims=[0])
    rec_filters = torch.stack([rec_lo.unsqueeze(0) * rec_lo.unsqueeze(1),
                               rec_lo.unsqueeze(0) * rec_hi.unsqueeze(1),
                               rec_hi.unsqueeze(0) * rec_lo.unsqueeze(1),
                               rec_hi.unsqueeze(0) * rec_hi.unsqueeze(1)], dim=0)

    rec_filters = rec_filters[:, None].repeat(out_size, 1, 1, 1)

    return dec_filters, rec_filters


def wavelet_transform(x, filters):
    b, c, h, w = x.shape
    pad = (filters.shape[2] // 2 - 1, filters.shape[3] // 2 - 1)
    x = F.conv2d(x, filters, stride=2, groups=c, padding=pad)
    x = x.reshape(b, c, 4, h // 2, w // 2)
    return x


def inverse_wavelet_transform(x, filters):
    b, c, _, h_half, w_half = x.shape
    pad = (filters.shape[2] // 2 - 1, filters.shape[3] // 2 - 1)
    x = x.reshape(b, c * 4, h_half, w_half)
    x = F.conv_transpose2d(x, filters, stride=2, groups=c, padding=pad)
    return x


class WTConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=5, stride=1, bias=True, wt_levels=1, wt_type='db1'):
        super(WTConv2d, self).__init__()

        assert in_channels == out_channels

        self.in_channels = in_channels
        self.wt_levels = wt_levels
        self.stride = stride
        self.dilation = 1

        self.wt_filter, self.iwt_filter = create_wavelet_filter(wt_type, in_channels, in_channels, torch.float)
        self.wt_filter = nn.Parameter(self.wt_filter, requires_grad=False)
        self.iwt_filter = nn.Parameter(self.iwt_filter, requires_grad=False)

        self.base_conv = nn.Conv2d(in_channels, in_channels, kernel_size, padding='same', stride=1, dilation=1,
                                   groups=in_channels, bias=bias)
        self.base_scale = _ScaleModule([1, in_channels, 1, 1])

        self.wavelet_convs = nn.ModuleList(
            [nn.Conv2d(in_channels * 4, in_channels * 4, kernel_size, padding='same', stride=1, dilation=1,
                       groups=in_channels * 4, bias=False) for _ in range(self.wt_levels)]
        )
        self.wavelet_scale = nn.ModuleList(
            [_ScaleModule([1, in_channels * 4, 1, 1], init_scale=0.1) for _ in range(self.wt_levels)]
        )

        if self.stride > 1:
            self.do_stride = nn.AvgPool2d(kernel_size=1, stride=stride)
        else:
            self.do_stride = None

    def forward(self, x):

        x_ll_in_levels = []
        x_h_in_levels = []
        shapes_in_levels = []

        curr_x_ll = x

        for i in range(self.wt_levels):
            curr_shape = curr_x_ll.shape
            shapes_in_levels.append(curr_shape)
            if (curr_shape[2] % 2 > 0) or (curr_shape[3] % 2 > 0):
                curr_pads = (0, curr_shape[3] % 2, 0, curr_shape[2] % 2)
                curr_x_ll = F.pad(curr_x_ll, curr_pads)

            curr_x = wavelet_transform(curr_x_ll, self.wt_filter)
            curr_x_ll = curr_x[:, :, 0, :, :]

            shape_x = curr_x.shape
            curr_x_tag = curr_x.reshape(shape_x[0], shape_x[1] * 4, shape_x[3], shape_x[4])
            curr_x_tag = self.wavelet_scale[i](self.wavelet_convs[i](curr_x_tag))
            curr_x_tag = curr_x_tag.reshape(shape_x)

            x_ll_in_levels.append(curr_x_tag[:, :, 0, :, :])
            x_h_in_levels.append(curr_x_tag[:, :, 1:4, :, :])

        next_x_ll = 0

        for i in range(self.wt_levels - 1, -1, -1):
            curr_x_ll = x_ll_in_levels.pop()
            curr_x_h = x_h_in_levels.pop()
            curr_shape = shapes_in_levels.pop()

            curr_x_ll = curr_x_ll + next_x_ll

            curr_x = torch.cat([curr_x_ll.unsqueeze(2), curr_x_h], dim=2)
            next_x_ll = inverse_wavelet_transform(curr_x, self.iwt_filter)

            next_x_ll = next_x_ll[:, :, :curr_shape[2], :curr_shape[3]]

        x_tag = next_x_ll
        assert len(x_ll_in_levels) == 0

        x = self.base_scale(self.base_conv(x))
        x = x + x_tag

        if self.do_stride is not None:
            x = self.do_stride(x)

        return x


class _ScaleModule(nn.Module):
    def __init__(self, dims, init_scale=1.0, init_bias=0):
        super(_ScaleModule, self).__init__()
        self.dims = dims
        self.weight = nn.Parameter(torch.ones(*dims) * init_scale)
        self.bias = None

    def forward(self, x):
        return torch.mul(self.weight, x)


class FreLayer(nn.Module):
    def __init__(self, nc, expand=2):
        super(FreLayer, self).__init__()
        self.process1 = nn.Sequential(
            nn.Conv2d(nc, expand * nc, 3, 1, 1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(expand * nc, nc, 3, 1, 1))
        self.fre_output = nn.Conv2d(nc, nc, 1)
        self.gamma = nn.Parameter(torch.zeros((1, nc, 1, 1)), requires_grad=True)

    def forward(self, x):
        res = x
        _, _, H, W = x.shape
        x_freq = torch.fft.rfft2(x, norm='backward')
        mag = torch.abs(x_freq)
        pha = torch.angle(x_freq)
        mag = self.process1(mag)
        real = mag * torch.cos(pha)
        imag = mag * torch.sin(pha)
        x_out = torch.complex(real, imag)
        x_out = torch.fft.irfft2(x_out, s=(H, W), norm='backward')
        x = res * x_out
        x = res + x * self.gamma
        x_out = self.fre_output(x)
        return x_out


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_features, hidden_features, 1, 1, 0, bias=True),
            nn.GELU(),
            nn.BatchNorm2d(hidden_features, eps=1e-5),
        )
        self.proj = nn.Conv2d(hidden_features, hidden_features, 3, 1, 1, groups=hidden_features)
        self.proj_act = nn.GELU()
        self.proj_bn = nn.BatchNorm2d(hidden_features, eps=1e-5)
        self.conv2 = nn.Sequential(
            nn.Conv2d(hidden_features, out_features, 1, 1, 0, bias=True),
            nn.BatchNorm2d(out_features, eps=1e-5),
        )
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        B, N, C = x.shape
        x = x.permute(0, 2, 1).reshape(B, C, int(N ** 0.5), int(N ** 0.5))
        x = self.conv1(x)
        x = self.drop(x)
        x = self.proj(x) + x
        x = self.proj_act(x)
        x = self.proj_bn(x)
        x = self.conv2(x)
        x = x.flatten(2).permute(0, 2, 1)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.conv = WTConv2d(in_channels=dim, out_channels=dim, wt_levels=1)
        self.fft = FreLayer(nc=dim)
        self.alpha = nn.Parameter(torch.ones(1))

    def forward(self, x):
        B, N, C = x.shape
        H = W = int(N ** (1 / 2))
        x = x.reshape(B, C, H, W)
        wavelet_out = self.conv(x)
        fft = self.fft(x)
        out = self.alpha * wavelet_out + (1 - self.alpha) * fft
        out = out.reshape(B, N, C)
        return out


class TransformerBlock(nn.Module):
    """
    Implementation of Transformer,
    """

    def __init__(self, dim, mlp_ratio=4., drop=0., drop_path=0.,
                 norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, drop=drop)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode="fan_out")
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


def l2_normalization(x):
    norm = torch.norm(x, p=2, dim=1, keepdim=True)
    x = torch.div(x, norm)
    return x


class SEBlock(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return y


class ECVNet(nn.Module):
    def __init__(self, num_classes=18, epsilon=1e-4):
        super(ECVNet, self).__init__()
        self.epsilon = epsilon
        self.backbone = resnet50(weights=torchvision.models.ResNet50_Weights.DEFAULT)
        self.backbone.fc = nn.Linear(self.backbone.fc.in_features, num_classes)
        self.transformer1 = TransformerBlock(256)
        self.transformer2 = TransformerBlock(512)
        self.transformer3 = TransformerBlock(1024)

        self.sigmoid = nn.Sigmoid()

        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.downsample = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=2, padding=1)

        self.conv1x1_1 = nn.Conv2d(2048, 512, kernel_size=1, stride=1, padding=0)

        self.conv2x1_2 = nn.Conv2d(512, 512, kernel_size=1, stride=1, padding=0)
        self.se_block2 = SEBlock(512)
        self.conv2x1_3 = nn.Conv2d(1024, 512, kernel_size=1, stride=1, padding=0)

        self.conv3x1_2 = nn.Conv2d(512, 512, kernel_size=1, stride=1, padding=0)
        self.se_block3 = SEBlock(512)
        self.conv3x1_3 = nn.Conv2d(512, 512, kernel_size=1, stride=1, padding=0)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(1536, num_classes)

    def forward(self, x):
        x_rgb = self.backbone.conv1(x)
        x_rgb = self.backbone.bn1(x_rgb)
        x_rgb = self.backbone.relu(x_rgb)
        x_rgb = self.backbone.maxpool(x_rgb)

        x1 = self.backbone.layer1(x_rgb)
        f1_b, f1_c, f1_h, f1_w = x1.shape
        f1 = x1.flatten(2).transpose(1, 2)
        f1 = self.transformer1(f1)
        f1 = f1.transpose(1, 2)
        f1 = f1.reshape(f1_b, f1_c, f1_h, f1_w)
        x1 = x1 + f1

        x2 = self.backbone.layer2(x1)
        f2_b, f2_c, f2_h, f2_w = x2.shape
        f2 = x2.flatten(2).transpose(1, 2)
        f2 = self.transformer2(f2)
        f2 = f2.transpose(1, 2)
        f2 = f2.reshape(f2_b, f2_c, f2_h, f2_w)
        x2 = x2 + f2

        x3 = self.backbone.layer3(x2)
        f3_b, f3_c, f3_h, f3_w = x3.shape
        f3 = x3.flatten(2).transpose(1, 2)
        f3 = self.transformer3(f3)
        f3 = f3.transpose(1, 2)
        f3 = f3.reshape(f3_b, f3_c, f3_h, f3_w)
        x3 = x3 + f3

        x4 = self.backbone.layer4(x3)

        # D1
        d1 = self.conv1x1_1(x4)

        # D2
        path3 = self.upsample(d1)
        path3 = self.se_block2(path3)
        path4 = self.upsample(d1)
        path4 = self.conv2x1_2(path4)
        x3_1 = self.conv2x1_3(x3)
        d2_1 = x3_1 * path3
        d2 = d2_1 + path4

        # D3
        path5 = self.upsample(d2)
        path5 = self.se_block3(path5)
        path6 = self.upsample(d2)
        path6 = self.conv3x1_2(path6)
        x2_1 = self.conv3x1_3(x2)
        d3_1 = x2_1 * path5
        d3 = d3_1 + path6

        d1 = self.upsample(d1)
        d3 = self.downsample(d3)

        p = torch.cat((d1, d2, d3), dim=1)

        x = self.avgpool(p)
        x = torch.flatten(x, 1)
        x = l2_normalization(x)
        f_out = self.fc(x)

        x_r = self.backbone.avgpool(x4)
        x_r = torch.flatten(x_r, 1)
        x_out = self.backbone.fc(x_r)

        out = (f_out + x_out) / 2
        return out


if __name__ == '__main__':
    model = ECVNet(num_classes=18)
    print(model)
    input_tensor = torch.randn(16, 3, 224, 224)  # 假设 batch_size=16, 3个通道, 224x224的图像

    # 调用模型的 forward 方法
    output = model(input_tensor)

    # 输出结果
    print("Model output shape:", output.shape)

