import math
import torch
from torch import nn as nn
from torch.nn import functional as F
from torch.nn import init as init
from torch.nn.modules.batchnorm import _BatchNorm

# try:
#     from basicsr.models.ops.dcn import (ModulatedDeformConvPack,
#                                         modulated_deform_conv)
# except ImportError:
#     # print('Cannot import dcn. Ignore this warning if dcn is not used. '
#     #       'Otherwise install BasicSR with compiling dcn.')
#

@torch.no_grad()
def default_init_weights(module_list, scale=1, bias_fill=0, **kwargs):
    """Initialize network weights.

    Args:
        module_list (list[nn.Module] | nn.Module): Modules to be initialized.
        scale (float): Scale initialized weights, especially for residual
            blocks. Default: 1.
        bias_fill (float): The value to fill bias. Default: 0
        kwargs (dict): Other arguments for initialization function.
    """
    if not isinstance(module_list, list):
        module_list = [module_list]
    for module in module_list:
        for m in module.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, **kwargs)
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.fill_(bias_fill)
            elif isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight, **kwargs)
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.fill_(bias_fill)
            elif isinstance(m, _BatchNorm):
                init.constant_(m.weight, 1)
                if m.bias is not None:
                    m.bias.data.fill_(bias_fill)


def make_layer(basic_block, num_basic_block, **kwarg):
    """Make layers by stacking the same blocks.

    Args:
        basic_block (nn.module): nn.module class for basic block.
        num_basic_block (int): number of blocks.

    Returns:
        nn.Sequential: Stacked blocks in nn.Sequential.
    """
    layers = []
    for _ in range(num_basic_block):
        layers.append(basic_block(**kwarg))
    return nn.Sequential(*layers)


class ResidualBlockNoBN(nn.Module):
    """Residual block without BN.

    It has a style of:
        ---Conv-ReLU-Conv-+-
         |________________|

    Args:
        num_feat (int): Channel number of intermediate features.
            Default: 64.
        res_scale (float): Residual scale. Default: 1.
        pytorch_init (bool): If set to True, use pytorch default init,
            otherwise, use default_init_weights. Default: False.
    """

    def __init__(self, num_feat=64, res_scale=1, pytorch_init=False):
        super(ResidualBlockNoBN, self).__init__()
        self.res_scale = res_scale
        self.conv1 = nn.Conv2d(num_feat, num_feat, 3, 1, 1, bias=True)
        self.conv2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1, bias=True)
        self.relu = nn.ReLU(inplace=True)

        if not pytorch_init:
            default_init_weights([self.conv1, self.conv2], 0.1)

    def forward(self, x):
        identity = x
        out = self.conv2(self.relu(self.conv1(x)))
        return identity + out * self.res_scale


class Upsample(nn.Sequential):
    """Upsample module.

    Args:
        scale (int): Scale factor. Supported scales: 2^n and 3.
        num_feat (int): Channel number of intermediate features.
    """

    def __init__(self, scale, num_feat):
        m = []
        if (scale & (scale - 1)) == 0:  # scale = 2^n
            for _ in range(int(math.log(scale, 2))):
                m.append(nn.Conv2d(num_feat, 4 * num_feat, 3, 1, 1))
                m.append(nn.PixelShuffle(2))
        elif scale == 3:
            m.append(nn.Conv2d(num_feat, 9 * num_feat, 3, 1, 1))
            m.append(nn.PixelShuffle(3))
        else:
            raise ValueError(f'scale {scale} is not supported. '
                             'Supported scales: 2^n and 3.')
        super(Upsample, self).__init__(*m)


def flow_warp(x,
              flow,
              interp_mode='bilinear',
              padding_mode='zeros',
              align_corners=True):
    """Warp an image or feature map with optical flow.

    Args:
        x (Tensor): Tensor with size (n, c, h, w).
        flow (Tensor): Tensor with size (n, h, w, 2), normal value.
        interp_mode (str): 'nearest' or 'bilinear'. Default: 'bilinear'.
        padding_mode (str): 'zeros' or 'border' or 'reflection'.
            Default: 'zeros'.
        align_corners (bool): Before pytorch 1.3, the default value is
            align_corners=True. After pytorch 1.3, the default value is
            align_corners=False. Here, we use the True as default.

    Returns:
        Tensor: Warped image or feature map.
    """
    assert x.size()[-2:] == flow.size()[1:3]
    _, _, h, w = x.size()
    # create mesh grid
    grid_y, grid_x = torch.meshgrid(
        torch.arange(0, h).type_as(x),
        torch.arange(0, w).type_as(x))
    grid = torch.stack((grid_x, grid_y), 2).float()  # W(x), H(y), 2
    grid.requires_grad = False

    vgrid = grid + flow
    # scale grid to [-1,1]
    vgrid_x = 2.0 * vgrid[:, :, :, 0] / max(w - 1, 1) - 1.0
    vgrid_y = 2.0 * vgrid[:, :, :, 1] / max(h - 1, 1) - 1.0
    vgrid_scaled = torch.stack((vgrid_x, vgrid_y), dim=3)
    output = F.grid_sample(
        x,
        vgrid_scaled,
        mode=interp_mode,
        padding_mode=padding_mode,
        align_corners=align_corners)

    # TODO, what if align_corners=False
    return output


def resize_flow(flow,
                size_type,
                sizes,
                interp_mode='bilinear',
                align_corners=False):
    """Resize a flow according to ratio or shape.

    Args:
        flow (Tensor): Precomputed flow. shape [N, 2, H, W].
        size_type (str): 'ratio' or 'shape'.
        sizes (list[int | float]): the ratio for resizing or the final output
            shape.
            1) The order of ratio should be [ratio_h, ratio_w]. For
            downsampling, the ratio should be smaller than 1.0 (i.e., ratio
            < 1.0). For upsampling, the ratio should be larger than 1.0 (i.e.,
            ratio > 1.0).
            2) The order of output_size should be [out_h, out_w].
        interp_mode (str): The mode of interpolation for resizing.
            Default: 'bilinear'.
        align_corners (bool): Whether align corners. Default: False.

    Returns:
        Tensor: Resized flow.
    """
    _, _, flow_h, flow_w = flow.size()
    if size_type == 'ratio':
        output_h, output_w = int(flow_h * sizes[0]), int(flow_w * sizes[1])
    elif size_type == 'shape':
        output_h, output_w = sizes[0], sizes[1]
    else:
        raise ValueError(
            f'Size type should be ratio or shape, but got type {size_type}.')

    input_flow = flow.clone()
    ratio_h = output_h / flow_h
    ratio_w = output_w / flow_w
    input_flow[:, 0, :, :] *= ratio_w
    input_flow[:, 1, :, :] *= ratio_h
    resized_flow = F.interpolate(
        input=input_flow,
        size=(output_h, output_w),
        mode=interp_mode,
        align_corners=align_corners)
    return resized_flow


# TODO: may write a cpp file
def pixel_unshuffle(x, scale):
    """ Pixel unshuffle.

    Args:
        x (Tensor): Input feature with shape (b, c, hh, hw).
        scale (int): Downsample ratio.

    Returns:
        Tensor: the pixel unshuffled feature.
    """
    b, c, hh, hw = x.size()
    out_channel = c * (scale**2)
    assert hh % scale == 0 and hw % scale == 0
    h = hh // scale
    w = hw // scale
    x_view = x.view(b, c, h, scale, w, scale)
    return x_view.permute(0, 1, 3, 5, 2, 4).reshape(b, out_channel, h, w)


# class DCNv2Pack(ModulatedDeformConvPack):
#     """Modulated deformable conv for deformable alignment.
#
#     Different from the official DCNv2Pack, which generates offsets and masks
#     from the preceding features, this DCNv2Pack takes another different
#     features to generate offsets and masks.
#
#     Ref:
#         Delving Deep into Deformable Alignment in Video Super-Resolution.
#     """
#
#     def forward(self, x, feat):
#         out = self.conv_offset(feat)
#         o1, o2, mask = torch.chunk(out, 3, dim=1)
#         offset = torch.cat((o1, o2), dim=1)
#         mask = torch.sigmoid(mask)
#
#         offset_absmean = torch.mean(torch.abs(offset))
#         if offset_absmean > 50:
#             logger = get_root_logger()
#             logger.warning(
#                 f'Offset abs mean is {offset_absmean}, larger than 50.')
#
#         return modulated_deform_conv(x, offset, mask, self.weight, self.bias,
#                                      self.stride, self.padding, self.dilation,
#                                      self.groups, self.deformable_groups)

def initialize_weights(net_l, scale=1):
    if not isinstance(net_l, list):
        net_l = [net_l]
    for net in net_l:
        for m in net.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, a=0, mode='fan_in')
                m.weight.data *= scale  # for residual block
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight, a=0, mode='fan_in')
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias.data, 0.0)


def make_layer(block, n_layers):
    layers = []
    for _ in range(n_layers):
        layers.append(block())
    return nn.Sequential(*layers)


class ResidualBlock_noBN(nn.Module):
    '''Residual block w/o BN
    ---Conv-ReLU-Conv-+-
     |________________|
    '''

    def __init__(self, nf=64):
        super(ResidualBlock_noBN, self).__init__()
        self.conv1 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.conv2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)

        # initialization
        initialize_weights([self.conv1, self.conv2], 0.1)

    def forward(self, x):
        identity = x
        out = F.relu(self.conv1(x), inplace=True)
        out = self.conv2(out)
        return identity + out

class ResidualBlock(nn.Module):
    '''Residual block w/o BN
    ---Conv-ReLU-Conv-+-
     |________________|
    '''

    def __init__(self, nf=64):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.bn = nn.BatchNorm2d(nf)
        self.conv2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)

        # initialization
        initialize_weights([self.conv1, self.conv2], 0.1)

    def forward(self, x):
        identity = x
        out = F.relu(self.bn(self.conv1(x)), inplace=True)
        out = self.conv2(out)
        return identity + out

import torch
import torch.nn.functional as F
import numpy as np
from kornia.filters import sobel
from archs.Fourier import *

class Down(nn.Module):
    def __init__(self, in_channels, chan_factor, bias=False):
        super(Down, self).__init__()

        self.bot = nn.Sequential(
            nn.AvgPool2d(2, ceil_mode=True, count_include_pad=False),
            nn.Conv2d(in_channels, int(in_channels * chan_factor), 1, stride=1, padding=0, bias=bias),
            # nn.BatchNorm2d(int(in_channels * chan_factor)),
            # # nn.LeakyReLU(0.1, inplace=True),
            # nn.SiLU(inplace=True)
        )

    def forward(self, x):
        return self.bot(x)

class DownSample(nn.Module):
    def __init__(self, in_channels, scale_factor, chan_factor=2, kernel_size=3):
        super(DownSample, self).__init__()
        self.scale_factor = int(np.log2(scale_factor))

        modules_body = []
        for i in range(self.scale_factor):
            modules_body.append(Down(in_channels, chan_factor))
            in_channels = int(in_channels * chan_factor)

        self.body = nn.Sequential(*modules_body)

    def forward(self, x):
        x = self.body(x)
        return x

class Up(nn.Module):
    def __init__(self, in_channels, chan_factor, bias=False):
        super(Up, self).__init__()

        self.bot = nn.Sequential(
            nn.Conv2d(in_channels, int(in_channels // chan_factor), 1, stride=1, padding=0, bias=bias),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=bias),
            # nn.BatchNorm2d(int(in_channels // chan_factor)),
            # # nn.LeakyReLU(0.1, inplace=True),
            # nn.SiLU(inplace=True)
        )

    def forward(self, x):
        return self.bot(x)

class UpSample(nn.Module):
    def __init__(self, in_channels, scale_factor, chan_factor=2, kernel_size=3):
        super(UpSample, self).__init__()
        self.scale_factor = int(np.log2(scale_factor))

        modules_body = []
        for i in range(self.scale_factor):
            modules_body.append(Up(in_channels, chan_factor))
            in_channels = int(in_channels // chan_factor)

        self.body = nn.Sequential(*modules_body)

    def forward(self, x):
        x = self.body(x)
        return x

class En_unit(nn.Module):
    def __init__(self, n_feat, chan_factor,bias=False, n=2,groups=1):
        super(En_unit, self).__init__()

        self.down=DownSample(n_feat, 2, chan_factor)
        self.conv=nn.Sequential(
            ProcessBlock(int(n_feat * chan_factor )),
            # nn.Conv2d(int(n_feat * chan_factor), int(n_feat * chan_factor), 3, 1, 1, bias=bias),
            # SpaBlock_RCB(int(n_feat * chan_factor ))
            # nn.BatchNorm2d(int(n_feat * chan_factor)),
            # # nn.LeakyReLU(0.1, inplace=True),
            # nn.SiLU(inplace=True)
        )

    def forward(self, x_in):
        x=self.down(x_in)
        x=self.conv(x)
        return x

class CEM(nn.Module):
    def __init__(self, ch=128):
        super(CEM,self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(ch,ch,3,1,1,groups=4),
            # nn.BatchNorm2d(ch),
            nn.LeakyReLU(0.1),
            # nn.SiLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(ch, ch, 3, 1, 1,groups=4),
            # nn.BatchNorm2d(ch),
            nn.LeakyReLU(0.1),
            # nn.SiLU(inplace=True)
        )

        self.sigm= nn.Sigmoid()
        self.tanh=nn.Tanh()

    def forward(self, fs):

        f1=fs[0]
        f2=fs[1]

        f12=f1+f2

        attention1 = self.sigm(self.conv1(f12))
        attention2 = self.sigm(self.conv2(f12))

        x = f1*attention1 + f2*attention2

        return x

class De_unit_concat2(nn.Module):
    def __init__(self, n_feat, chan_factor,bias=False, n=2,groups=1):
        super(De_unit_concat2, self).__init__()
        self.up=UpSample(n_feat, 2, chan_factor)
        self.conv0 = nn.Conv2d(int(n_feat / chan_factor),int(n_feat / chan_factor),3,1,1,bias=bias)
        # self.conv0 = SpaBlock(int(n_feat / chan_factor))

        self.fusion= nn.Sequential(
            nn.Conv2d(2*int(n_feat / chan_factor),int(n_feat / chan_factor),3,1,1,bias=bias),
            # nn.BatchNorm2d(int(n_feat / chan_factor)),
            # nn.LeakyReLU(0.1, inplace=True),
            # nn.SiLU(inplace=True)
        )

    def forward(self, x_in, x_en):
        x=self.conv0(self.up(x_in))
        x=self.fusion(torch.cat((x,x_en),dim=1))

        return x

class lightnessNet(nn.Module):
    def __init__(self,
                 inp_channels=3,
                 out_channels=3,
                 n_feat=32,
                 chan_factor=2,
                 bias=False,
                 ):

        super(lightnessNet, self).__init__()
        self.inp_channels=inp_channels
        self.out_channels=out_channels

        self.conv_in = nn.Sequential(
            nn.Conv2d(inp_channels, n_feat, kernel_size=3, padding=1, bias=bias),
            # SpaBlock(int(n_feat * chan_factor ** 0)),
            # ProcessBlock(int(n_feat * chan_factor ** 0)),
        )

        self.down1=En_unit(int((chan_factor ** 0) * n_feat),chan_factor,bias)
        self.down2=En_unit(int((chan_factor ** 1) * n_feat),chan_factor,bias)
        self.down3=En_unit(int((chan_factor ** 2) * n_feat),chan_factor,bias)
        self.down4=En_unit(int((chan_factor ** 3) * n_feat),chan_factor,bias)

        self.up1 = De_unit_concat2(int((chan_factor ** 4) * n_feat), chan_factor, bias)
        self.up2 = De_unit_concat2(int((chan_factor ** 3) * n_feat), chan_factor, bias)
        self.up3 = De_unit_concat2(int((chan_factor ** 2) * n_feat), chan_factor, bias)
        self.up4 = De_unit_concat2(int((chan_factor ** 1) * n_feat), chan_factor, bias)

        self.conv_out = nn.Sequential(
            # ProcessBlock(int(n_feat * chan_factor ** 0)),
            # SpaBlock(int(n_feat * chan_factor ** 0)),
            nn.Conv2d(n_feat, out_channels, kernel_size=3, padding=1, bias=bias),
        )

    def forward(self, inp_img):

        inv_map = 1-inp_img
        edg_map = sobel(inp_img)

        shallow_feats = self.conv_in(inp_img)

        down1=self.down1(shallow_feats)
        down2=self.down2(down1)
        down3=self.down3(down2)
        down4=self.down4(down3)

        up1 = self.up1(down4,down3*inv_map[:,:,::8,::8]*(1+edg_map[:,:,::8,::8]))
        up2 = self.up2(up1,down2*inv_map[:,:,::4,::4]*(1+edg_map[:,:,::4,::4]))
        up3 = self.up3(up2,down1*inv_map[:,:,::2,::2]*(1+edg_map[:,:,::2,::2]))
        up4 = self.up4(up3,shallow_feats*inv_map*(1+edg_map))

        # up1 = self.up1(down4, down3)
        # up2 = self.up2(up1, down2)
        # up3 = self.up3(up2, down1)
        # up4 = self.up4(up3, shallow_feats)

        out_img = self.conv_out(up4)+inp_img

        return out_img,[shallow_feats,down1,down2,down3,down4]

class ChromNet(nn.Module):
    def __init__(self,
                 inp_channels=3,
                 out_channels=3,
                 n_feat=32,
                 chan_factor=2,
                 bias=False,
                 color_aug=False
                 ):
        super(ChromNet, self).__init__()
        self.inp_channels = inp_channels
        self.out_channels = out_channels
        self.color_aug = color_aug

        self.conv_in = nn.Sequential(
            nn.Conv2d(inp_channels, int(n_feat * chan_factor ** 0), kernel_size=3, padding=1, bias=bias),
            # ProcessBlock(int(n_feat * chan_factor ** 0)),
        )

        self.down1 = En_unit(int((chan_factor ** 0) * n_feat), chan_factor, bias, groups=1)
        self.down2 = En_unit(int((chan_factor ** 1) * n_feat), chan_factor, bias, groups=2)

        self.fusion2 = CEM(ch=int((chan_factor ** 2) * n_feat))
        # self.fusion2 = nn.Conv2d(int((chan_factor ** 2) * n_feat * 2),int((chan_factor ** 2) * n_feat),3,1,1)

        self.up1 = De_unit_concat2(int((chan_factor ** 4) * n_feat), chan_factor, bias, 2,groups=8)
        self.up2 = De_unit_concat2(int((chan_factor ** 3) * n_feat), chan_factor, bias, 2,groups=4)
        self.up3 = De_unit_concat2(int((chan_factor ** 2) * n_feat), chan_factor, bias, 2,groups=2)
        self.up4 = De_unit_concat2(int((chan_factor ** 1) * n_feat), chan_factor, bias, 2,groups=1)

        self.encode_q = nn.Conv2d(int((chan_factor ** 2) * n_feat), 313, kernel_size=1, padding=0, dilation=1, stride=1, bias=True)

        self.conv_out = nn.Sequential(
            # ProcessBlock(int(n_feat * chan_factor ** 0)),
            nn.Conv2d(n_feat, out_channels, kernel_size=3, padding=1, bias=bias),
        )

    def forward(self, inp_img, mid=None, ref_map=None,saturation = None):
        if saturation:
            inp_img*=saturation
        if ref_map is not None:
            if saturation:
                ref_map *= saturation
            shallow_feats_ = self.conv_in(ref_map)
            shallow_feats = self.conv_in(inp_img)
            down1_ = self.down1(shallow_feats_)
            down1 = self.down1(shallow_feats)
            down2_ = self.down2(down1_)
            down2 = 0.7 * self.down2(down1)  + 0.3 * down2_

        else:
            if self.color_aug:
                random_number1 = torch.rand(1)
                if random_number1>0.5:
                    random_number2 = 0.5 + 0.5 * torch.rand(1)
                    random_number2 = random_number2.to(inp_img.device)
                    inp_img = inp_img*random_number2
            shallow_feats = self.conv_in(inp_img)
            down1 = self.down1(shallow_feats)
            down2 = self.down2(down1)

        up1 = self.up1(mid[4], mid[3])
        up2 = self.up2(up1, mid[2])

        up2 = self.fusion2([up2, down2])
        # up2 = self.fusion2(torch.cat((up2, down2),dim=1))
        q = self.encode_q(up2)
        up3 = self.up3(up2,mid[1])
        up4 = self.up4(up3,mid[0])

        out_img = self.conv_out(up4)

        return out_img, q