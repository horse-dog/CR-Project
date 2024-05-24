import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Union
from torch.nn.common_types import _size_2_t


from models.base.pixel import PixelAttention
from models.base.transformer import PositionEmbeddingLearned, VisionTransformerEncoder, TransformerEncoderLayer
from models.retnet.retnet import RetNet
from utils.make_image import convert


class FusionConv2d(nn.Module):
    def __init__(
        self,
        in_opt_channels: int,
        out_opt_channels: int,
        in_sar_channels: int,
        out_sar_channels: int,
        kernel_size: _size_2_t,
        stride: _size_2_t = 1,
        padding: Union[str, _size_2_t] = 0,
        dilation: _size_2_t = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = 'zeros',
        device=None,
        dtype=None,
    ):
        super().__init__()

        self.in_opt_channels = in_opt_channels
        self.out_opt_channels = out_opt_channels
        self.in_sar_channels = in_sar_channels
        self.out_sar_channels = out_sar_channels
        
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.conv1 = nn.Conv2d(in_opt_channels + in_sar_channels, out_opt_channels, kernel_size,
                                stride, padding, dilation, groups, bias, padding_mode, device, dtype)
        self.conv2 = nn.Conv2d(in_sar_channels, out_sar_channels, kernel_size, stride, padding,
            dilation, groups, bias, padding_mode, device, dtype)
        self.winsize = self.kernel_size * self.kernel_size
    
    def forward(self, input_opt, input_sar: torch.Tensor, weight_in):
        # input : (B, C, H, W)
        # weight: (B, 1, H, W)
        weight_maskUpdater = torch.ones(1, 1, self.kernel_size, self.kernel_size, requires_grad=False).to(input_opt)
        sum_weight = F.conv2d(weight_in, weight_maskUpdater, bias=None,
                              stride=self.stride, padding=self.padding, dilation=self.dilation, groups=1)
        m_input = input_opt * weight_in
        s_input = input_sar * (1 - weight_in)
        m_out = self.conv1(torch.cat([m_input, s_input], dim=1))
        ratio = 2 * self.winsize / (self.winsize + sum_weight)
        if self.conv1.bias is not None:
            bias_view = self.conv1.bias.view(1, self.conv1.out_channels, 1, 1)
            m_out = (m_out - bias_view) * ratio + bias_view
        else:
            m_out = m_out * ratio
        s_out = self.conv2(input_sar)
        weight = F.max_pool2d(weight_in, kernel_size=self.kernel_size, 
                            stride=self.stride, padding=self.padding, dilation=self.dilation)
        return m_out, s_out, weight


# --------------------------
# PConv-BatchNorm-Activation
# --------------------------
class FConvBNActiv(nn.Module):
    def __init__(self, in_opt_channels, out_opt_channels, in_sar_channels, out_sar_channels,
                 bn=True, sample='none-3', activ='relu', bias=False):
        super(FConvBNActiv, self).__init__()
        self.in_opt_channels = in_opt_channels
        self.out_opt_channels = out_opt_channels
        self.in_sar_channels = in_sar_channels
        self.out_sar_channels = out_sar_channels
        if sample == 'down-7':
            self.conv = FusionConv2d(in_opt_channels, out_opt_channels, in_sar_channels, out_sar_channels,
                                      kernel_size=7, stride=2, padding=3, bias=bias)
        elif sample == 'down-5':
            self.conv = FusionConv2d(in_opt_channels, out_opt_channels, in_sar_channels, out_sar_channels,
                                      kernel_size=5, stride=2, padding=2, bias=bias)
        elif sample == 'down-3':
            self.conv = FusionConv2d(in_opt_channels, out_opt_channels, in_sar_channels, out_sar_channels,
                                      kernel_size=3, stride=2, padding=1, bias=bias)
        else:
            self.conv = FusionConv2d(in_opt_channels, out_opt_channels, in_sar_channels, out_sar_channels,
                                      kernel_size=3, stride=1, padding=1, bias=bias)
        if bn:
            self.bn = nn.InstanceNorm2d(out_opt_channels + out_sar_channels)
        if activ == 'relu':
            self.activation = nn.ReLU()
        elif activ == 'leaky':
            self.activation = nn.LeakyReLU(negative_slope=0.2)

    def disable_bn_grad(self):
        if hasattr(self, 'bn'):
            for param in self.bn.parameters():
                param.requires_grad = False

    def forward(self, images_opt, images_sar, masks):
        images_opt, images_sar, masks = self.conv(images_opt, images_sar, masks)
        images = torch.cat([images_opt, images_sar], dim=1)
        if hasattr(self, 'bn'):
            images = self.bn(images)
        if hasattr(self, 'activation'):
            images = self.activation(images)

        images_opt, images_sar = images[:, :self.out_opt_channels, :, :], images[:, self.out_opt_channels:, :, :]
        return images_opt, images_sar, masks


class ResnetBlock(nn.Module):
    """
    resnet block implementation
    """
    def __init__(self, in_channels, out_channels, hid_dim, alpha=0.1) -> None:
        super().__init__()
        hid_channnels = hid_dim
        m = list()
        m.append(nn.Conv2d(in_channels, hid_channnels, kernel_size=1, bias=True, stride=1))
        m.append(nn.ReLU(inplace=True))
        m.append(nn.Conv2d(hid_channnels, out_channels, kernel_size=1, bias=False, stride=1))
        self.net = nn.Sequential(*m)
        self.alpha = alpha

    def forward(self, x):
        out = self.net(x)
        out = self.alpha * out + x
        return out


# ------------
# CR Model.
# ------------
class SCTCR(nn.Module):
    def __init__(self, in_sar_channels=2, in_opt_channels=13, opt_dim=128, sar_dim=128, num_trans_layer=12, num_heads=8,
                  up_sampling_node='nearest', use_retnet=False, use_transformer=True, use_resblock=False):
        super(SCTCR, self).__init__()
        self.opt_dim = opt_dim
        self.sar_dim = sar_dim
        self.up_sampling_node = up_sampling_node
        self.use_retnet = use_retnet
        self.use_transformer = use_transformer
        self.attn = PixelAttention(3)
        self.ec_images_1 = FConvBNActiv(in_opt_channels, opt_dim, in_sar_channels, sar_dim, bn=False, sample='down-7')
        self.ec_images_2 = FConvBNActiv(opt_dim, 2 * opt_dim, sar_dim, 2 * sar_dim, sample='down-5')
        self.ec_images_3 = FConvBNActiv(2 * opt_dim, 4 * opt_dim, 2 * sar_dim, 4 * sar_dim, sample='down-5')
        self.ec_images_4 = FConvBNActiv(4 * opt_dim, 8 * opt_dim, 4 * sar_dim, 8 * sar_dim, sample='down-3')
        self.dc_images_4 = FConvBNActiv(12 * opt_dim, 4 * opt_dim, 12 * sar_dim, 4 * sar_dim, activ='leaky')
        self.dc_images_3 = FConvBNActiv(6 * opt_dim, 2 * opt_dim, 6 * sar_dim, 2 * sar_dim, activ='leaky')
        self.dc_images_2 = FConvBNActiv(3 * opt_dim, opt_dim, 3 * sar_dim, sar_dim, activ='leaky')
        self.dc_images_1 = FConvBNActiv(opt_dim + in_opt_channels, in_opt_channels, sar_dim + in_sar_channels, 
                                        in_sar_channels, bn=False, sample='none-3', activ=None, bias=True)
        self.pixconv = nn.Conv2d(in_opt_channels + in_sar_channels, in_opt_channels, 1, 1, 0)
        self.tanh = nn.Tanh()
        hidden_dim = 8*(opt_dim + sar_dim)
        if use_retnet:
            self.transformer = RetNet(layers=num_trans_layer, hidden_dim=hidden_dim, ffn_size=2*hidden_dim, heads=num_heads)
        elif use_transformer:
            mm = TransformerEncoderLayer(d_model=hidden_dim, nhead=num_heads, dim_feedforward=2*hidden_dim)
            self.pos_embd = PositionEmbeddingLearned(num_pos_feats=hidden_dim)
            self.transformer = VisionTransformerEncoder(mm, num_trans_layer)
        m = list()
        for _ in range(4):
            m.append(nn.Conv2d(2, 1, 3, 1, 1))
        self.wconv = nn.ModuleList(m)
        self.use_resblock = use_resblock
        if use_resblock:
            self.resblock = ResnetBlock(in_opt_channels, in_opt_channels, 64)

    def disable_bn_grad(self):
        self.ec_images_1.disable_bn_grad()
        self.ec_images_2.disable_bn_grad()
        self.ec_images_3.disable_bn_grad()
        self.ec_images_4.disable_bn_grad()

    def disable_unet_grad(self):
        for param in self.parameters():
            param.requires_grad = False
        for param in self.transformer.parameters():
            param.requires_grad = True

    def forward(self, sar, cld, return_weight=False):
        if cld.shape[1] == 3: # RGB
            input_masks = self.attn(cld)
        else: # full bands
            input_masks = self.attn(cld[:, [3,2,1], :, :])
        
        ec_images = {}
        ec_images['ec_images_0'], ec_images['ec_images_sar_0'], ec_images['ec_images_masks_0'] = \
            cld, sar, input_masks
        ec_images['ec_images_1'], ec_images['ec_images_sar_1'], ec_images['ec_images_masks_1'] = \
            self.ec_images_1(cld, sar, input_masks)
        ec_images['ec_images_2'], ec_images['ec_images_sar_2'], ec_images['ec_images_masks_2'] = \
            self.ec_images_2(ec_images['ec_images_1'], ec_images['ec_images_sar_1'], ec_images['ec_images_masks_1'])
        ec_images['ec_images_3'], ec_images['ec_images_sar_3'], ec_images['ec_images_masks_3'] = \
            self.ec_images_3(ec_images['ec_images_2'], ec_images['ec_images_sar_2'], ec_images['ec_images_masks_2'])
        ec_images['ec_images_4'], ec_images['ec_images_sar_4'], ec_images['ec_images_masks_4'] = \
            self.ec_images_4(ec_images['ec_images_3'], ec_images['ec_images_sar_3'], ec_images['ec_images_masks_3'])
        
        x = torch.cat([ec_images['ec_images_4'], ec_images['ec_images_sar_4']], dim=1)

        if self.use_retnet:
            x = self.transformer.forward(x)
        else:
            pos = self.pos_embd(x, None)
            pos = pos.flatten(-2).permute(0, 2, 1)
            x = self.transformer.forward(x, pos=pos)

        ec_images['ec_images_4'], ec_images['ec_images_sar_4'] = x[:, :8*self.opt_dim, :, :], x[:, 8*self.opt_dim:, :, :]
        
        # --------------
        # images decoder
        # --------------
        dc_images, dc_images_sar, dc_images_masks = ec_images['ec_images_4'], ec_images['ec_images_sar_4'], ec_images['ec_images_masks_4']
        for i in range(4, 0, -1):
            ec_images_skip = 'ec_images_{:d}'.format(i - 1)
            ec_images_skip_sar = 'ec_images_sar_{:d}'.format(i - 1)
            ec_images_masks = 'ec_images_masks_{:d}'.format(i - 1)
            
            dc_conv = 'dc_images_{:d}'.format(i)
            
            dc_images = F.interpolate(dc_images, scale_factor=2, mode=self.up_sampling_node)
            dc_images_sar = F.interpolate(dc_images_sar, scale_factor=2, mode=self.up_sampling_node)
            dc_images_masks = F.interpolate(dc_images_masks, scale_factor=2, mode=self.up_sampling_node)
            
            dc_images = torch.cat((dc_images, ec_images[ec_images_skip]), dim=1)
            dc_images_sar = torch.cat((dc_images_sar, ec_images[ec_images_skip_sar]), dim=1)
            dc_images_masks = torch.cat((dc_images_masks, ec_images[ec_images_masks]), dim=1)

            dc_images_masks = self.wconv[4-i](dc_images_masks)
            dc_images, dc_images_sar, dc_images_masks = getattr(self, dc_conv)(dc_images, dc_images_sar, dc_images_masks)

        dc_images = self.pixconv(torch.cat([dc_images, dc_images_sar], dim=1))
        outputs = self.tanh(dc_images)
        outputs = cld * input_masks + outputs * (1 - input_masks)

        if self.use_resblock:
            outputs = self.resblock(outputs)
        
        if self.training == False:
            outputs = torch.clamp(outputs, 0, 1)
        
        if return_weight:
            return outputs, input_masks
        else:
            return outputs