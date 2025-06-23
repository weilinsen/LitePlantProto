import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torch
import numpy as np
from utils import cosine_similarity, euclidean_dist_similarity
from models.SpatialFormer import SFSA
from torchvision import models
from models.pvt.pvtv2 import pvt_v2_b2
from models.Resnet.Resnet import resnet50, resnet18
from models.VGG.VGG import vgg16
import math
from models.HBP import HBP
from models.ODConv.odconv import ODConv2d
from models.neck import tSF
# from models.modeling import VisionTransformer, CONFIGS
from models.EfficientNet.efficientnet import efficientnet_b0, efficientnet_b7
from shufflenet import *
from models.coodinateAtt import CoordAtt
# import torchvision2
# from models_vit import vit_base_patch16
# from model_vit import vit_base_patch16
# from timm.models.layers import trunc_normal_
class ChannelAttentionModule(nn.Module):
    def __init__(self, channel, ratio=16):
        super(ChannelAttentionModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.shared_MLP = nn.Sequential(
            nn.Conv2d(channel, channel // ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(channel // ratio, channel, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avgout = self.shared_MLP(self.avg_pool(x))
        maxout = self.shared_MLP(self.max_pool(x))
        return self.sigmoid(avgout + maxout)

def KL_dist(x, y):
    """
    Args:
        x (torch.Tensor): shape (n, d). n usually n_way*n_query
        y (torch.Tensor): shape (m, d). m usually n_way
    Returns:
        torch.Tensor: shape(n, 1). For each query, the KL divergence to each centroid
    """
    x = x / x.sum(1).view(x.size(0), 1).expand(x.size(0), x.size(1))
    y = y / y.sum(1).view(y.size(0), 1).expand(y.size(0), y.size(1))

    n = x.size(0)
    m = y.size(0)
    d = x.size(1)
    assert d == y.size(1)

    x = x.unsqueeze(1).expand(n, m, d)
    y = y.unsqueeze(0).expand(n, m, d)

    # Since log (0) is negative infinity, let's add a small value to avoid it
    eps = 0.0001

    return -(x * torch.log((x+eps) / (y+eps))).sum(2)*100
class SpatialAttentionModule(nn.Module):
    def __init__(self):
        super(SpatialAttentionModule, self).__init__()
        self.conv2d = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=7, stride=1, padding=3)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avgout = torch.mean(x, dim=1, keepdim=True)
        maxout, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avgout, maxout], dim=1)
        out = self.sigmoid(self.conv2d(out))
        return out


class CBAM(nn.Module):
    def __init__(self, channel):
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttentionModule(channel)
        self.spatial_attention = SpatialAttentionModule()

    def forward(self, x):
        out = self.channel_attention(x) * x
        out = self.spatial_attention(out) * out
        return out


import torch
from torch import nn


class ResBlock(nn.Module):
    def __init__(self, n_chans):
        super(ResBlock, self).__init__()
        self.conv = nn.Conv2d(n_chans, n_chans,
                              kernel_size=3, padding=1,
                              bias=False)
        self.batch_norm = nn.BatchNorm2d(num_features=n_chans)  # <5>
        torch.nn.init.kaiming_normal_(self.conv.weight,
                                      nonlinearity='relu')  # <6>
        torch.nn.init.constant_(self.batch_norm.weight, 0.5)  # <7>
        torch.nn.init.zeros_(self.batch_norm.bias)

    def forward(self, x):
        out = self.conv(x)
        out = self.batch_norm(out)
        out = torch.relu(out)
        return out + x

class ECA(nn.Module):
    def __init__(self,in_channel,gamma=2,b=1):
        super(ECA, self).__init__()
        k=int(abs((math.log(in_channel,2)+b)/gamma))
        kernel_size=k if k % 2 else k+1
        padding=kernel_size//2
        self.pool=nn.AdaptiveAvgPool2d(output_size=1)
        self.conv=nn.Sequential(
            nn.Conv1d(in_channels=1,out_channels=1,kernel_size=kernel_size,padding=padding,bias=False),
            nn.Sigmoid()
        )

    def forward(self,x):
        out=self.pool(x)
        out=out.view(x.size(0),1,x.size(1))
        out=self.conv(out)
        out=out.view(x.size(0),x.size(1),1,1)
        return out*x

class SE_Block(nn.Module):
    def __init__(self, ch_in, reduction=16):
        super(SE_Block, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)  # 全局自适应池化
        self.fc = nn.Sequential(
            nn.Linear(ch_in, ch_in // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(ch_in // reduction, ch_in, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)  # squeeze操作
        y = self.fc(y).view(b, c, 1, 1)  # FC获取通道注意力权重，是具有全局信息的
        return x * y.expand_as(x)  # 注意力作用每一个通道上

class SimAM(torch.nn.Module):
    def __init__(self, e_lambda=1e-4):
        super(SimAM, self).__init__()

        self.activaton = nn.Sigmoid()
        self.e_lambda = e_lambda

    def __repr__(self):
        s = self.__class__.__name__ + '('
        s += ('lambda=%f)' % self.e_lambda)
        return s

    @staticmethod
    def get_module_name():
        return "simam"

    def forward(self, x):
        b, c, h, w = x.size()

        n = w * h - 1

        x_minus_mu_square = (x - x.mean(dim=[2, 3], keepdim=True)).pow(2)
        y = x_minus_mu_square / (4 * (x_minus_mu_square.sum(dim=[2, 3], keepdim=True) / n + self.e_lambda)) + 0.5

        return x * self.activaton(y)

class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        return x.view(x.size(0), -1)

class GlobalContextBlock(nn.Module):
    def __init__(self, in_channels, scale = 16):
        super(GlobalContextBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = self.in_channels//scale

        self.Conv_key = nn.Conv2d(self.in_channels, 1, 1)
        self.SoftMax = nn.Softmax(dim=1)

        self.Conv_value = nn.Sequential(
            nn.Conv2d(self.in_channels, self.out_channels, 1),
            nn.LayerNorm([self.out_channels, 1, 1]),
            nn.ReLU(),
            nn.Conv2d(self.out_channels, self.in_channels, 1),
        )

    def forward(self, x):
        b, c, h, w = x.size()
        # key -> [b, 1, H, W] -> [b, 1, H*W] ->  [b, H*W, 1]
        key = self.SoftMax(self.Conv_key(x).view(b, 1, -1).permute(0, 2, 1).view(b, -1, 1).contiguous())
        query = x.view(b, c, h*w)
        # [b, c, h*w] * [b, H*W, 1]
        concate_QK = torch.matmul(query, key)
        concate_QK = concate_QK.view(b, c, 1, 1).contiguous()
        value = self.Conv_value(concate_QK)
        out = x + value
        return out


class MS_CAM(nn.Module):
    '''
    单特征 进行通道加权,作用类似SE模块
    '''

    def __init__(self, channels=64, r=4):
        super(MS_CAM, self).__init__()
        inter_channels = int(channels // r)

        self.local_att = nn.Sequential(
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
        )

        self.global_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        xl = self.local_att(x)
        xg = self.global_att(x)
        xlg = xl + xg
        wei = self.sigmoid(xlg)
        return x * wei

class NonLocalBlock(nn.Module):
    def __init__(self, channel):
        super(NonLocalBlock, self).__init__()
        self.inter_channel = channel // 2
        self.conv_phi = nn.Conv2d(in_channels=channel, out_channels=self.inter_channel, kernel_size=1, stride=1,padding=0, bias=False)
        self.conv_theta = nn.Conv2d(in_channels=channel, out_channels=self.inter_channel, kernel_size=1, stride=1, padding=0, bias=False)
        self.conv_g = nn.Conv2d(in_channels=channel, out_channels=self.inter_channel, kernel_size=1, stride=1, padding=0, bias=False)
        self.softmax = nn.Softmax(dim=1)
        self.conv_mask = nn.Conv2d(in_channels=self.inter_channel, out_channels=channel, kernel_size=1, stride=1, padding=0, bias=False)

    def forward(self, x):
        # [N, C, H , W]
        b, c, h, w = x.size()
        # [N, C/2, H * W]
        x_phi = self.conv_phi(x).view(b, c, -1)
        # [N, H * W, C/2]
        x_theta = self.conv_theta(x).view(b, c, -1).permute(0, 2, 1).contiguous()
        x_g = self.conv_g(x).view(b, c, -1).permute(0, 2, 1).contiguous()
        # [N, H * W, H * W]
        mul_theta_phi = torch.matmul(x_theta, x_phi)
        mul_theta_phi = self.softmax(mul_theta_phi)
        # [N, H * W, C/2]
        mul_theta_phi_g = torch.matmul(mul_theta_phi, x_g)
        # [N, C/2, H, W]
        mul_theta_phi_g = mul_theta_phi_g.permute(0,2,1).contiguous().view(b,self.inter_channel, h, w)
        # [N, C, H , W]
        mask = self.conv_mask(mul_theta_phi_g)
        out = mask + x
        return out


class GAM_Attention(nn.Module):
    def __init__(self, in_channels, out_channels, rate=4):
        super(GAM_Attention, self).__init__()

        self.channel_attention = nn.Sequential(
            nn.Linear(in_channels, int(in_channels / rate)),
            nn.ReLU(inplace=True),
            nn.Linear(int(in_channels / rate), in_channels)
        )

        self.spatial_attention = nn.Sequential(
            nn.Conv2d(in_channels, int(in_channels / rate), kernel_size=7, padding=3),
            nn.BatchNorm2d(int(in_channels / rate)),
            nn.ReLU(inplace=True),
            nn.Conv2d(int(in_channels / rate), out_channels, kernel_size=7, padding=3),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, x):
        b, c, h, w = x.shape
        x_permute = x.permute(0, 2, 3, 1).view(b, -1, c)
        x_att_permute = self.channel_attention(x_permute).view(b, h, w, c)
        x_channel_att = x_att_permute.permute(0, 3, 1, 2)

        x = x * x_channel_att

        x_spatial_att = self.spatial_attention(x).sigmoid()
        out = x * x_spatial_att

        return out


class Inception(nn.Module):
    def __init__(self, in_channels):
        super(Inception, self).__init__()

        self.branch1 = nn.Sequential(

            nn.Conv2d(in_channels, 16, kernel_size=1)

        )
        self.branch5 = nn.Sequential(

            nn.Conv2d(in_channels, 16, kernel_size=1),

            nn.Conv2d(16, 24, kernel_size=5, padding=2)

        )
        self.branch3 = nn.Sequential(

            nn.Conv2d(in_channels, 16, kernel_size=1),

            nn.Conv2d(16, 24, kernel_size=3, padding=1),

            nn.Conv2d(24, 24, kernel_size=3, padding=1)

        )
        self.branch_pool = nn.Conv2d(in_channels, 24, kernel_size=1)

    def forward(self, x):
        branch1 = self.branch1(x)
        branch5 = self.branch5(x)
        branch3 = self.branch3(x)
        branch_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        branch_pool = self.branch_pool(branch_pool)

        return torch.cat((branch1, branch5, branch3, branch_pool), dim=1)


class ChannelAttention(nn.Module):
    def __init__(self, channel, reduction=16):
        super(ChannelAttention, self).__init__()
        mid_channel = channel // reduction

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.shared_MLP = nn.Sequential(
            nn.Linear(in_features=channel, out_features=mid_channel),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=mid_channel, out_features=channel)
        )

    def forward(self, x):
        avg = self.avg_pool(x).view(x.size(0), -1)
        out = self.shared_MLP(avg).unsqueeze(2).unsqueeze(3).expand_as(x)
        return out


class SpatialAttention(nn.Module):
    def __init__(self, channel, reduction=16, dilation_conv_num=2, dilation_rate=4):
        super(SpatialAttention, self).__init__()
        mid_channel = channel // reduction
        self.reduce_conv = nn.Sequential(
            nn.Conv2d(channel, mid_channel, kernel_size=1),
            nn.BatchNorm2d(mid_channel),
            nn.ReLU(inplace=True)
        )
        dilation_convs_list = []
        for i in range(dilation_conv_num):
            dilation_convs_list.append(
                nn.Conv2d(mid_channel, mid_channel, kernel_size=3, padding=dilation_rate, dilation=dilation_rate))
            dilation_convs_list.append(nn.BatchNorm2d(mid_channel))
            dilation_convs_list.append(nn.ReLU(inplace=True))
        self.dilation_convs = nn.Sequential(*dilation_convs_list)
        self.final_conv = nn.Conv2d(mid_channel, channel, kernel_size=1)

    def forward(self, x):
        y = self.reduce_conv(x)
        x = self.dilation_convs(y)
        out = self.final_conv(y)  # .expand_as(x)
        return out


class BAM(nn.Module):
    """
        BAM: Bottleneck Attention Module
        https://arxiv.org/pdf/1807.06514.pdf
    """

    def __init__(self, channel):
        super(BAM, self).__init__()
        self.channel_attention = ChannelAttention(channel)
        self.spatial_attention = SpatialAttention(channel)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        att = 1 + self.sigmoid(self.channel_attention(x) * self.spatial_attention(x))
        return att * x

# class PvtC2F(nn.Module):
#     def __init__(self):
#         super(PvtC2F, self).__init__()
#         self.backbone = pvt_v2_b2()  # [64, 128, 320, 512]
#         path = './pvt_v2_b2.pth'
#         save_model = torch.load(path)
#         model_dict = self.backbone.state_dict()
#         state_dict = {k: v for k, v in save_model.items() if k in model_dict.keys()}
#         model_dict.update(state_dict)
#         self.backbone.load_state_dict(model_dict)
#
#     def forward(self, x):
#         raw_shape = x.shape
#         x = x.view(-1, *raw_shape[-3:])
#
#         x1, x2, x3, x4 = self.backbone(x)
#         print(x1.shape)
#         print(x2.shape)
#         print(x3.shape)
#         print(x4.shape)
#         # x1:bs, 64, 88, 88
#         # x2:bs,128, 44, 44
#         # x3:bs,320, 22, 22
#         # x4:bs,512, 11, 11

class SubSpace(nn.Module):
    """
    Subspace class.
    ...
    Attributes
    ----------
    nin : int
        number of input feature volume.
    Methods
    -------
    __init__(nin)
        initialize method.
    forward(x)
        forward pass.
    """

    def __init__(self, nin):
        super(SubSpace, self).__init__()
        self.conv_dws = nn.Conv2d(
            nin, nin, kernel_size=1, stride=1, padding=0, groups=nin
        )
        self.bn_dws = nn.BatchNorm2d(nin, momentum=0.9)
        self.relu_dws = nn.ReLU(inplace=False)

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)

        self.conv_point = nn.Conv2d(
            nin, 1, kernel_size=1, stride=1, padding=0, groups=1
        )
        self.bn_point = nn.BatchNorm2d(1, momentum=0.9)
        self.relu_point = nn.ReLU(inplace=False)

        self.softmax = nn.Softmax(dim=2)

    def forward(self, x):
        out = self.conv_dws(x)
        out = self.bn_dws(out)
        out = self.relu_dws(out)

        out = self.maxpool(out)

        out = self.conv_point(out)
        out = self.bn_point(out)
        out = self.relu_point(out)

        m, n, p, q = out.shape
        out = self.softmax(out.view(m, n, -1))
        out = out.view(m, n, p, q)

        out = out.expand(x.shape[0], x.shape[1], x.shape[2], x.shape[3])

        out = torch.mul(out, x)

        out = out + x

        return out

import torch
from torch import nn

class EMA(nn.Module):
    def __init__(self, channels, factor=8):
        super(EMA, self).__init__()
        self.groups = factor
        assert channels // self.groups > 0
        self.softmax = nn.Softmax(-1)
        self.agp = nn.AdaptiveAvgPool2d((1, 1))
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        self.gn = nn.GroupNorm(channels // self.groups, channels // self.groups)
        self.conv1x1 = nn.Conv2d(channels // self.groups, channels // self.groups, kernel_size=1, stride=1, padding=0)
        self.conv3x3 = nn.Conv2d(channels // self.groups, channels // self.groups, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        b, c, h, w = x.size()
        group_x = x.reshape(b * self.groups, -1, h, w)  # b*g,c//g,h,w
        x_h = self.pool_h(group_x)
        x_w = self.pool_w(group_x).permute(0, 1, 3, 2)
        hw = self.conv1x1(torch.cat([x_h, x_w], dim=2))
        x_h, x_w = torch.split(hw, [h, w], dim=2)
        x1 = self.gn(group_x * x_h.sigmoid() * x_w.permute(0, 1, 3, 2).sigmoid())
        x2 = self.conv3x3(group_x)
        x11 = self.softmax(self.agp(x1).reshape(b * self.groups, -1, 1).permute(0, 2, 1))
        x12 = x2.reshape(b * self.groups, c // self.groups, -1)  # b*g, c//g, hw
        x21 = self.softmax(self.agp(x2).reshape(b * self.groups, -1, 1).permute(0, 2, 1))
        x22 = x1.reshape(b * self.groups, c // self.groups, -1)  # b*g, c//g, hw
        weights = (torch.matmul(x11, x12) + torch.matmul(x21, x22)).reshape(b * self.groups, 1, h, w)
        return (group_x * weights.sigmoid()).reshape(b, c, h, w)


class DepthWiseConv(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, padding):
        # 这一行千万不要忘记
        super(DepthWiseConv, self).__init__()

        # 逐通道卷积
        self.depth_conv = nn.Conv2d(in_channels=in_channel,
                                    out_channels=in_channel,
                                    kernel_size=kernel_size,
                                    stride=1,
                                    padding=padding,
                                    groups=in_channel)
        # groups是一个数，当groups=in_channel时,表示做逐通道卷积

        # 逐点卷积
        self.point_conv = nn.Conv2d(in_channels=in_channel,
                                    out_channels=out_channel,
                                    kernel_size=1,
                                    stride=1,
                                    padding=0,
                                    groups=1)


def forward(self, input):
    out = self.depth_conv(input)
    print(out.shape)
    out = self.point_conv(out)
    return out

class ConvEncoderC2F(nn.Module):
    def __init__(self, input_img_channels, hidden_channels):
        super(ConvEncoderC2F, self).__init__()
        self.hidden_channels = hidden_channels
        self.input_img_channels = input_img_channels
        self.cb1 = self.conv_block(self.input_img_channels, self.hidden_channels)
        self.cb2 = self.conv_block(self.hidden_channels, self.hidden_channels)
        self.cb3 = self.conv_block(self.hidden_channels, self.hidden_channels)
        self.cb4 = self.conv_block(self.hidden_channels, self.hidden_channels)

        self.rfb1_1 = RFB_modified(self.hidden_channels, self.hidden_channels)
        self.rfb2_1 = RFB_modified(self.hidden_channels, self.hidden_channels)
        self.rfb3_1 = RFB_modified(self.hidden_channels, self.hidden_channels)
        self.rfb4_1 = RFB_modified(self.hidden_channels, self.hidden_channels)


        # 互增强模块
        self.feb1 = FEB(hidden_channels)
        self.feb2 = FEB(hidden_channels)
        self.feb3 = FEB(hidden_channels)

        # self.eca1 = DualAxisConv2d(hidden_channels)
        # self.eca2 = DualAxisConv2d(hidden_channels)
        # self.eca3 = DualAxisConv2d(hidden_channels)
        # self.eca1 = SE_Block(hidden_channels)
        # self.eca2 = SE_Block(hidden_channels)
        # self.eca3 = SE_Block(hidden_channels)
        # self.eca1 = nn.Identity()
        # self.eca2 = nn.Identity()
        # self.eca3 = nn.Identity()
        # self.eca1 = ResBlock(hidden_channels)
        # self.eca2 = ResBlock(hidden_channels)
        # self.eca3 = ResBlock(hidden_channels)



        self.eca1 = ECA(hidden_channels)
        self.eca2 = ECA(hidden_channels)
        self.eca3 = ECA(hidden_channels)

        # self.eca1 = GlobalContextBlock(hidden_channels)
        # self.eca2 = GlobalContextBlock(hidden_channels)
        # self.eca3 = GlobalContextBlock(hidden_channels)

        # self.eca1 = MS_CAM(hidden_channels)
        # self.eca2 = MS_CAM(hidden_channels)
        # self.eca3 = MS_CAM(hidden_channels)

        # self.eca1 = BAM(hidden_channels)
        # self.eca2 = BAM(hidden_channels)
        # self.eca3 = BAM(hidden_channels)

        # self.eca1 = GAM_Attention(hidden_channels, hidden_channels)
        # self.eca2 = GAM_Attention(hidden_channels, hidden_channels)
        # self.eca3 = GAM_Attention(hidden_channels, hidden_channels)





        # self.eca1 = NonLocalBlock(hidden_channels)
        # self.eca2 = NonLocalBlock(hidden_channels)
        # self.eca3 = NonLocalBlock(hidden_channels)





        # self.eca1 = CBAM(hidden_channels)
        # self.eca2 = CBAM(hidden_channels)
        # self.eca3 = CBAM(hidden_channels)
        self.ema = EMA(hidden_channels)

        # self.eca1 = SimAM()
        # self.eca2 = SimAM()
        # self.eca3 = SimAM()
        self.upconv3 = BasicConv2d(hidden_channels, hidden_channels, kernel_size=3, stride=1, padding=1, relu=True)
        self.upconv2 = BasicConv2d(hidden_channels, hidden_channels, kernel_size=3, stride=1, padding=1, relu=True)
        self.upconv1 = BasicConv2d(hidden_channels, hidden_channels, kernel_size=3, stride=1, padding=1, relu=True)
        self.flatt = Flatten()

    def forward(self, x):
        raw_shape = x.shape
        x = x.view(-1, *raw_shape[-3:])

        x1 = self.cb1(x)
        x2 = self.cb2(x1)
        # x2 = self.sim(x2)

        x3 = self.cb3(x2)
        x4 = self.cb4(x3)


        x1_1 = self.rfb1_1(x1)  # channel -> 32
        x2_1 = self.rfb2_1(x2)  # channel -> 32
        x3_1 = self.rfb3_1(x3)  # channel -> 32
        x4_1 = self.rfb4_1(x4)  # channel -> 32

        x43 = self.feb3(x3_1, x4_1)
        out43 = self.upconv3(self.eca3(x43) + x43)
        # x = self.flatt(out43)

        x432 = self.feb2(x2_1, out43)
        out432 = self.upconv2(self.eca2(x432) + x432)
        #
        # x = self.flatt(out432)

        x4321 = self.feb1(x1_1, out432)
        out4321 = self.upconv1(self.eca1(x4321) + x4321)

        # out4321 = self.ema(out4321)



        x = self.flatt(out4321)
        x = x.view(*raw_shape[:-3], -1)
        return x

    @staticmethod
    def conv_block(in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=7, padding=3),
            # DepthWiseConv(in_channels, out_channels, 7, 3),
            # ODConv2d(in_channels, out_channels, kernel_size=7, stride=1,
            #          padding=3, reduction=0.0625, kernel_num=1),
            # GhostModule(in_channels, out_channels, 7),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

class ConvEncoder(nn.Module):
    def __init__(self, input_img_channels, hidden_channels):
        super(ConvEncoder, self).__init__()
        self.hidden_channels = hidden_channels
        self.input_img_channels = input_img_channels

        # Spatial transformer localization-network
        self.localization = nn.Sequential(
            nn.Conv2d(64, 8, kernel_size=3),

            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True),
            nn.Conv2d(8, 10, kernel_size=3),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True)
        )

        # Regressor for the 3 * 2 affine matrix
        self.fc_loc = nn.Sequential(
            nn.Linear(10 * 2 * 2, 32),
            nn.ReLU(True),
            nn.Linear(32, 3 * 2)
        )

        # Initialize the weights/bias with identity transformation
        self.fc_loc[2].weight.data.zero_()
        self.fc_loc[2].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))

        self.encoder = nn.Sequential(
            self.conv_block(self.input_img_channels, self.hidden_channels),
            self.conv_block(self.hidden_channels, self.hidden_channels),
            self.conv_block(self.hidden_channels, self.hidden_channels),
            self.conv_block(self.hidden_channels, self.hidden_channels),

        )
        self.flatt = Flatten()



    # Spatial transformer network forward function
    def stn(self, x):
        xs = self.localization(x)
        # print(xs.shape)

        xs = xs.view(-1, 10 * 2 * 2)
        theta = self.fc_loc(xs)
        theta = theta.view(-1, 2, 3)

        grid = F.affine_grid(theta, x.size())
        x = F.grid_sample(x, grid)
        return x

    def forward(self, x):
        raw_shape = x.shape
        x = x.view(-1, *raw_shape[-3:])
        x = self.encoder(x)
        # x = self.stn(x)
        x = self.flatt(x)
        x = x.view(*raw_shape[:-3], -1)
        return x

    @staticmethod
    def conv_block(in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=7, padding=3),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
class CALayer(nn.Module):
    # reduction；降维比例为r=16
    def __init__(self, channel, reduction=16):
        super(CALayer, self).__init__()
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # feature channel downscale a undpscale --> channel weight
        self.conv_du = nn.Sequential(
            # channel // reduction，输出降维，即论文中的1x1xC/r
            nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            # channel，输出升维，即论文中的1x1xC
            nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        '''
        x就是 HxWxC 通道  y是权重
        y权重通过上面方式求出，然后 和x求乘积
        使得重要通道权重更大，不重要通道权重减小
        '''
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y
class Refine(torch.nn.Module):
    def __init__(self, channel):
        super(Refine, self).__init__()
        self.gc = GlobalContextBlock(channel)
        self.mp = nn.MaxPool2d(kernel_size=(2,2))
        self.avg = nn.AdaptiveAvgPool2d(1)
    def forward(self, x):
        x = self.gc(x)
        # x = self.avg(x)
        # x = soft_pool2d(x)
        x = self.mp(x)
        return x

# class ASFF(nn.Module):
#     def __init__(self, level, rfb=False):
#         super(ASFF, self).__init__()
#         self.level = level
#         # 输入的三个特征层的channels, 根据实际修改
#         self.dim = [512, 256, 256]
#         self.inter_dim = self.dim[self.level]
#         # 每个层级三者输出通道数需要一致
#         if level==0:
#             self.stride_level_1 = conv_bn_relu(self.dim[1], self.inter_dim, 3, 2)
#             self.stride_level_2 = conv_bn_relu(self.dim[2], self.inter_dim, 3, 2)
#             self.expand = conv_bn_relu(self.inter_dim, 1024, 3, 1)
#         elif level==1:
#             self.compress_level_0 = conv_bn_relu(self.dim[0], self.inter_dim, 1, 1)
#             self.stride_level_2 = conv_bn_relu(self.dim[2], self.inter_dim, 3, 2)
#             self.expand = conv_bn_relu(self.inter_dim, 512, 3, 1)
#         elif level==2:
#             self.compress_level_0 = conv_bn_relu(self.dim[0], self.inter_dim, 1, 1)
#             if self.dim[1] != self.dim[2]:
#                 self.compress_level_1 = conv_bn_relu(self.dim[1], self.inter_dim, 1, 1)
#             self.expand = add_conv(self.inter_dim, 256, 3, 1)
#         compress_c = 8 if rfb else 16
#         self.weight_level_0 = conv_bn_relu(self.inter_dim, compress_c, 1, 1)
#         self.weight_level_1 = conv_bn_relu(self.inter_dim, compress_c, 1, 1)
#         self.weight_level_2 = conv_bn_relu(self.inter_dim, compress_c, 1, 1)
#
#         self.weight_levels = nn.Conv2d(compress_c*3, 3, 1, 1, 0)
#
#   # 尺度大小 level_0 < level_1 < level_2
#     def forward(self, x_level_0, x_level_1, x_level_2):
#         # Feature Resizing过程
#         if self.level==0:
#             level_0_resized = x_level_0
#             level_1_resized = self.stride_level_1(x_level_1)
#             level_2_downsampled_inter =F.max_pool2d(x_level_2, 3, stride=2, padding=1)
#             level_2_resized = self.stride_level_2(level_2_downsampled_inter)
#         elif self.level==1:
#             level_0_compressed = self.compress_level_0(x_level_0)
#             level_0_resized =F.interpolate(level_0_compressed, 2, mode='nearest')
#             level_1_resized =x_level_1
#             level_2_resized =self.stride_level_2(x_level_2)
#         elif self.level==2:
#             level_0_compressed = self.compress_level_0(x_level_0)
#             level_0_resized =F.interpolate(level_0_compressed, 4, mode='nearest')
#             if self.dim[1] != self.dim[2]:
#                 level_1_compressed = self.compress_level_1(x_level_1)
#                 level_1_resized = F.interpolate(level_1_compressed, 2, mode='nearest')
#             else:
#                 level_1_resized =F.interpolate(x_level_1, 2, mode='nearest')
#             level_2_resized =x_level_2
#     # 融合权重也是来自于网络学习
#         level_0_weight_v = self.weight_level_0(level_0_resized)
#         level_1_weight_v = self.weight_level_1(level_1_resized)
#         level_2_weight_v = self.weight_level_2(level_2_resized)
#         levels_weight_v = torch.cat((level_0_weight_v, level_1_weight_v,
#                                      level_2_weight_v),1)
#         levels_weight = self.weight_levels(levels_weight_v)
#         levels_weight = F.softmax(levels_weight, dim=1)   # alpha产生
#     # 自适应融合
#         fused_out_reduced = level_0_resized * levels_weight[:,0:1,:,:]+\
#                             level_1_resized * levels_weight[:,1:2,:,:]+\
#                             level_2_resized * levels_weight[:,2:,:,:]
#
#         out = self.expand(fused_out_reduced)
#         return out

class BasicConv(nn.Module):
    def __init__(
        self,
        in_planes,
        out_planes,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        relu=True,
        bn=True,
        bias=False,
    ):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(
            in_planes,
            out_planes,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )
        self.bn = (
            nn.BatchNorm2d(out_planes, eps=1e-5, momentum=0.01, affine=True)
            if bn
            else None
        )
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x

class ZPool(nn.Module):
    def forward(self, x):
        return torch.cat( (torch.max(x,1)[0].unsqueeze(1), torch.mean(x,1).unsqueeze(1)), dim=1 )

class AttentionGate(nn.Module):
    def __init__(self):
        super(AttentionGate, self).__init__()
        kernel_size = 7
        self.compress = ZPool()
        self.conv = BasicConv(2, 1, kernel_size, stride=1, padding=(kernel_size-1) // 2, relu=False)
    def forward(self, x):
        x_compress = self.compress(x)
        x_out = self.conv(x_compress)
        scale = torch.sigmoid_(x_out)
        return x * scale

class TripletAttention(nn.Module):
    def __init__(self, no_spatial=False):
        super(TripletAttention, self).__init__()
        self.cw = AttentionGate()
        self.hc = AttentionGate()
        self.no_spatial=no_spatial
        if not no_spatial:
            self.hw = AttentionGate()
    def forward(self, x):
        x_perm1 = x.permute(0,2,1,3).contiguous()
        x_out1 = self.cw(x_perm1)
        x_out11 = x_out1.permute(0,2,1,3).contiguous()
        x_perm2 = x.permute(0,3,2,1).contiguous()
        x_out2 = self.hc(x_perm2)
        x_out21 = x_out2.permute(0,3,2,1).contiguous()
        if not self.no_spatial:
            x_out = self.hw(x)
            x_out = 1/3 * (x_out + x_out11 + x_out21)
        else:
            x_out = 1/2 * (x_out11 + x_out21)
        return x_out

class ASFF(nn.Module):
    def __init__(self, c1=512, c2=512):
        super(ASFF, self).__init__()
        channel = 16
        self.weight_v_1 = BasicConv2d(c1, channel, 1)
        self.weight_v_2 = BasicConv2d(c2, channel, 1)
        self.weight_levels = nn.Conv2d(channel*2, 2, 1, 1, 0)
        self.expand = BasicConv2d(512, 256, 3, 1, 1)

    def forward(self, x1, x2):
        # return x1+x2
        x1_weight_v = self.weight_v_1(x1)
        x2_weight_v = self.weight_v_2(x2)
        weight_v = torch.cat((x1_weight_v, x2_weight_v), 1)
        levels_weight = self.weight_levels(weight_v)
        levels_weight = F.softmax(levels_weight, dim=1)
        fused_out = x1 * levels_weight[:,0:1,:,:] + x2*levels_weight[:,1:2,:,:]
        # out = self.expand(fused_out)
        return fused_out

class HBP(nn.Module):
    def __init__(self):
        super(HBP, self).__init__()
        # self.backbone = torchvision.models.resnet50(pretrained=True)
        self.backbone = torchvision.models.vgg16(pretrained=True)

        self.features = self.backbone.features
        self.features_conv5 = torch.nn.Sequential(*list(self.features.children()))[:-7]
        self.features_conv5_1 = torch.nn.Sequential(*list(self.features.children())[-7:-5])

        # self.features_conv5_1 = torch.nn.Sequential(*list(self.features.children())[:-5])
        self.features_conv5_2 = torch.nn.Sequential(*list(self.features.children())[-5:-3])
        self.features_conv5_3 = torch.nn.Sequential(*list(self.features.children())[-3:-1])
        # self.features_conv5_4 = torch.nn.Sequential(*list(self.features.children())[-1])
        # print(self.features)
        # print(self.features_conv5_1)
        # print(self.features_conv5_2)
        # print(self.features_conv5_3)
        # self.sim = SimAM()
        self.sim = TripletAttention()
        self.refine1 = Refine(512)
        self.refine2 = Refine(512)
        self.refine3 = Refine(512)
        self.ASFF1 = ASFF()
        self.ASFF2 = ASFF()
        self.ASFF3 = ASFF()
        self.ca = CALayer(1536)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.max_gap = nn.AdaptiveMaxPool2d(1)
        self.conv = nn.Conv2d(1024, 1, 1)
        self.sigmoid = nn.Sigmoid()
        self.bilinear_proj_1 = torch.nn.Conv2d(512, 8192, kernel_size=1, bias=True)
        self.bilinear_proj_2 = torch.nn.Conv2d(512, 8192, kernel_size=1, bias=True)
        self.bilinear_proj_3 = torch.nn.Conv2d(512, 8192, kernel_size=1, bias=True)
        self.flatten = Flatten()

        # self.aspp = ASFF(1)

    def hbp_1_2(self, conv1, conv2):
        N = conv1.size()[0]
        proj_1 = self.bilinear_proj_1(conv1)
        proj_2 = self.bilinear_proj_2(conv2)
        X = proj_1 * proj_2
        X = torch.sum(X.view(X.size()[0], X.size()[1], -1), dim=2)
        X = X.view(N, 8192)
        X = torch.sign(X) * torch.sqrt(torch.abs(X) + 1e-5)
        X = torch.nn.functional.normalize(X)
        return X

    def hbp_1_3(self, conv1, conv3):
        N = conv1.size()[0]
        proj_1 = self.bilinear_proj_1(conv1)
        proj_3 = self.bilinear_proj_3(conv3)
        X = proj_1 * proj_3
        X = torch.sum(X.view(X.size()[0], X.size()[1], -1), dim=2)
        X = X.view(N, 8192)
        X = torch.sign(X) * torch.sqrt(torch.abs(X) + 1e-5)
        X = torch.nn.functional.normalize(X)
        return X

    def hbp_2_3(self, conv2, conv3):
        N = conv2.size()[0]
        proj_2 = self.bilinear_proj_2(conv2)
        proj_3 = self.bilinear_proj_3(conv3)
        X = proj_2 * proj_3
        X = torch.sum(X.view(X.size()[0], X.size()[1], -1), dim=2)
        X = X.view(N, 8192)
        X = torch.sign(X) * torch.sqrt(torch.abs(X) + 1e-5)
        X = torch.nn.functional.normalize(X)
        return X
    def forward(self, x):
        raw_shape = x.shape
        X = x.view(-1, *raw_shape[-3:])

        X = self.features_conv5(X)
        # X = self.sim(X)
        X_conv5_1 = self.features_conv5_1(X)
        X_conv5_2 = self.features_conv5_2(X_conv5_1)
        X_conv5_3 = self.features_conv5_3(X_conv5_2)
        # X_conv5_1 = self.refine1(X_conv5_1)
        # X_conv5_2 = self.refine2(X_conv5_2)
        # X_conv5_3 = self.refine3(X_conv5_3)

        # X_branch_1 = self.hbp_1_2(X_conv5_1, X_conv5_2)
        # X_branch_2 = self.hbp_1_3(X_conv5_1, X_conv5_3)
        # X_branch_3 = self.hbp_2_3(X_conv5_2, X_conv5_3)

        X_branch_1 = self.ASFF1(X_conv5_1, X_conv5_2)
        X_branch_2 = self.ASFF2(X_conv5_1, X_conv5_3)
        X_branch_3 = self.ASFF3(X_conv5_2, X_conv5_3)
        #
        X_branch = X_branch_1+ X_branch_2 + X_branch_3
        # X_branch = self.gap(X_branch)
        B, C, H, W = X_branch.shape

        gap_branch = self.gap(X_conv5_3).expand(B, C, H, W) # b c h w
        result = torch.cat((X_conv5_3, gap_branch), dim=1)  # B 2C H W
        # result = X_branch * gap_branch
        result = self.conv(result) # B 1 H W
        att = self.sigmoid(result) # b 1 h w
        out = (att * X_conv5_3)
        # out = (att * X_branch).view(B, C, -1) # B C HW
        # X_branch = torch.sum(out, dim=2)



        # X_branch = torch.cat([X_branch_1, X_branch_2, X_branch_3], dim=1)

        # out = self.aspp(X_conv5_1, X_conv5_2, X_conv5_3)
        # X = torch.cat((X_conv5_1, X_conv5_2, X_conv5_3), dim=1)
        # X = self.ca(X).view(X.shape[0], -1)

        # x = self.backbone(x)
        # x = self.backbone.conv1(x)
        # x = self.backbone.bn1(x)
        # x = self.backbone.relu(x)
        # x = self.backbone.maxpool(x)
        #
        # x = self.backbone.layer1(x)
        # x = self.backbone.layer2(x)
        # x = self.backbone.layer3(x)
        # x = self.backbone.layer4(x)
        x = self.flatten(X_branch)
        # x = self.encoder(x)
        x = x.view(*raw_shape[:-3], -1)
        return x

class Resnet18(nn.Module):
    def __init__(self):
        super(Resnet18, self).__init__()
        self.backbone = torchvision.models.resnet18(pretrained=True)
        # self.backbone = torchvision.models.vgg19_bn(pretrained=True)
        self.flatten = Flatten()

    def forward(self, x):
        raw_shape = x.shape
        x = x.view(-1, *raw_shape[-3:])

        # x = self.backbone(x)
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)

        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)


        x = self.flatten(x)
        # x = self.encoder(x)
        x = x.view(*raw_shape[:-3], -1)
        return x

class Resnet18W2(nn.Module):
    def __init__(self):
        super(Resnet18W2, self).__init__()
        self.backbone = torchvision.models.resnet18(pretrained=True)
        self.ta = TripletAttention()
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.max_gap = nn.AdaptiveMaxPool2d(1)
        self.conv = nn.Conv2d(1024, 1, 1)
        self.sigmoid = nn.Sigmoid()
        self.ASFF1 = ASFF(512, 512)
        self.ASFF2 = ASFF(512, 512)
        self.ASFF3 = ASFF(512, 512)
        self.upsample = cus_sample
        self.conv2 = nn.Conv2d(128, 512, 1)
        self.conv3 = nn.Conv2d(256, 512, 1)
        # self.backbone = torchvision.models.vgg19_bn(pretrained=True)
        self.flatten = Flatten()

    def forward(self, x):
        raw_shape = x.shape
        x = x.view(-1, *raw_shape[-3:])

        # x = self.backbone(x)
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)

        x1 = self.backbone.layer1(x)
        x2 = self.backbone.layer2(x1)
        x3 = self.backbone.layer3(x2)
        x4 = self.backbone.layer4(x3)


        x2 = self.upsample(x2, size=(x4.shape[2], x4.shape[3]))
        x3 = self.upsample(x3, size=(x4.shape[2], x4.shape[3]))
        x2 = self.conv2(x2)
        x3 = self.conv3(x3)

        X_branch_1 = self.ASFF1(x2, x3)
        X_branch_2 = self.ASFF2(x2, x4)
        X_branch_3 = self.ASFF3(x3, x4)
        x = X_branch_1 + X_branch_2 + X_branch_3
        B, C, H, W = x.shape

        gap_branch = self.gap(x).expand(B, C, H, W)  # b c h w
        result = torch.cat((x, gap_branch), dim=1)  # B 2C H W
        # result = X_branch * gap_branch
        result = self.conv(result)  # B 1 H W
        att = self.sigmoid(result)  # b 1 h w
        out = (att * x).view(B, C, -1)  # B C HW
        x = torch.sum(out, dim=2)


        x = self.flatten(x)
        # x = self.encoder(x)
        x = x.view(*raw_shape[:-3], -1)
        return x

class VGG19(nn.Module):
    def __init__(self):
        super(VGG19, self).__init__()
        self.backbone = torchvision.models.vgg19_bn(pretrained=True)
        self.features = self.backbone.features
        # self.backbone = torchvision.models.vgg19_bn(pretrained=True)
        self.flatten = Flatten()

    def forward(self, x):
        raw_shape = x.shape
        x = x.view(-1, *raw_shape[-3:])

        x = self.features(x)
        # x = self.backbone.conv1(x)
        # x = self.backbone.bn1(x)
        # x = self.backbone.relu(x)
        # x = self.backbone.maxpool(x)
        #
        # x = self.backbone.layer1(x)
        # x = self.backbone.layer2(x)
        # x = self.backbone.layer3(x)
        # x = self.backbone.layer4(x)
        x = self.flatten(x)
        # x = self.encoder(x)
        x = x.view(*raw_shape[:-3], -1)
        return x

class InceptionV3(nn.Module):
    def __init__(self):
        super(InceptionV3, self).__init__()
        self.backbone = torchvision.models.inception_v3(pretrained=True)
        # self.backbone = torchvision.models.vgg19_bn(pretrained=True)
        self.flatten = Flatten()

    def forward(self, x):
        raw_shape = x.shape
        x = x.view(-1, *raw_shape[-3:])

        x = self.backbone.Conv2d_1a_3x3(x)
        # 149 x 149 x 32
        x = self.backbone.Conv2d_2a_3x3(x)
        # 147 x 147 x 32
        x = self.backbone.Conv2d_2b_3x3(x)
        # 147 x 147 x 64
        x = F.max_pool2d(x, kernel_size=3, stride=2)
        # 73 x 73 x 64
        x = self.backbone.Conv2d_3b_1x1(x)
        # 73 x 73 x 80
        x = self.backbone.Conv2d_4a_3x3(x)
        # 71 x 71 x 192
        x = F.max_pool2d(x, kernel_size=3, stride=2)
        # 35 x 35 x 192
        x = self.backbone.Mixed_5b(x)
        # 35 x 35 x 256
        x = self.backbone.Mixed_5c(x)
        # 35 x 35 x 288
        x = self.backbone.Mixed_5d(x)
        # 35 x 35 x 288
        x = self.backbone.Mixed_6a(x)
        # 17 x 17 x 768
        x = self.backbone.Mixed_6b(x)
        # 17 x 17 x 768
        x = self.backbone.Mixed_6c(x)
        # 17 x 17 x 768
        x = self.backbone.Mixed_6d(x)
        # 17 x 17 x 768
        x = self.backbone.Mixed_6e(x)
        # 17 x 17 x 768
        # 17 x 17 x 768
        x = self.backbone.Mixed_7a(x)
        # 8 x 8 x 1280
        x = self.backbone.Mixed_7b(x)
        # 8 x 8 x 2048
        x = self.backbone.Mixed_7c(x)

        # x = self.backbone(x)
        # print(x.shape)
        # x = self.backbone.conv1(x)
        # x = self.backbone.bn1(x)
        # x = self.backbone.relu(x)
        # x = self.backbone.maxpool(x)
        #
        # x = self.backbone.layer1(x)
        # x = self.backbone.layer2(x)
        # x = self.backbone.layer3(x)
        # x = self.backbone.layer4(x)
        x = self.flatten(x)
        # x = self.encoder(x)
        x = x.view(*raw_shape[:-3], -1)
        return x


def cus_sample(feat, **kwargs):
    """
    :param feat: 输入特征
    :param kwargs: size或者scale_factor
    """
    assert len(kwargs.keys()) == 1 and list(kwargs.keys())[0] in ["size", "scale_factor"]
    return F.interpolate(feat, **kwargs, mode="bilinear", align_corners=False)

class FEB(nn.Module):
    def __init__(self, channel):
        super(FEB, self).__init__()

        self.upsample = cus_sample
        self.conv = nn.Conv2d(channel*2, channel, 3, padding=1)
        # self.conv =  ODConv2d(channel*2, channel, kernel_size = 3, stride=1,
        #                      padding=1, reduction=0.0625, kernel_num=1)
        self.sigmoid = nn.Sigmoid()
        channel = 16
        self.weight_v_1 = BasicConv2d(64, channel, 1)
        self.weight_v_2 = BasicConv2d(64, channel, 1)
        self.weight_levels = nn.Conv2d(channel * 2, 2, 1, 1, 0)
    def forward(self, x, y):
        y = self.upsample(y, scale_factor=2)
        # print(x.shape)
        # print(y.shape)
        # return x + y

        # N = x.size()[0]
        # X = x * y
        # X = torch.sum(X.view(X.size()[0], X.size()[1], -1), dim=2)
        # print(X.shape)
        # X = X.view(N, 8192)
        # X = torch.sign(X) * torch.sqrt(torch.abs(X) + 1e-5)
        # X = torch.nn.functional.normalize(X)
        # return X
        x1_weight_v = self.weight_v_1(x)
        x2_weight_v = self.weight_v_2(y)
        weight_v = torch.cat((x1_weight_v, x2_weight_v), 1)
        levels_weight = self.weight_levels(weight_v)
        levels_weight = F.softmax(levels_weight, dim=1)
        fused_out = x * levels_weight[:, 0:1, :, :] + y * levels_weight[:, 1:2, :, :]
        return fused_out
        # return x+y

        # xy = torch.cat((x, y), 1)
        # # out = self.conv(xy)
        # xy_map = self.sigmoid(self.conv(xy))
        # ex = x * xy_map + x
        # ey = y * xy_map + y
        # return ex + ey
        # return out

class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, relu=False, bn=True):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)

        self.bn = nn.BatchNorm2d(out_planes) if bn else None
        self.relu = nn.ReLU(inplace=True) if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x

class RFB_modified(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(RFB_modified, self).__init__()
        self.relu = nn.ReLU(True)
        self.branch0 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
        )
        self.branch1 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 3), padding=(0, 1)),
            BasicConv2d(out_channel, out_channel, kernel_size=(3, 1), padding=(1, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=3, dilation=3)
        )
        self.branch2 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 5), padding=(0, 2)),
            BasicConv2d(out_channel, out_channel, kernel_size=(5, 1), padding=(2, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=5, dilation=5)
        )
        self.branch3 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 7), padding=(0, 3)),
            BasicConv2d(out_channel, out_channel, kernel_size=(7, 1), padding=(3, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=7, dilation=7)
        )
        self.conv_cat = BasicConv2d(4 * out_channel, out_channel, 3, padding=1)
        self.conv_res = BasicConv2d(in_channel, out_channel, 1)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        # print('---')
        # print(x0.shape)
        # print(x1.shape)
        # print(x2.shape)
        # print(x3.shape)

        x_cat = self.conv_cat(torch.cat((x0, x1, x2, x3), 1))

        x = self.relu(x_cat + self.conv_res(x))
        return x

class DualAxisConv2d(nn.Module):
    def __init__(self, inchannel, kernel=3, band_kernel_size=13):
        super().__init__()
        gc = int(inchannel * (1/2))
        spatialGC = int(gc * (1/2))
        self.split_indexes = (gc, gc)
        self.spilt_spatial_indexes = (spatialGC, spatialGC)
        self.spatialX = nn.Conv2d(spatialGC, spatialGC, kernel_size=(1, band_kernel_size), padding=(0, band_kernel_size//2),
                                  groups=spatialGC)
        self.spatialY = nn.Conv2d(spatialGC, spatialGC, kernel_size=(band_kernel_size, 1), padding=(band_kernel_size // 2, 0),
                                  groups=spatialGC)
        self.deepConv = nn.Sequential(
            nn.Conv2d(gc, gc, kernel, padding=1, groups=gc),
            nn.Conv2d(gc, gc, 1)
        )

    def forward(self, x):
        u = x.clone()
        x_sf, x_wf = torch.split(x, self.split_indexes, dim=1)
        sX, sY = torch.split(x_sf, self.spilt_spatial_indexes, dim=1)
        sX = self.spatialX(sX)
        sY = self.spatialY(sY)
        spatialF = torch.cat((sX, sY), dim=1)
        deepF = self.deepConv(x_wf)
        attn = torch.cat((spatialF, deepF), dim=1)
        return u * attn

class Resnet18C2F(nn.Module):
    def __init__(self, channel=64):
        super(Resnet18C2F, self).__init__()

        self.backbone = resnet18(pretrained=True)

        # for (name, param) in self.backbone.named_parameters():
        #     param.requires_grad = False

        # self.rfb1_1 = RFB_modified(64, channel)
        # self.rfb2_1 = RFB_modified(128, channel)
        # self.rfb3_1 = RFB_modified(256, channel)
        # self.rfb4_1 = RFB_modified(512, channel)
        self.rfb1_1 = nn.Conv2d(64, channel, 1)
        self.rfb2_1 = nn.Conv2d(128, channel, 1)
        self.rfb3_1 = nn.Conv2d(256, channel, 1)
        self.rfb4_1 = nn.Conv2d(512, channel, 1)

        # 互增强模块
        self.feb1 = FEB(channel)
        self.feb2 = FEB(channel)
        self.feb3 = FEB(channel)

        self.eca1 = ECA(channel)
        self.eca2 = ECA(channel)
        self.eca3 = ECA(channel)
        # self.eca1 = nn.Identity()
        # self.eca2 = nn.Identity()
        # self.eca3 = nn.Identity()
        self.upconv3 = BasicConv2d(channel, channel, kernel_size=3, stride=1, padding=1, relu=True)
        self.upconv2 = BasicConv2d(channel, channel, kernel_size=3, stride=1, padding=1, relu=True)
        self.upconv1 = BasicConv2d(channel, channel, kernel_size=3, stride=1, padding=1, relu=True)



        self.flatten = Flatten()


    def forward(self, x):
        raw_shape = x.shape
        x = x.view(-1, *raw_shape[-3:])
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)

        x1 = self.backbone.layer1(x)
        x2 = self.backbone.layer2(x1)
        x3 = self.backbone.layer3(x2)
        x4 = self.backbone.layer4(x3)
        # print(x1.shape)
        # print(x2.shape)
        # print(x3.shape)
        # print(x4.shape)

        x1_1 = self.rfb1_1(x1)  # channel -> 32
        x2_1 = self.rfb2_1(x2)  # channel -> 32
        x3_1 = self.rfb3_1(x3)  # channel -> 32
        x4_1 = self.rfb4_1(x4)  # channel -> 32

        x43 = self.feb3(x3_1, x4_1)
        out43 = self.upconv3(self.eca3(x43) + x43)
        # x = self.flatten(out43)
        x432 = self.feb2(x2_1, out43)
        out432 = self.upconv2(self.eca2(x432) + x432)
        # x = self.flatten(out432)

        x4321 = self.feb1(x1_1, out432)
        out4321 = self.upconv1(self.eca1(x4321) + x4321)
        # out4321 = self.stn(out4321)
        x = self.flatten(out4321)
        # x = self.encoder(x)
        x = x.view(*raw_shape[:-3], -1)
        return x

class denseNet(nn.Module):
    def __init__(self):
        super(denseNet, self).__init__()
        self.backbone = torchvision.models.densenet121(pretrained=False)
        net_dict = self.backbone.state_dict()
        predict_model = torch.load('densenet121-a639ec97.'
                                   'pth')
        state_dict = {k: v for k, v in predict_model.items() if k in net_dict.keys()}  # 寻找网络中公共层，并保留预训练参数
        net_dict.update(state_dict)  # 将预训练参数更新到新的网络层
        self.backbone.load_state_dict(net_dict)
        self.flatten = Flatten()

    def forward(self, x):
        raw_shape = x.shape
        x = x.view(-1, *raw_shape[-3:])
        x = self.backbone.features(x)
        # out4321 = self.stn(out4321)
        x = self.flatten(x)
        # x = self.encoder(x)
        x = x.view(*raw_shape[:-3], -1)
        return x

class GhostModule(nn.Module):
    def __init__(self, inp, oup, kernel_size=1, ratio=2, dw_size=3, stride=1, relu=True):
        super(GhostModule, self).__init__()
        self.oup = oup
        init_channels = math.ceil(oup / ratio)
        new_channels = init_channels*(ratio-1)

        self.primary_conv = nn.Sequential(
            nn.Conv2d(inp, init_channels, kernel_size, stride, kernel_size//2, bias=False),
            nn.BatchNorm2d(init_channels),
            nn.ReLU(inplace=True) if relu else nn.Sequential(), )
  # cheap操作，注意利用了分组卷积进行通道分离
        self.cheap_operation = nn.Sequential(
            nn.Conv2d(init_channels, new_channels, dw_size, 1, dw_size//2, groups=init_channels, bias=False),
            nn.BatchNorm2d(new_channels),
            nn.ReLU(inplace=True) if relu else nn.Sequential(),)

    def forward(self, x):
        x1 = self.primary_conv(x)  #主要的卷积操作
        x2 = self.cheap_operation(x1) # cheap变换操作
        out = torch.cat([x1,x2], dim=1) # 二者cat到一起
        return out[:,:self.oup,:,:]


class shuffle(nn.Module):
    def __init__(self, group=2):
        super(shuffle, self).__init__()
        self.group = group

    def forward(self, x):
        """shuffle操作：[N,C,H,W] -> [N,g,C/g,H,W] -> [N,C/g,g,H,w] -> [N,C,H,W]"""
        num, channel, height, width = x.size()
        x = x.view(num, self.group, channel // self.group, height, width)
        x = x.permute(0, 2, 1, 3, 4)
        x = x.reshape(num, channel, height, width)
        return x


from torchvision.transforms import Resize
from torch.autograd._functions import Resize
class hbp_layer(nn.Module):
    def __init__(self, channel1, channel2, scale_factor=2):
        super(hbp_layer, self).__init__()
        self.out_channel = 8192
        self.scale_factor = scale_factor
        self.bilinear_proj_1 = torch.nn.Conv2d(channel1, self.out_channel, kernel_size=1, bias=True)
        self.bilinear_proj_2 = torch.nn.Conv2d(channel2, self.out_channel, kernel_size=1, bias=True)
        self.shuffleConv1 = nn.Sequential(
            nn.Conv2d(channel1, self.out_channel, kernel_size=1, bias=True, groups=4),
            shuffle(),
            nn.Conv2d(self.out_channel, self.out_channel, kernel_size=1, bias=True, groups=4)
        )

        self.shuffleConv2 = nn.Sequential(
            nn.Conv2d(channel2, self.out_channel, kernel_size=1, bias=True, groups=8),
            shuffle(),
            nn.Conv2d(self.out_channel, self.out_channel, kernel_size=1, bias=True, groups=8)
        )
        self.w = nn.Parameter(torch.ones(2))
        self.sim = SimAM()
        self.upsample = cus_sample

    # 将通道均匀打乱，111222 -> 121212
    def channel_shuffle(self, x, groups):
        batchsize, num_channels, height, width = x.size()
        channels_per_group = num_channels // groups

        # reshape
        x = x.view(batchsize, groups, channels_per_group, height, width)

        x = torch.transpose(x, 1, 2).contiguous()

        # flatten
        x = x.view(batchsize, -1, height, width)

        return x

    def forward(self, conv1, conv2):
        conv2 = self.upsample(conv2, size=(conv1.shape[2], conv1.shape[3]))
        # conv2 = conv2.resize_as_(conv1)
        # conv2 = Resize.apply(conv2, (conv1.shape[2], conv1.shape[3]))
        # print(conv2.shape)
        N = conv1.size()[0]
        # proj_1 = self.bilinear_proj_1(conv1)
        # proj_2 = self.bilinear_proj_2(conv2)
        # proj_1 = self.shuffleConv1(conv1)
        # proj_2 = self.bilinear_proj_2(conv2)
        proj_1 = self.shuffleConv1(conv1)
        proj_2 = self.shuffleConv2(conv2)


        # w1 = torch.exp(self.w[0]) / torch.sum(torch.exp(self.w))
        # w2 = torch.exp(self.w[1]) / torch.sum(torch.exp(self.w))
        # X = w1 * proj_1 * w2 * proj_2
        X = proj_1 * proj_2
        # X = self.avg(X)
        X = torch.sum(X.view(X.size()[0], X.size()[1], -1), dim=2)
        X = X.view(N, self.out_channel)
        X = torch.sign(X) * torch.sqrt(torch.abs(X) + 1e-5)
        X = torch.nn.functional.normalize(X)
        return X

class fhbp_layer(nn.Module):
    def __init__(self, channel1, channel2, scale_factor=2):
        super(fhbp_layer, self).__init__()
        self.out_channel = 8192
        self.scale_factor = scale_factor
        self.RANK_ATOMS = 1
        self.NUM_CLUSTER = 2048
        self.JOINT_EMB = self.RANK_ATOMS * self.NUM_CLUSTER
        self.bilinear_proj_1 = torch.nn.Linear(channel1, self.JOINT_EMB)
        self.bilinear_proj_2 = torch.nn.Linear(channel2, self.JOINT_EMB)
        self.avg1d = nn.AvgPool1d(kernel_size=784)
        self.upsample = cus_sample
    def forward(self, conv1, conv2):
        conv2 = self.upsample(conv2, size=(conv1.shape[2], conv1.shape[3]))
        # conv2 = conv2.resize_as_(conv1)
        # conv2 = Resize.apply(conv2, (conv1.shape[2], conv1.shape[3]))
        # print(conv2.shape)
        N = conv1.size()[0]
        w = conv1.shape[2]
        h = conv1.shape[3]
        c1 = conv1.shape[1]
        c2 = conv2.shape[1]
        conv1 = conv1.permute(0, 2, 3, 1).contiguous().view(-1, c1)
        conv2 = conv2.permute(0, 2, 3, 1).contiguous().view(-1, c2)
        proj_1 = self.bilinear_proj_1(conv1)
        proj_2 = self.bilinear_proj_2(conv2)
        # w1 = torch.exp(self.w[0]) / torch.sum(torch.exp(self.w))
        # w2 = torch.exp(self.w[1]) / torch.sum(torch.exp(self.w))
        # X = w1 * proj_1 * w2 * proj_2
        X = proj_1.mul(proj_2)
        X = X.view(-1, 1, self.NUM_CLUSTER, self.RANK_ATOMS)
        X = torch.squeeze(torch.sum(X, 3))

        zero = torch.zeros(X.shape).cuda()
        X = torch.mul(torch.sign(X), torch.max((torch.abs(X)-0.001/2), zero))

        X = X.view(N, h*w, -1)
        X = X.view(N, h*w, -1).permute(0, 2, 1)
        X = torch.squeeze(self.avg1d(X))
        X = torch.sqrt(F.relu(X)) - torch.sqrt(F.relu(-X))
        X = F.normalize(X, p=2, dim=1)
        # X = self.avg(X)
        # X = torch.sum(X.view(X.size()[0], X.size()[1], -1), dim=2)
        # X = X.view(N, self.out_channel)
        # X = torch.sign(X) * torch.sqrt(torch.abs(X) + 1e-5)
        # X = torch.nn.functional.normalize(X)
        return X

class DeformConv2d(nn.Module):
    def __init__(self, inc, outc, kernel_size=3, padding=1, stride=1, bias=None, modulation=False):
        """
        Args:
            modulation (bool, optional): If True, Modulated Defomable Convolution (Deformable ConvNets v2).
        """
        super(DeformConv2d, self).__init__()
        self.kernel_size = kernel_size
        self.padding = padding
        self.stride = stride
        self.zero_padding = nn.ZeroPad2d(padding)
        # conv则是实际进行的卷积操作，注意这里步长设置为卷积核大小，因为与该卷积核进行卷积操作的特征图是由输出特征图中每个点扩展为其对应卷积核那么多个点后生成的。
        self.conv = nn.Conv2d(inc, outc, kernel_size=kernel_size, stride=kernel_size, bias=bias)
        # p_conv是生成offsets所使用的卷积，输出通道数为卷积核尺寸的平方的2倍，代表对应卷积核每个位置横纵坐标都有偏移量。
        self.p_conv = nn.Conv2d(inc, 2 * kernel_size * kernel_size, kernel_size=3, padding=1, stride=stride)
        nn.init.constant_(self.p_conv.weight, 0)
        self.p_conv.register_backward_hook(self._set_lr)

        self.modulation = modulation  # modulation是可选参数,若设置为True,那么在进行卷积操作时,对应卷积核的每个位置都会分配一个权重。
        if modulation:
            self.m_conv = nn.Conv2d(inc, kernel_size * kernel_size, kernel_size=3, padding=1, stride=stride)
            nn.init.constant_(self.m_conv.weight, 0)
            self.m_conv.register_backward_hook(self._set_lr)

    @staticmethod
    def _set_lr(module, grad_input, grad_output):
        grad_input = (grad_input[i] * 0.1 for i in range(len(grad_input)))
        grad_output = (grad_output[i] * 0.1 for i in range(len(grad_output)))

    def forward(self, x):
        offset = self.p_conv(x)
        if self.modulation:
            m = torch.sigmoid(self.m_conv(x))

        dtype = offset.data.type()
        ks = self.kernel_size
        N = offset.size(1) // 2

        if self.padding:
            x = self.zero_padding(x)

        # (b, 2N, h, w)
        p = self._get_p(offset, dtype)

        # (b, h, w, 2N)
        p = p.contiguous().permute(0, 2, 3, 1)
        q_lt = p.detach().floor()
        q_rb = q_lt + 1

        q_lt = torch.cat([torch.clamp(q_lt[..., :N], 0, x.size(2) - 1), torch.clamp(q_lt[..., N:], 0, x.size(3) - 1)],
                         dim=-1).long()
        q_rb = torch.cat([torch.clamp(q_rb[..., :N], 0, x.size(2) - 1), torch.clamp(q_rb[..., N:], 0, x.size(3) - 1)],
                         dim=-1).long()
        q_lb = torch.cat([q_lt[..., :N], q_rb[..., N:]], dim=-1)
        q_rt = torch.cat([q_rb[..., :N], q_lt[..., N:]], dim=-1)

        # clip p
        p = torch.cat([torch.clamp(p[..., :N], 0, x.size(2) - 1), torch.clamp(p[..., N:], 0, x.size(3) - 1)], dim=-1)

        # bilinear kernel (b, h, w, N)
        g_lt = (1 + (q_lt[..., :N].type_as(p) - p[..., :N])) * (1 + (q_lt[..., N:].type_as(p) - p[..., N:]))
        g_rb = (1 - (q_rb[..., :N].type_as(p) - p[..., :N])) * (1 - (q_rb[..., N:].type_as(p) - p[..., N:]))
        g_lb = (1 + (q_lb[..., :N].type_as(p) - p[..., :N])) * (1 - (q_lb[..., N:].type_as(p) - p[..., N:]))
        g_rt = (1 - (q_rt[..., :N].type_as(p) - p[..., :N])) * (1 + (q_rt[..., N:].type_as(p) - p[..., N:]))

        # (b, c, h, w, N)
        x_q_lt = self._get_x_q(x, q_lt, N)
        x_q_rb = self._get_x_q(x, q_rb, N)
        x_q_lb = self._get_x_q(x, q_lb, N)
        x_q_rt = self._get_x_q(x, q_rt, N)

        # (b, c, h, w, N)
        x_offset = g_lt.unsqueeze(dim=1) * x_q_lt + \
                   g_rb.unsqueeze(dim=1) * x_q_rb + \
                   g_lb.unsqueeze(dim=1) * x_q_lb + \
                   g_rt.unsqueeze(dim=1) * x_q_rt

        # modulation
        if self.modulation:
            m = m.contiguous().permute(0, 2, 3, 1)
            m = m.unsqueeze(dim=1)
            m = torch.cat([m for _ in range(x_offset.size(1))], dim=1)
            x_offset *= m

        x_offset = self._reshape_x_offset(x_offset, ks)
        out = self.conv(x_offset)

        return out

    def _get_p_n(self, N, dtype):
        # 由于卷积核中心点位置是其尺寸的一半，于是中心点向左（上）方向移动尺寸的一半就得到起始点，向右（下）方向移动另一半就得到终止点
        p_n_x, p_n_y = torch.meshgrid(
            torch.arange(-(self.kernel_size - 1) // 2, (self.kernel_size - 1) // 2 + 1),
            torch.arange(-(self.kernel_size - 1) // 2, (self.kernel_size - 1) // 2 + 1))
        # (2N, 1)
        p_n = torch.cat([torch.flatten(p_n_x), torch.flatten(p_n_y)], 0)
        p_n = p_n.view(1, 2 * N, 1, 1).type(dtype)

        return p_n

    def _get_p_0(self, h, w, N, dtype):
        # p0_y、p0_x就是输出特征图每点映射到输入特征图上的纵、横坐标值。
        p_0_x, p_0_y = torch.meshgrid(
            torch.arange(1, h * self.stride + 1, self.stride),
            torch.arange(1, w * self.stride + 1, self.stride))

        p_0_x = torch.flatten(p_0_x).view(1, 1, h, w).repeat(1, N, 1, 1)
        p_0_y = torch.flatten(p_0_y).view(1, 1, h, w).repeat(1, N, 1, 1)
        p_0 = torch.cat([p_0_x, p_0_y], 1).type(dtype)

        return p_0

    # 输出特征图上每点（对应卷积核中心）加上其对应卷积核每个位置的相对（横、纵）坐标后再加上自学习的（横、纵坐标）偏移量。
    # p0就是将输出特征图每点对应到卷积核中心，然后映射到输入特征图中的位置；
    # pn则是p0对应卷积核每个位置的相对坐标；
    def _get_p(self, offset, dtype):
        N, h, w = offset.size(1) // 2, offset.size(2), offset.size(3)

        # (1, 2N, 1, 1)
        p_n = self._get_p_n(N, dtype)
        # (1, 2N, h, w)
        p_0 = self._get_p_0(h, w, N, dtype)
        p = p_0 + p_n + offset
        return p

    def _get_x_q(self, x, q, N):
        # 计算双线性插值点的4邻域点对应的权重
        b, h, w, _ = q.size()
        padded_w = x.size(3)
        c = x.size(1)
        # (b, c, h*w)
        x = x.contiguous().view(b, c, -1)

        # (b, h, w, N)
        index = q[..., :N] * padded_w + q[..., N:]  # offset_x*w + offset_y
        # (b, c, h*w*N)
        index = index.contiguous().unsqueeze(dim=1).expand(-1, c, -1, -1, -1).contiguous().view(b, c, -1)

        x_offset = x.gather(dim=-1, index=index).contiguous().view(b, c, h, w, N)

        return x_offset

    @staticmethod
    def _reshape_x_offset(x_offset, ks):
        b, c, h, w, N = x_offset.size()
        x_offset = torch.cat([x_offset[..., s:s + ks].contiguous().view(b, c, h, w * ks) for s in range(0, N, ks)],
                             dim=-1)
        x_offset = x_offset.contiguous().view(b, c, h * ks, w * ks)

        return x_offset

class FOE(nn.Module):
    def __init__(self, channel):
        super(FOE, self).__init__()
        self.dc = DeformConv2d(channel, channel)
        self.avg = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        dcx = self.dc(x)
        avg_dcx = self.avg(dcx)
        y = dcx * avg_dcx
        return dcx * y

class Resnet18SE(nn.Module):
    def __init__(self, channel=64):
        super(Resnet18SE, self).__init__()


        self.backbone = resnet18(pretrained=True)
        # self.bilinear_proj_1 = RFB_modified(256, 8192)
        # self.bilinear_proj_2 = RFB_modified(512, 8192)
        self.hbp_3_4 = hbp_layer(256, 512, 2)
        self.hbp_2_4 = hbp_layer(128, 512, 4)
        self.hbp_2_3 = hbp_layer(128, 256, 8)
        self.hbp_0_1 = hbp_layer(64, 64)
        self.hbp_0_2 = hbp_layer(64, 128)
        self.hbp_1_2 = hbp_layer(64, 128)
        self.tsf = tSF(feature_dim=512,
            num_queries=5,
            num_heads=4, FFN_method='MLP')
        self.foe = SimAM()
        # for param in self.backbone.parameters():
        #     param.requires_grad = False

        self.flatten = Flatten()

    def forward(self, x):
        raw_shape = x.shape
        x = x.view(-1, *raw_shape[-3:])
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        conv0 = x
        x = self.backbone.maxpool(x)


        x1 = self.backbone.layer1(x)

        x2 = self.backbone.layer2(x1)
        # x2 = self.foe(x2)
        x3 = self.backbone.layer3(x2)
        x4 = self.backbone.layer4(x3)

        x4= self.tsf(x4)
        # x4 = self.iden(x4)
        #
        # x4 = self.foe(x4)

        # print(x.shape)
        # print(x1.shape)
        # print(x2.shape)
        # print(x3.shape)
        # print(x4.shape)
        #
        X1 = self.hbp_3_4(x3, x4)
        X2 = self.hbp_2_4(x2, x4)
        X3 = self.hbp_2_3(x2, x3)
        # X1 = self.hbp_0_1(conv0, x1)
        # X2 = self.hbp_0_2(conv0, x2)
        # X3 = self.hbp_1_2(x1, x2)



        X_total = torch.cat((X1, X2, X3), dim=1)
        # X_total = X1 + X2 + X3

        # print(x1.shape)
        # print(x2.shape)
        # print(x3.shape)
        # print(x4.shape)


        x = self.flatten(X_total)
        # x = self.encoder(x)
        x = x.view(*raw_shape[:-3], -1)
        return x



def add_gaussian_noise(tensor, mean=0, std=1):
    noise = torch.randn(tensor.size()) * std + mean
    noisy_tensor = tensor.cpu() + noise
    return noisy_tensor.cuda()


# class VIT(nn.Module):
#     def __init__(self):
#         super(VIT, self).__init__()
#         self.model = models_vit.__dict__['vit_base_patch16'](
#             num_classes=1000,
#             drop_path_rate=0.1,
#             global_pool=True)
#
#         checkpoint = torch.load('./mae_finetuned_vit_base.pth', map_location='cpu')
#         checkpoint_model = checkpoint['model']
#         state_dict = self.model.state_dict()
#         self.model.load_state_dict(checkpoint_model, strict=False)
#
#
#         self.flatten = Flatten()
#
#     def forward(self, x):
#         raw_shape = x.shape
#         x = x.view(-1, *raw_shape[-3:])
#         out = self.model(x)
#
#         x = self.flatten(X_total)
#         # x = self.encoder(x)
#         x = x.view(*raw_shape[:-3], -1)
#         return x

class MobileNet(nn.Module):
    def __init__(self):
        super(MobileNet, self).__init__()
        self.backbone = torchvision.models.mobilenet_v2(pretrained=True)
        # self.backbone = torchvision2.models.mobilenet_v2()
        self.features = self.backbone.features
        self.features_conv6 = torch.nn.Sequential(*list(self.features.children()))[:-4]
        self.features_conv5 = torch.nn.Sequential(*list(self.features.children()))[-4:-3]
        self.features_conv4 = torch.nn.Sequential(*list(self.features.children()))[-3:-2]
        self.features_conv3 = torch.nn.Sequential(*list(self.features.children()))[-2:-1]
        self.features_conv2 = torch.nn.Sequential(*list(self.features.children()))[-1]
        # print(self.features_conv6)
        # print(self.features_conv4)
        # print(self.features_conv3)
        self.ta = TripletAttention()
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.max_gap = nn.AdaptiveMaxPool2d(1)
        self.conv = nn.Conv2d(640, 1, 1)
        self.sigmoid = nn.Sigmoid()
        self.ASFF1 = ASFF(320, 320)
        self.ASFF2 = ASFF(320, 320)
        self.ASFF3 = ASFF(320, 320)
        self.conv1 = nn.Conv2d(160, 320, 1)
        self.conv2 = nn.Conv2d(160, 320, 1)
        # print( self.features_conv3 )

        self.flatten = Flatten()

    def forward(self, x):
        raw_shape = x.shape
        x = x.view(-1, *raw_shape[-3:])
        x = self.features_conv6(x)
        # x = self.ta(x)
        x1 = self.features_conv5(x)
        x2 = self.features_conv4(x1)
        x3 = self.features_conv3(x2)
        x1 = self.conv1(x1)
        x2 = self.conv2(x2)
        X_branch_1 = self.ASFF1(x1, x2)
        X_branch_2 = self.ASFF2(x1, x3)
        X_branch_3 = self.ASFF3(x2, x3)
        x = X_branch_1 + X_branch_2 + X_branch_3
        B, C, H, W = x.shape

        gap_branch = self.gap(x).expand(B, C, H, W)  # b c h w
        result = torch.cat((x, gap_branch), dim=1)  # B 2C H W
        # result = X_branch * gap_branch
        result = self.conv(result)  # B 1 H W
        att = self.sigmoid(result)  # b 1 h w
        out = (att * x).view(B, C, -1)  # B C HW
        # x = torch.sum(out, dim=2)

        x = self.flatten(out)
        # x = self.encoder(x)
        x = x.view(*raw_shape[:-3], -1)
        return x

class GhostModule(nn.Module):
    def __init__(self, inp, oup, kernel_size=1, ratio=2, dw_size=3, stride=1, relu=True):
        super(GhostModule, self).__init__()
        self.oup = oup
        init_channels = math.ceil(oup / ratio)
        new_channels = init_channels*(ratio-1)

        self.primary_conv = nn.Sequential(
            nn.Conv2d(inp, init_channels, kernel_size, stride, kernel_size//2, bias=False),
            nn.BatchNorm2d(init_channels),
            nn.ReLU(inplace=True) if relu else nn.Sequential(),
        )

        self.cheap_operation = nn.Sequential(
            nn.Conv2d(init_channels, new_channels, dw_size, 1, dw_size//2, groups=init_channels, bias=False),
            nn.BatchNorm2d(new_channels),
            nn.ReLU(inplace=True) if relu else nn.Sequential(),
        )

    def forward(self, x):
        x1 = self.primary_conv(x)
        x2 = self.cheap_operation(x1)
        out = torch.cat([x1,x2], dim=1)
        return out[:,:self.oup,:,:]

class Shuffle_asff(nn.Module):
    def __init__(self, c1=512, c2=512):
        super(Shuffle_asff, self).__init__()
        channel = 32
        self.weight_v_1 = BasicConv2d(976, channel, 1)
        self.weight_v_2 = BasicConv2d(976, channel, 1)
        self.weight_v_3 = BasicConv2d(976, channel, 1)
        # self.weight_v_1 = GhostModule(976, channel)
        # self.weight_v_2 = GhostModule(976, channel)
        # self.weight_v_3 = GhostModule(976, channel)
        self.weight_levels = nn.Conv2d(channel*3, 3, 1, 1, 0)
        # self.expand = BasicConv2d(512, 256, 3, 1, 1)
        self.conv2_4 = BasicConv2d(244, 976, 1)
        self.conv3_4 = BasicConv2d(488, 976, 1)
        # self.c = nn.Conv2d(976*3, 976, 1)
        self.upsample = cus_sample

    def forward(self, x2, x3, x4):
        # x2 = self.conv2_4(x2)
        # x2 = self.upsample(x2, size=(x4.shape[2], x4.shape[3]))
        # x3 = self.conv3_4(x3)
        # x3 = self.upsample(x3, size=(x4.shape[2], x4.shape[3]))
        # x = torch.cat((x2, x3, x4), dim=1)
        # return self.c(x)


        # x2 = self.conv2_4(x2)
        # x2 = self.upsample(x2, size=(x4.shape[2], x4.shape[3]))
        # x3 = self.conv3_4(x3)
        # x3 = self.upsample(x3, size=(x4.shape[2], x4.shape[3]))
        #
        #
        # return x2 + x3 + x4


        # CSAFF
        x2 = self.conv2_4(x2)
        x2 = self.upsample(x2, size=(x4.shape[2], x4.shape[3]))
        x3 = self.conv3_4(x3)
        x3 = self.upsample(x3, size=(x4.shape[2], x4.shape[3]))
        # return x1+x2
        x1_weight_v = self.weight_v_1(x2)
        x2_weight_v = self.weight_v_2(x3)
        x3_weight_v = self.weight_v_3(x4)
        weight_v = torch.cat((x1_weight_v, x2_weight_v, x3_weight_v), 1)
        levels_weight = self.weight_levels(weight_v)
        levels_weight = F.softmax(levels_weight, dim=1)
        fused_out = x2 * levels_weight[:,0:1,:,:] + x3*levels_weight[:,1:2,:,:] + x4*levels_weight[:,2:3,:,:]
        # # out = self.expand(fused_out)
        return fused_out

class InstanceAtt(nn.Module):
    def __init__(self, c):
        super(InstanceAtt, self).__init__()
        self.hidden_c = 96
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.maxgap = nn.AdaptiveMaxPool2d(1)
        self.mlp = nn.Sequential(
            nn.Linear(c, self.hidden_c),
            nn.ReLU(),
            nn.Linear(self.hidden_c, c),
        )
        k = int(abs((math.log(c, 2) + 1) / 2))
        kernel_size = k if k % 2 else k + 1
        padding = kernel_size // 2
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=1, kernel_size=kernel_size, padding=padding, bias=False),
            # nn.Conv1d(in_channels=1, out_channels=1, kernel_size=kernel_size, padding=padding, bias=False),
            nn.Sigmoid()
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        gap_x = self.gap(x).view(x.shape[0], 1, -1)
        max_x = self.maxgap(x).view(x.shape[0], 1, -1)
        gap_x = self.conv(gap_x).view(x.shape[0], x.shape[1], 1, 1)
        max_x = self.conv(max_x).view(x.shape[0], x.shape[1], 1, 1)
        # gap_x = self.gap(x).view(x.shape[0], -1)
        # max_x = self.maxgap(x).view(x.shape[0], -1)
        # gap_x = self.mlp(gap_x).view(x.shape[0], x.shape[1], 1, 1).sigmoid()
        # max_x = self.mlp(max_x).view(x.shape[0], x.shape[1], 1, 1).sigmoid()
        # att = self.sigmoid(gap_x + max_x)
        return gap_x *x + max_x*x

# 全集上的PVT-tsf
class PVT_tSF(nn.Module):
    def __init__(self):
        super(PVT_tSF, self).__init__()
        # self.backbone = torchvision.models.mobilenet_v2(pretrained=True)
        self.backbone = pvt_v2_b2()
        path = 'pvt_v2_b2.pth'
        save_model = torch.load(path)
        model_dict = self.backbone.state_dict()
        state_dict = {k: v for k, v in save_model.items() if k in model_dict.keys()}
        model_dict.update(state_dict)
        self.backbone.load_state_dict(model_dict)
        self.hbp_3_4 = hbp_layer(320, 512, 2)
        self.hbp_2_4 = hbp_layer(128, 512, 4)
        self.hbp_2_3 = hbp_layer(128, 320, 8)
        self.tsf = tSF(feature_dim=512,
                           num_queries=5,
                           num_heads=4, FFN_method='MLP')
        self.avg = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(512, 38)


    def forward(self, x):
        raw_shape = x.shape
        x = x.view(-1, *raw_shape[-3:])

        B = x.shape[0]
        outs = []

        # stage 1
        x1, H, W = self.backbone.patch_embed1(x)
        for i, blk in enumerate(self.backbone.block1):
            x1 = blk(x1, H, W)
        x1 = self.backbone.norm1(x1)
        x1 = x1.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(x1)

        # stage 2
        x2, H, W = self.backbone.patch_embed2(x1)
        for i, blk in enumerate(self.backbone.block2):
            x2 = blk(x2, H, W)
        x2 = self.backbone.norm2(x2)
        x2 = x2.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(x2)


        # stage 3
        x3, H, W = self.backbone.patch_embed3(x2)
        for i, blk in enumerate(self.backbone.block3):
            x3 = blk(x3, H, W)
        x3 = self.backbone.norm3(x3)
        x3 = x3.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(x3)

        # print(x3.shape)

        # x3 = self.ASFF2(x2, x3)

        # stage 4
        x4, H, W = self.backbone.patch_embed4(x3)
        for i, blk in enumerate(self.backbone.block4):
            x4 = blk(x4, H, W)
        x4 = self.backbone.norm4(x4)
        x4 = x4.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(x4)

        x4 = self.tsf(x4)


        # x_2_3 = self.hbp_2_3(x2, x3)
        # x_2_4 = self.hbp_2_4(x2, x4)
        # x_3_4 = self.hbp_3_4(x3, x4)
        #
        #
        # out = torch.cat((x_2_3, x_2_4, x_3_4), dim=1)
        # print(out.shape)
        #
        #
        x = self.avg(x4)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class EfficientNet(nn.Module):
    def __init__(self):
        super(EfficientNet, self).__init__()
        # self.backbone = torchvision.models.mobilenet_v2(pretrained=True)
        # self.backbone = efficientnet_b0(pretrained=True)
        # self.backbone = torchvision.models.shufflenet_v2(pretrained=True)
        # self.backbone = shufflenet_v2_x1_0()
        self.backbone = shufflenet_v2_x2_0()
        net_dict = self.backbone.state_dict()
        predict_model = torch.load('shufflenetv2_x2_0-8be3c8ee.pth')
        state_dict = {k: v for k, v in predict_model.items() if k in net_dict.keys()}  # 寻找网络中公共层，并保留预训练参数
        net_dict.update(state_dict)  # 将预训练参数更新到新的网络层
        self.backbone.load_state_dict(net_dict)
        ### here
        self.shuffle_asff = Shuffle_asff()
        # self.cbam = CBAM(976)
        # self.se = SE_Block(976)
        # self.eca = ECA(976)
        # self.bam = BAM(976)
        self.instance_att = InstanceAtt(976)
        # # # # # self.coodinate_att = CoordAtt(976, 976)
        self.gap = nn.AdaptiveAvgPool2d(1)
        # self.maxgap = nn.AdaptiveMaxPool2d(1)
        self.sigmoid = nn.Sigmoid()
        # self.conv = BasicConv2d(1952, 976, 1)
        self.conv = nn.Conv2d(1952, 976, 1, groups=976)
        ####



        # self.backbone = torch.hub.load('huawei-noah/ghostnet', 'ghostnet_1x', pretrained=True)

        # torch.hub._validate_not_a_forked_repo = lambda a, b, c: True
        # self.backbone = torch.hub.load('pytorch/vision:v0.10.0', 'shufflenet_v2_x1_0', pretrained=True)
        # print(self.backbone)
        # self.backbone = efficientnet_b7(pretrained=True)

        # print(self.backbone)
        # self.features = self.backbone.features
        # # self.features_conv5 = torch.nn.Sequential(*list(self.features.children()))[:-7]
        # self.features_conv5 = torch.nn.Sequential(*list(self.features.children()))[:-3]
        # self.features_conv4 = torch.nn.Sequential(*list(self.features.children()))[-3:-1]
        # # print( self.features_conv5 )

        self.flatten = Flatten()

    def forward(self, x):
        raw_shape = x.shape
        x = x.view(-1, *raw_shape[-3:])
        # x = self.backbone.conv_first(x)
        # for i in range(len(self.backbone.inverted_residual_setting)):
        #     x = getattr(self.backbone, 'layer%d' % (i + 1))(x)
        # x = self.backbone.conv_last(x)

        # x = self.backbone(x)
        x = self.backbone.conv1(x)
        x = self.backbone.maxpool(x)
        x2 = self.backbone.stage2(x)
        x3 = self.backbone.stage3(x2)
        x4 = self.backbone.stage4(x3)

        # x = x4

        # x = self.cbam(x)
        # x = self.eca(x)
        # x = self.se(x)
        # x = self.bam(x)
        x = self.shuffle_asff(x2, x3, x4)
        x = self.instance_att(x)
        # # # # x = self.coodinate_att(x)
        B, C, H, W = x.shape
        #
        #
        #
        # #
        gap_branch = self.gap(x).expand(B, C, H, W)  # b c h w
        result = torch.cat((x, gap_branch), dim=1)  # B 2C H W
        # result = X_branch * gap_branch
        result = self.conv(result)  # B 1 H W
        att = self.sigmoid(result)  # b 1 h w
        x = (att * x)  # B C HW



        # print(x2.shape)
        # print(x3.shape)
        # print(x4.shape)
        # x = self.backbone.conv5(x)


        # x = self.ta(x)
        # x = self.features_conv4(x)

        x = self.flatten(x)
        # x = self.encoder(x)
        x = x.view(*raw_shape[:-3], -1)
        return x

class VIT(nn.Module):
    def __init__(self):
        super(VIT, self).__init__()
        # self.backbone = torchvision.models.mobilenet_v2(pretrained=True)
        self.backbone = VisionTransformer(CONFIGS['ViT-B_16'], 96)
        self.backbone.load_from(np.load('ViT-B_16.npz'))
        print(self.backbone.transformer)
        # self.backbone = efficientnet_b7(pretrained=True)

        # print(self.backbone)
        # self.features = self.backbone.features
        # # self.features_conv5 = torch.nn.Sequential(*list(self.features.children()))[:-7]
        # self.features_conv5 = torch.nn.Sequential(*list(self.features.children()))[:-3]
        # self.features_conv4 = torch.nn.Sequential(*list(self.features.children()))[-3:-1]
        # self.ta = TripletAttention()
        # # print( self.features_conv5 )

        self.flatten = Flatten()

    def forward(self, x):
        raw_shape = x.shape
        x = x.view(-1, *raw_shape[-3:])
        x = self.backbone.transformer(x)
        print(x.shape)

        x = self.flatten(x)
        # x = self.encoder(x)
        x = x.view(*raw_shape[:-3], -1)
        return x

class PVTASFF(nn.Module):
    def __init__(self, c1, c2):
        super(PVTASFF, self).__init__()
        self.conv = nn.Conv2d(c1, c2, 1)

        channel = 16
        self.weight_v_1 = BasicConv2d(c2, channel, 1)
        self.weight_v_2 = BasicConv2d(c2, channel, 1)
        self.weight_levels = nn.Conv2d(channel * 2, 2, 1, 1, 0)
        self.expand = BasicConv2d(512, 256, 3, 1, 1)
        self.upsample = cus_sample


    def forward(self, x1, x2):
        x1 = self.conv(x1)
        x1 = self.upsample(x1, size=(x2.shape[2], x2.shape[3]))
        # return x1+x2
        x1_weight_v = self.weight_v_1(x1)
        x2_weight_v = self.weight_v_2(x2)
        weight_v = torch.cat((x1_weight_v, x2_weight_v), 1)
        levels_weight = self.weight_levels(weight_v)
        levels_weight = F.softmax(levels_weight, dim=1)
        fused_out = x1 * levels_weight[:, 0:1, :, :] + x2 * levels_weight[:, 1:2, :, :]
        # out = self.expand(fused_out)
        return fused_out


class FF(nn.Module):
    def __init__(self):
        super(FF, self).__init__()
        # self.conv = nn.Conv2d(c1, c2, 1)
        #
        c2 = 512
        self.conv2_4 = BasicConv2d(128, 512, 1)
        self.conv3_4 = BasicConv2d(320, 512, 1)
        channel = 16
        self.weight_v_1 = BasicConv2d(c2, channel, 1)
        self.weight_v_2 = BasicConv2d(c2, channel, 1)
        self.weight_v_3 = BasicConv2d(c2, channel, 1)
        self.weight_levels = nn.Conv2d(channel * 3, 3, 1, 1, 0)
        self.upsample = cus_sample


    def forward(self, x2, x3, x4):
        x2 = self.conv2_4(x2)
        x2 = self.upsample(x2, size=(x4.shape[2], x4.shape[3]))
        x3 = self.conv3_4(x3)
        x3 = self.upsample(x3, size=(x4.shape[2], x4.shape[3]))

        x1_weight_v = self.weight_v_1(x2)
        x2_weight_v = self.weight_v_2(x3)
        x3_weight_v = self.weight_v_3(x4)
        weight_v = torch.cat((x1_weight_v, x2_weight_v, x3_weight_v), 1)
        levels_weight = self.weight_levels(weight_v)
        levels_weight = F.softmax(levels_weight, dim=1)
        fused_out = x2 * levels_weight[:,0:1,:,:] + x3*levels_weight[:,1:2,:,:] + x4*levels_weight[:,2:3,:,:]
        # out = self.expand(fused_out)
        return fused_out


class PVT_FF(nn.Module):
    def __init__(self):
        super(PVT_FF, self).__init__()
        self.backbone = pvt_v2_b2()
        path = 'pvt_v2_b2.pth'
        save_model = torch.load(path)
        model_dict = self.backbone.state_dict()
        state_dict = {k: v for k, v in save_model.items() if k in model_dict.keys()}
        model_dict.update(state_dict)
        self.backbone.load_state_dict(model_dict)
        self.FF = FF()
        self.flatten = Flatten()

    def forward(self, x):
        raw_shape = x.shape
        x = x.view(-1, *raw_shape[-3:])

        B = x.shape[0]
        outs = []

        # stage 1
        x1, H, W = self.backbone.patch_embed1(x)
        for i, blk in enumerate(self.backbone.block1):
            x1 = blk(x1, H, W)
        x1 = self.backbone.norm1(x1)
        x1 = x1.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(x1)

        # stage 2
        x2, H, W = self.backbone.patch_embed2(x1)
        for i, blk in enumerate(self.backbone.block2):
            x2 = blk(x2, H, W)
        x2 = self.backbone.norm2(x2)
        x2 = x2.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(x2)

        # stage 3
        x3, H, W = self.backbone.patch_embed3(x2)
        for i, blk in enumerate(self.backbone.block3):
            x3 = blk(x3, H, W)
        x3 = self.backbone.norm3(x3)
        x3 = x3.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(x3)

        # print(x3.shape)

        # x3 = self.ASFF2(x2, x3)

        # stage 4
        x4, H, W = self.backbone.patch_embed4(x3)
        for i, blk in enumerate(self.backbone.block4):
            x4 = blk(x4, H, W)
        x4 = self.backbone.norm4(x4)
        x4 = x4.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(x4)

        #
        y = self.FF(x2, x3, x4)


        x = self.flatten(y)
        # x = self.encoder(x)
        x = x.view(*raw_shape[:-3], -1)
        return x

class PVT(nn.Module):
    def __init__(self):
        super(PVT, self).__init__()
        self.backbone = pvt_v2_b2()
        path = 'pvt_v2_b2.pth'
        save_model = torch.load(path)
        model_dict = self.backbone.state_dict()
        state_dict = {k: v for k, v in save_model.items() if k in model_dict.keys()}
        model_dict.update(state_dict)
        self.backbone.load_state_dict(model_dict)
        self.hbp_3_4 = hbp_layer(320, 512, 2)
        self.hbp_2_4 = hbp_layer(128, 512, 4)
        self.hbp_2_3 = hbp_layer(128, 320, 8)

        self.tsf = tSF(feature_dim=512,
                       num_queries=5,
                       num_heads=4, FFN_method='MLP')


        self.flatten = Flatten()

    def forward(self, x):
        raw_shape = x.shape
        x = x.view(-1, *raw_shape[-3:])

        B = x.shape[0]
        outs = []

        # stage 1
        x1, H, W = self.backbone.patch_embed1(x)
        for i, blk in enumerate(self.backbone.block1):
            x1 = blk(x1, H, W)
        x1 = self.backbone.norm1(x1)
        x1 = x1.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(x1)

        # stage 2
        x2, H, W = self.backbone.patch_embed2(x1)
        for i, blk in enumerate(self.backbone.block2):
            x2 = blk(x2, H, W)
        x2 = self.backbone.norm2(x2)
        x2 = x2.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(x2)

        # x2 = self.ASFF1(x1, x2)
        # print(x1.shape)
        # print(x2.shape)


        # stage 3
        x3, H, W = self.backbone.patch_embed3(x2)

        for i, blk in enumerate(self.backbone.block3):
            x3 = blk(x3, H, W)
        x3 = self.backbone.norm3(x3)
        x3 = x3.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(x3)

        # print(x3.shape)

        # x3 = self.ASFF2(x2, x3)

        # stage 4
        x4, H, W = self.backbone.patch_embed4(x3)

        for i, blk in enumerate(self.backbone.block4):
            x4 = blk(x4, H, W)
        x4 = self.backbone.norm4(x4)
        x4 = x4.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(x4)




        # x4 = self.ASFF3(x3, x4)





        # x1, x2, x3, x4 = self.backbone(x)
        #

        x4 = self.tsf(x4)
        #
        x_2_3 = self.hbp_2_3(x2, x3)
        x_2_4 = self.hbp_2_4(x2, x4)
        x_3_4 = self.hbp_3_4(x3, x4)
        #
        # # print(x1.shape)
        # # print(x2.shape)
        # # print(x3.shape)
        # # print(x4.shape)
        #
        out = torch.cat((x_2_3, x_2_4, x_3_4), dim=1)
        ###分割线
        # B, C, H, W = x.shape
        #
        # gap_branch = self.gap(x).expand(B, C, H, W)  # b c h w
        # result = torch.cat((x, gap_branch), dim=1)  # B 2C H W
        # # result = X_branch * gap_branch
        # result = self.conv(result)  # B 1 H W
        # att = self.sigmoid(result)  # b 1 h w
        # out = (att * x)
        # # out = (att * X_branch).view(B, C, -1) # B C HW
        # # X_branch = torch.sum(out, dim=2)

        x = self.flatten(out)
        # x = self.encoder(x)
        x = x.view(*raw_shape[:-3], -1)
        return x

class MAE(nn.Module):
    def __init__(self):
        super(MAE, self).__init__()
        # self.backbone = torchvision.models.mobilenet_v2(pretrained=True)
        self.backbone = vit_base_patch16()
        checkpoint = torch.load('mae_pretrain_vit_base.pth', map_location='cpu')
        checkpoint_model = checkpoint['model']
        print("Load pre-trained checkpoint from: %s" % 'mae_pretrain_vit_base')
        state_dict = self.backbone.state_dict()
        for k in ['head.weight', 'head.bias']:
            if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
                print(f"Removing key {k} from pretrained checkpoint")
                del checkpoint_model[k]

        # # interpolate position embedding
        self.interpolate_pos_embed(self.backbone, checkpoint_model)

        # load pre-trained model
        msg = self.backbone.load_state_dict(checkpoint_model, strict=False)
        print(msg)

        # if args.global_pool:
        #     assert set(msg.missing_keys) == {'head.weight', 'head.bias', 'fc_norm.weight', 'fc_norm.bias'}
        # else:
        #     assert set(msg.missing_keys) == {'head.weight', 'head.bias'}

        # manually initialize fc layer
        trunc_normal_(self.backbone.head.weight, std=2e-5)

        # self.backbone = efficientnet_b7(pretrained=True)

        # print(self.backbone)
        # self.features = self.backbone.features
        # # self.features_conv5 = torch.nn.Sequential(*list(self.features.children()))[:-7]
        # self.features_conv5 = torch.nn.Sequential(*list(self.features.children()))[:-3]
        # self.features_conv4 = torch.nn.Sequential(*list(self.features.children()))[-3:-1]
        # self.ta = TripletAttention()
        # # print( self.features_conv5 )

        self.flatten = Flatten()

    def interpolate_pos_embed(model, checkpoint_model):
        if 'pos_embed' in checkpoint_model:
            pos_embed_checkpoint = checkpoint_model['pos_embed']
            embedding_size = pos_embed_checkpoint.shape[-1]
            num_patches = model.patch_embed.num_patches
            num_extra_tokens = model.pos_embed.shape[-2] - num_patches
            # height (== width) for the checkpoint position embedding
            orig_size = int((pos_embed_checkpoint.shape[-2] - num_extra_tokens) ** 0.5)
            # height (== width) for the new position embedding
            new_size = int(num_patches ** 0.5)
            # class_token and dist_token are kept unchanged
            if orig_size != new_size:
                print("Position interpolate from %dx%d to %dx%d" % (orig_size, orig_size, new_size, new_size))
                extra_tokens = pos_embed_checkpoint[:, :num_extra_tokens]
                # only the position tokens are interpolated
                pos_tokens = pos_embed_checkpoint[:, num_extra_tokens:]
                pos_tokens = pos_tokens.reshape(-1, orig_size, orig_size, embedding_size).permute(0, 3, 1, 2)
                pos_tokens = torch.nn.functional.interpolate(
                    pos_tokens, size=(new_size, new_size), mode='bicubic', align_corners=False)
                pos_tokens = pos_tokens.permute(0, 2, 3, 1).flatten(1, 2)
                new_pos_embed = torch.cat((extra_tokens, pos_tokens), dim=1)
                checkpoint_model['pos_embed'] = new_pos_embed

    def forward(self, x):
        raw_shape = x.shape
        x = x.view(-1, *raw_shape[-3:])
        x = self.backbone(x)
        # print(x.shape)

        x = self.flatten(x)
        # x = self.encoder(x)
        x = x.view(*raw_shape[:-3], -1)
        return x

def get_similarity(proto, query):
    proto = proto.permute(0, 2, 1)
    query = query.permute(0, 2, 1)
    way = proto.shape[0]
    num_query = query.shape[0]
    proto = proto.unsqueeze(0).repeat([num_query, 1, 1, 1])
    query = query.unsqueeze(1).repeat([1, way, 1, 1])
    # proto.shape, query.shape
    proto = proto.permute(0, 1, 3, 2)
    query = query.permute(0, 1, 3, 2)
    feature_size = proto.shape[-2]
    proto = proto.unsqueeze(-3)
    query = query.unsqueeze(-2)
    query = query.repeat(1, 1, 1, feature_size, 1)
    similarity_map = F.cosine_similarity(proto, query, dim=-1)
    return similarity_map

class PrototypicalNetwork(nn.Module):
    def __init__(self, input_img_channels, hidden_channels):
        super(PrototypicalNetwork, self).__init__()
        # self.encoder = ConvEncoder(input_img_channels, hidden_channels)
        # self.encoder = ConvEncoderC2F(input_img_channels, hidden_channels)
        # self.encoder = HBP(pretrained=True)
        # self.encoder = Resnet18SE(64)
        # self.encoder = HBP()
        self.encoder = EfficientNet()
        # self.encoder = VIT()
        # self.encoder = PVT()
        # self.encoder = PVT_FF()
        # self.encode = VIT()
        # self.encoder = resnet50(pretrained=True)
        # self.encoder = PvtC2F()
        # self.encoder = denseNet()
        # self.encoder = Resnet18()
        # self.encoder = Resnet18W2()
        # self.encoder = HBP()
        # self.encoder = InceptionV3()
        # self.encoder = MAE()
        # self.encoder = MobileNet()
        # self.encoder = Resnet18C2F(64)
        # net_dict = self.encoder.state_dict()
        # predict_model = torch.load('ConvC2F_best.pth.tar')
        # state_dict = {k: v for k, v in predict_model.items() if k in net_dict.keys()}  # 寻找网络中公共层，并保留预训练参数
        # net_dict.update(state_dict)  # 将预训练参数更新到新的网络层
        # self.encoder.load_state_dict(net_dict)
        # print('成功导入')
        # self.SFSA = SFSA(512)

    def test_mode(self, x_support, x_query):
        x_proto = self.encoder(x_support)  # (n, k, embed_dim)
        x_proto = x_proto.mean(1)  # (n, embed_dim)
        x_q = self.encoder(x_query)  # (n, q, embed_dim)
        x_q = x_q.view(-1, x_q.shape[-1])  # (n*q, embed_dim)

        sim_result = self.similarity(x_q, x_proto)  # (n*q, n)

        log_p_y = F.log_softmax(sim_result, dim=1)
        # print('log_p_y:', log_p_y)

        return log_p_y  # (n*q, n)

    def forward(self, x_support, x_query):
        """
        infer an n-way k-shot task
        :param x_support: (n, k, c, w, h)
        :param x_query: (n, q, c, w, h) or (q, c, w, h)
        :return: (q, n)
        """


        n = x_support.shape[0]
        k = x_support.shape[1]
        q = x_query.shape[1]
        # print('x_support', x_support.shape)
        # print('x_query', x_query.shape)
        # # 添加高斯噪声
        # noisy_support = []
        # for i in range(x_support.shape[0]):
        #     tmp = []
        #     for j in range(x_support.shape[1]):
        #         img = x_support[i][j]
        #         tmp.append(add_gaussian_noise(img, 0, 0.5))
        #     tmp = torch.stack([img for img in tmp], dim=0)
        #     noisy_support.append(tmp)
        # noisy_support = torch.stack([img for img in noisy_support], dim=0)
        #
        # noisy_query = []
        # for i in range(x_query.shape[0]):
        #     tmp = []
        #     for j in range(x_query.shape[1]):
        #         img = x_query[i][j]
        #         tmp.append(add_gaussian_noise(img, 0, 0.5))
        #     tmp = torch.stack([img for img in tmp], dim=0)
        #     noisy_query.append(tmp)
        # noisy_query = torch.stack([img for img in noisy_query], dim=0)


        x_proto = self.encoder(x_support)  # (n, k, embed_dim)
        x_q = self.encoder(x_query)  # (n, q, embed_dim)
        # x_proto, x_q = SFSA(x_proto, x_q)
        # x_proto = Flatten(x_proto).view(n, k, -1)
        # x_q = Flatten(x_q).view(n, q, -1)

        x_proto_mean = x_proto.mean(1)  # (n, embed_dim)

        x_proto_mean_relation = x_proto_mean.unsqueeze(1).repeat(1, q, 1).view(n*q, -1)


        x_q_2= x_q.view(-1, x_q.shape[-1])  # (n*q, embed_dim)


        # 反向
        x_q_mean = x_q.mean(1)
        x_p = x_proto.view(-1, x_proto.shape[-1])

        # # 这里是噪声图片的计算
        # x_proto_noisy = self.encoder(noisy_support)  # (n, k, embed_dim)
        # x_proto_noisy = x_proto_noisy.mean(1)  # (n, embed_dim)
        # x_q_noisy = self.encoder(noisy_query)  # (n, q, embed_dim)
        # x_q_noisy = x_q_noisy.view(-1, x_q_noisy.shape[-1])  # (n*q, embed_dim)

        # 这里单独计算全局LOSS
        # x_q_global = self.classifier(x_q_2)


        sim_result = self.similarity(x_q_2, x_proto_mean)  # (n*q, n)
        # sim_result_cos = self.similarity(x_q_2, x_proto_mean, 'kl')

        sim_result_back = self.similarity(x_p, x_q_mean)


        # sim_result_noisy = self.similarity(x_q_noisy, x_proto_noisy)

        log_p_y = F.log_softmax(sim_result, dim=1)
        # log_cos = F.log_softmax(sim_result_cos, dim=1)
        # log_p_y = log_p_y + log_cos
        log_p_y_back = F.log_softmax(sim_result_back, dim=1)


        # lop_p_y_cos = F.log_softmax(sim_result_cos, dim=1)
        # print('log_p_y:', log_p_y)

        # return log_p_y, log_p_y_back, lop_p_y_cos  # (n*q, n)
        return log_p_y, log_p_y_back
        # return log_p_y


    @staticmethod
    def similarity(a, b, sim_type='euclidean'):
        methods = {'euclidean': euclidean_dist_similarity, 'cosine': cosine_similarity, 'kl': KL_dist}
        assert sim_type in methods.keys(), f'type must be in {methods.keys()}'
        return methods[sim_type](a, b)  # 值越大相似度越高



class PCA(object):
    def __init__(self, n_components=2):
        self.n_components = n_components

    def fit(self, X):
        n = X.shape[0]
        self.mean = torch.mean(X, axis=0)
        X = X - self.mean
        covariance_matrix = 1 / n * torch.matmul(X.T, X)
        eigenvalues, eigenvectors = torch.eig(covariance_matrix, eigenvectors=True)
        eigenvalues = torch.norm(eigenvalues, dim=1)
        idx = torch.argsort(-eigenvalues)
        eigenvectors = eigenvectors[:, idx]
        self.proj_mat = eigenvectors[:, 0:self.n_components]

    def transform(self, X):
        X = X - self.mean
        return X.matmul(self.proj_mat)

# class PrototypicalNetwork(nn.Module):
#     def __init__(self, input_img_channels, hidden_channels):
#         super(PrototypicalNetwork, self).__init__()
#         self.encoder = Resnet18SE(64)
#         #
#         # checkpoint = torch.load('./runs/exp169/checkpoint/proto_best.pth')
#         #
#         # model_dict = self.encoder.state_dict()
#         # model_list1 = list(checkpoint.keys())
#         # model_list2 = list(model_dict.keys())
#         # len1 = len(model_list1)
#         # len2 = len(model_list2)
#         # m, n = 0, 0
#         # while True:
#         #     if m >= len1 or n >= len2:
#         #         break
#         #     layername1, layername2 = model_list1[m], model_list2[n]
#         #     w1, w2 = checkpoint[layername1], model_dict[layername2]
#         #     if w1.shape != w2.shape:
#         #         continue
#         #     model_dict[layername2] = checkpoint[layername1]
#         #     m += 1
#         #     n += 1
#         # self.encoder.load_state_dict(model_dict)
#         #
#         self.encoder2 = HBP()
#         # self.encoder = VGG19()
#         # self.encoder2 = VGG19()
#         self.pca = PCA(n_components=512)
#         # net_dict = self.encoder.state_dict()
#         # predict_model = torch.load('ConvC2F_best.pth.tar')
#         # state_dict = {k: v for k, v in predict_model.items() if k in net_dict.keys()}  # 寻找网络中公共层，并保留预训练参数
#         # net_dict.update(state_dict)  # 将预训练参数更新到新的网络层
#         # self.encoder.load_state_dict(net_dict)
#         # print('成功导入')
#         # self.SFSA = SFSA(512)
#
#     def soft_voting(self, predicted_probas) -> np.array:
#         sv_predicted_proba = torch.mean(predicted_probas, dim=0)
#         sv_predicted_proba[:, -1] = 1 - torch.sum(sv_predicted_proba[:, :-1], axis=1)
#
#         return sv_predicted_proba, sv_predicted_proba.argmax(axis=1)
#
#     def PCA_svd(self, X, k, center=True):
#         n = X.size()[0]
#         ones = torch.ones(n).view([n, 1])
#         h = ((1 / n) * torch.mm(ones, ones.t())) if center else torch.zeros(n * n).view([n, n])
#         H = torch.eye(n) - h
#         H = H.cuda()
#         X_center = torch.mm(H.double(), X.double())
#         u, s, v = torch.svd(X_center)
#         components = v[:k].t()
#         # explained_variance = torch.mul(s[:k], s[:k])/(n-1)
#         return components
#
#     def test_mode(self, x_support, x_query):
#         x_proto = self.encoder(x_support)  # (n, k, embed_dim)
#         x_proto2 = self.encoder2(x_support)
#         x_proto = x_proto.mean(1)  # (n, embed_dim)
#
#         x_proto2 = x_proto2.mean(1)
#
#         x_q = self.encoder(x_query)  # (n, q, embed_dim)
#         x_q = x_q.view(-1, x_q.shape[-1])  # (n*q, embed_dim)
#         x_q2 = self.encoder2(x_query)
#         x_q2 = x_q2.view(-1, x_q2.shape[-1])
#
#         # x_q = x_q + x_q2
#         # x_proto_mean = x_proto + x_proto2
#         sim_result = self.similarity(x_q, x_proto)  # (n*q, n)
#         sim_result2 = self.similarity(x_q2, x_proto2)
#         sim_result = (sim_result + sim_result2)/2
#         # list = [None] * 2
#         # list[0] = sim_result
#         # list[1] = sim_result2
#         # # list[2] = sim_result_total
#         #
#         #
#         # sv_predicted_proba, _ = self.soft_voting(torch.stack(list, dim=0))
#
#         log_p_y = F.log_softmax(sim_result, dim=1)
#         # print('log_p_y:', log_p_y)
#
#         return log_p_y  # (n*q, n)
#
#     def forward(self, x_support, x_query):
#         """
#         infer an n-way k-shot task
#         :param x_support: (n, k, c, w, h)
#         :param x_query: (n, q, c, w, h) or (q, c, w, h)
#         :return: (q, n)
#         """
#         n = x_support.shape[0]
#         k = x_support.shape[1]
#         q = x_query.shape[1]
#
#         x_proto = self.encoder(x_support)  # (n, k, embed_dim)
#         x_proto2 = self.encoder2(x_support)
#
#
#
#         x_q = self.encoder(x_query)  # (n, q, embed_dim)
#         x_q2 = self.encoder2(x_query)
#         # x_proto, x_q = SFSA(x_proto, x_q)
#         # x_proto = Flatten(x_proto).view(n, k, -1)
#         # x_q = Flatten(x_q).view(n, q, -1)
#         x_proto_mean = x_proto.mean(1)  # (n, embed_dim)
#         x_proto_mean_2 = x_proto2.mean(1)
#
#
#
#
#
#         x_q_2= x_q.view(-1, x_q.shape[-1])  # (n*q, embed_dim)
#         x_q2_2 = x_q2.view(-1, x_q2.shape[-1])
#
#         # x_q_2 = x_q_2 + x_q2_2
#         # x_proto_mean = x_proto_mean + x_proto_mean_2
#
#
#
#
#
#         sim_result = self.similarity(x_q_2, x_proto_mean)  # (n*q, n)
#         sim_result2 = self.similarity(x_q2_2, x_proto_mean_2)
#         sim_result = (sim_result + sim_result2)/2
#         # list = [None] * 2
#         # list[0] = sim_result
#         # list[1] = sim_result2
#         # # list[2] = sim_result_total
#         # sv_predicted_proba, _ = self.soft_voting(torch.stack(list, dim=0))
#         # print(sv_predicted_proba)
#
#
#         # sim_result_noisy = self.similarity(x_q_noisy, x_proto_noisy)
#
#         log_p_y = F.log_softmax(sim_result, dim=1)
#         log_p_y_back = F.log_softmax(sim_result, dim=1)
#
#         # lop_p_y_cos = F.log_softmax(sim_result_cos, dim=1)
#         # print('log_p_y:', log_p_y)
#
#         # return log_p_y, log_p_y_back, lop_p_y_cos  # (n*q, n)
#         return log_p_y, log_p_y_back
#         # return log_p_y
#
#
#     @staticmethod
#     def similarity(a, b, sim_type='euclidean'):
#         methods = {'euclidean': euclidean_dist_similarity, 'cosine': cosine_similarity, 'kl': KL_dist}
#         assert sim_type in methods.keys(), f'type must be in {methods.keys()}'
#         return methods[sim_type](a, b)  # 值越大相似度越高