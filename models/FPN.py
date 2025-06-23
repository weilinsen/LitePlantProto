import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torch
from utils import cosine_similarity, euclidean_dist_similarity
from .SpatialFormer import SFSA
from torchvision import models


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

        self.eca1 = SimAM()
        self.eca2 = SimAM()
        self.eca3 = SimAM()

        self.upconv3 = BasicConv2d(hidden_channels, hidden_channels, kernel_size=3, stride=1, padding=1, relu=True)
        self.upconv2 = BasicConv2d(hidden_channels, hidden_channels, kernel_size=3, stride=1, padding=1, relu=True)
        self.upconv1 = BasicConv2d(hidden_channels, hidden_channels, kernel_size=3, stride=1, padding=1, relu=True)
        self.flatt = Flatten()

    def forward(self, x):
        raw_shape = x.shape
        x = x.view(-1, *raw_shape[-3:])

        x1 = self.cb1(x)
        x2 = self.cb2(x1)
        x3 = self.cb3(x2)
        x4 = self.cb4(x3)

        x1_1 = self.rfb1_1(x1)  # channel -> 32
        x2_1 = self.rfb2_1(x2)  # channel -> 32
        x3_1 = self.rfb3_1(x3)  # channel -> 32
        x4_1 = self.rfb4_1(x4)  # channel -> 32

        x43 = self.feb3(x3_1, x4_1)
        out43 = self.upconv3(self.eca3(x43) + x43)
        x = self.flatt(out43)
        #
        # x432 = self.feb2(x2_1, out43)
        # out432 = self.upconv2(self.eca2(x432) + x432)
        #
        # x = self.flatt(out432)

        # x4321 = self.feb1(x1_1, out432)
        # out4321 = self.upconv1(self.eca1(x4321) + x4321)


        # x = self.flatt(out4321)
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

class Resnet18(nn.Module):
    def __init__(self):
        super(Resnet18, self).__init__()
        self.backbone = torchvision.models.resnet18(pretrained=False)
        self.flatten = Flatten()

    def forward(self, x):
        raw_shape = x.shape
        x = x.view(-1, *raw_shape[-3:])
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
        self.sigmoid = nn.Sigmoid()
    def forward(self, x, y):
        y = self.upsample(y, scale_factor=2)
        # print(x.shape)
        # print(y.shape)

        xy = torch.cat((x, y), 1)
        # out = self.conv(xy)
        xy_map = self.sigmoid(self.conv(xy))
        ex = x * xy_map + x
        ey = y * xy_map + y
        return ex + ey
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
        # # Spatial transformer localization-network
        # self.localization = nn.Sequential(
        #     nn.Conv2d(64, 8, kernel_size=5),
        #     nn.MaxPool2d(2, stride=2),
        #     nn.ReLU(True),
        #     nn.Conv2d(8, 10, kernel_size=3),
        #     nn.MaxPool2d(2, stride=2),
        #     nn.ReLU(True)
        # )
        #
        # # Regressor for the 3 * 2 affine matrix
        # self.fc_loc = nn.Sequential(
        #     nn.Linear(10 * 12 * 12, 32),
        #     nn.ReLU(True),
        #     nn.Linear(32, 3 * 2)
        # )
        #
        # # Initialize the weights/bias with identity transformation
        # self.fc_loc[2].weight.data.zero_()
        # self.fc_loc[2].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))

        self.backbone = torchvision.models.resnet18(pretrained=False)

        self.rfb1_1 = RFB_modified(64, channel)
        self.rfb2_1 = RFB_modified(128, channel)
        self.rfb3_1 = RFB_modified(256, channel)
        self.rfb4_1 = RFB_modified(512, channel)

        # 互增强模块
        self.feb1 = FEB(channel)
        self.feb2 = FEB(channel)
        self.feb3 = FEB(channel)

        self.eca1 = DualAxisConv2d(channel)
        self.eca2 = DualAxisConv2d(channel)
        self.eca3 = DualAxisConv2d(channel)
        self.upconv3 = BasicConv2d(channel, channel, kernel_size=3, stride=1, padding=1, relu=True)
        self.upconv2 = BasicConv2d(channel, channel, kernel_size=3, stride=1, padding=1, relu=True)
        self.upconv1 = BasicConv2d(channel, channel, kernel_size=3, stride=1, padding=1, relu=True)

        self.flatten = Flatten()

    # # Spatial transformer network forward function
    # def stn(self, x):
    #     xs = self.localization(x)
    #     # print(xs.shape)
    #     # print(xs.shape)
    #     xs = xs.view(-1, 10 * 12 * 12)
    #     theta = self.fc_loc(xs)
    #     theta = theta.view(-1, 2, 3)
    #
    #     grid = F.affine_grid(theta, x.size())
    #     x = F.grid_sample(x, grid)
    #     return x

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
def add_gaussian_noise(tensor, mean=0, std=1):
    noise = torch.randn(tensor.size()) * std + mean
    noisy_tensor = tensor.cpu() + noise
    return noisy_tensor.cuda()


import torch.nn as nn
import torch.nn.functional as F



class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, stride=stride, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out


class FPN(nn.Module):
    def __init__(self, block, layers):
        super(FPN, self).__init__()
        self.in_planes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        # Bottom-up layers
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.upsample = cus_sample
        self.flatten = Flatten()
        # Top layer
        self.toplayer = nn.Conv2d(2048, 256, kernel_size=1, stride=1, padding=0)  # Reduce channels
        # Smooth layers
        self.smooth1 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.smooth2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.smooth3 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        # Lateral layers
        self.latlayer1 = nn.Conv2d(1024, 256, kernel_size=1, stride=1, padding=0)
        self.latlayer2 = nn.Conv2d(512, 256, kernel_size=1, stride=1, padding=0)
        self.latlayer3 = nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_planes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_planes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )
        layers = []
        layers.append(block(self.in_planes, planes, stride, downsample))
        self.in_planes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.in_planes, planes))
        return nn.Sequential(*layers)

    def _upsample_add(self, x, y):
        _, _, H, W = y.size()
        return F.interpolate(x, size=(H, W), mode='bilinear', align_corners=True) + y

    def forward(self, x):
        raw_shape = x.shape
        x = x.view(-1, *raw_shape[-3:])
        # Bottom-up
        c1 = F.relu(self.bn1(self.conv1(x)))
        c1 = F.max_pool2d(c1, kernel_size=3, stride=2, padding=1)
        # print(f'c1:{c1.shape}')
        c2 = self.layer1(c1)
        # print(f'c2:{c2.shape}')
        c3 = self.layer2(c2)
        # print(f'c3:{c3.shape}')
        c4 = self.layer3(c3)
        # print(f'c4:{c4.shape}')
        c5 = self.layer4(c4)
        # print(f'c5:{c5.shape}')
        # Top-down
        p5 = self.toplayer(c5)
        # print(f'p5:{p5.shape}')
        p4 = self._upsample_add(p5, self.latlayer1(c4))
        # print(f'latlayer1(c4):{self.latlayer1(c4).shape}, p4:{p4.shape}')
        p3 = self._upsample_add(p4, self.latlayer2(c3))
        # print(f'latlayer1(c3):{self.latlayer2(c3).shape}, p3:{p3.shape}')
        p2 = self._upsample_add(p3, self.latlayer3(c2))
        # print(f'latlayer1(c2):{self.latlayer3(c2).shape}, p2:{p2.shape}')
        # Smooth
        p4 = self.smooth1(p4)
        p3 = self.smooth2(p3)
        p2 = self.smooth3(p2)
        p4 = self.upsample(p4, scale_factor=4)
        p3 = self.upsample(p3, scale_factor=2)
        print(p4.shape)
        print(p3.shape)
        print(p2.shape)
        # out = p4+p3+p2
        out = p2
        out = self.flatten(out)

        # x = self.encoder(x)
        out = out.view(*raw_shape[:-3], -1)
        print(out.shape)
        return out




class FPN_Network(nn.Module):
    def __init__(self, input_img_channels, hidden_channels):
        super(FPN_Network, self).__init__()
        # self.encoder = ConvEncoder(input_img_channels, hidden_channels)
        # self.encoder = ConvEncoderC2F(input_img_channels, hidden_channels)
        # self.encoder = Resnet18()
        # self.encoder = Resnet18C2F(64)
        self.encoder = FPN(Bottleneck, [3,4,6,3])
        # self.SFSA = SFSA()

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
        x_proto_mean = x_proto.mean(1)  # (n, embed_dim)
        x_q = self.encoder(x_query)  # (n, q, embed_dim)
        x_q_2= x_q.view(-1, x_q.shape[-1])  # (n*q, embed_dim)

        # 反向
        x_q_mean = x_q.mean(1)
        x_p = x_proto.view(-1, x_proto.shape[-1])

        # # 这里是噪声图片的计算
        # x_proto_noisy = self.encoder(noisy_support)  # (n, k, embed_dim)
        # x_proto_noisy = x_proto_noisy.mean(1)  # (n, embed_dim)
        # x_q_noisy = self.encoder(noisy_query)  # (n, q, embed_dim)
        # x_q_noisy = x_q_noisy.view(-1, x_q_noisy.shape[-1])  # (n*q, embed_dim)

        sim_result = self.similarity(x_q_2, x_proto_mean)  # (n*q, n)

        sim_result_back = self.similarity(x_p, x_q_mean)

        # sim_result_noisy = self.similarity(x_q_noisy, x_proto_noisy)

        log_p_y = F.log_softmax(sim_result, dim=1)
        log_p_y_back = F.log_softmax(sim_result_back, dim=1)
        print('log_p_y:', log_p_y.shape)

        return log_p_y, log_p_y_back  # (n*q, n)
        # return log_p_y

    @staticmethod
    def similarity(a, b, sim_type='cosine'):
        methods = {'euclidean': euclidean_dist_similarity, 'cosine': cosine_similarity}
        assert sim_type in methods.keys(), f'type must be in {methods.keys()}'
        return methods[sim_type](a, b)  # 值越大相似度越高
