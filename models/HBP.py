import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torch
from utils import cosine_similarity, euclidean_dist_similarity
from .SpatialFormer import SFSA
from torchvision import models
# from pvtv2 import pvt_v2_b2
from .Resnet.Resnet import resnet50
from .VGG.VGG import vgg16
import math
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

class HBP(torch.nn.Module):
    def __init__(self, pretrained):
        super(HBP, self).__init__()
        # Convolution and pooling layers of VGG-16.
        # self.features = torchvision.models.vgg16(pretrained=pretrained).features
        self.backbone = vgg16()
        if pretrained:
            path = './vgg16-397923af.pth'
            save_model = torch.load(path)
            model_dict = self.backbone.state_dict()
            state_dict = {k: v for k, v in save_model.items() if k in model_dict.keys()}
            model_dict.update(state_dict)
        self.features = self.backbone.features
        # print(*list(self.features.children()))
        # for i in range(len(list(self.features.children()))):
        #     print(i)
        #     print(list(self.features.children())[i])
        self.features_conv5 = torch.nn.Sequential(*list(self.features.children()))[:-7]
        self.features_conv5_1 = torch.nn.Sequential(*list(self.features.children())[-7:-5])

        # self.features_conv5_1 = torch.nn.Sequential(*list(self.features.children())[:-5])
        self.features_conv5_2 = torch.nn.Sequential(*list(self.features.children())[-5:-3])
        self.features_conv5_3 = torch.nn.Sequential(*list(self.features.children())[-3:-1])
        self.bilinear_proj_1 = torch.nn.Conv2d(512, 8192, kernel_size=1, bias=True)
        self.bilinear_proj_2 = torch.nn.Conv2d(512, 8192, kernel_size=1, bias=True)
        self.bilinear_proj_3 = torch.nn.Conv2d(512, 8192, kernel_size=1, bias=True)
        self.sim = SimAM()
        # self.se = SE_Block(1536)
        self.ca = CALayer(1536)
        self.refine1 = Refine(512)
        self.refine2 = Refine(512)
        self.refine3 = Refine(512)
        # Linear classifier.
        self.fc = torch.nn.Linear(8192 * 3, 5)
        if pretrained:
            # Freeze all previous layers.
            for param in self.features_conv5_1.parameters():
                param.requires_grad = False
            for param in self.features_conv5_2.parameters():
                param.requires_grad = False
            for param in self.features_conv5_3.parameters():
                param.requires_grad = False

        # Initialize the fc layers.
        torch.nn.init.xavier_normal_(self.fc.weight.data)
        if self.fc.bias is not None:
            torch.nn.init.constant_(self.fc.bias.data, val=0)

    def hbp_1_2(self, conv1, conv2):
        N = conv1.size()[0]
        proj_1 = self.bilinear_proj_1(conv1)
        proj_2 = self.bilinear_proj_2(conv2)
        assert (proj_1.size() == (N, 8192, 14, 14))
        X = proj_1 * proj_2
        assert (X.size() == (N, 8192, 14, 14))
        X = torch.sum(X.view(X.size()[0], X.size()[1], -1), dim=2)
        X = X.view(N, 8192)
        X = torch.sign(X) * torch.sqrt(torch.abs(X) + 1e-5)
        X = torch.nn.functional.normalize(X)
        return X

    def hbp_1_3(self, conv1, conv3):
        N = conv1.size()[0]
        proj_1 = self.bilinear_proj_1(conv1)
        proj_3 = self.bilinear_proj_3(conv3)
        assert (proj_1.size() == (N, 8192, 14, 14))
        X = proj_1 * proj_3
        assert (X.size() == (N, 8192, 14, 14))
        X = torch.sum(X.view(X.size()[0], X.size()[1], -1), dim=2)
        X = X.view(N, 8192)
        X = torch.sign(X) * torch.sqrt(torch.abs(X) + 1e-5)
        X = torch.nn.functional.normalize(X)
        return X

    def hbp_2_3(self, conv2, conv3):
        N = conv2.size()[0]
        proj_2 = self.bilinear_proj_2(conv2)
        proj_3 = self.bilinear_proj_3(conv3)
        assert (proj_2.size() == (N, 8192, 14, 14))
        X = proj_2 * proj_3
        assert (X.size() == (N, 8192, 14, 14))
        X = torch.sum(X.view(X.size()[0], X.size()[1], -1), dim=2)
        X = X.view(N, 8192)
        X = torch.sign(X) * torch.sqrt(torch.abs(X) + 1e-5)
        X = torch.nn.functional.normalize(X)
        return X

    def forward(self,X):
        N = X.size()[0]
        # assert X.size() == (N, 3, 224, 224)
        X = self.features_conv5(X)
        X = self.sim(X)
        X_conv5_1 = self.features_conv5_1(X)
        X_conv5_2 = self.features_conv5_2(X_conv5_1)
        X_conv5_3 = self.features_conv5_3(X_conv5_2)

        X_conv5_1 = self.refine1(X_conv5_1)
        X_conv5_2 = self.refine2(X_conv5_2)
        X_conv5_3 = self.refine3(X_conv5_3)
        X = torch.cat((X_conv5_1, X_conv5_2, X_conv5_3), dim=1)
        X = self.ca(X).view(X.shape[0], -1)
        # X = self.se(X).view(X.shape[0], -1)
        return X



class PrototypicalNetwork3(nn.Module):
    def __init__(self, input_img_channels, hidden_channels):
        super(PrototypicalNetwork3, self).__init__()
        # self.encoder = ConvEncoder(input_img_channels, hidden_channels)
        self.encoder = HBP(pretrained=True)
        # self.encoder = ConvEncoderC2F(input_img_channels, hidden_channels)
        # self.encoder = Resnet18()
        # self.encoder = Resnet18C2F(64)
        # self.SFSA = SFSA()
        self.hidden_channels = hidden_channels
        self.hidden_size = 12544
        self.sim = SimAM()
        shots = 5
        self.conv1 = nn.Conv2d(1, 32, (shots, 1), padding=(shots // 2, 0))
        self.conv2 = nn.Conv2d(32, 64, (shots, 1), padding=(shots // 2, 0))
        self.conv_final = nn.Conv2d(64, 1, (shots, 1), stride=(shots, 1))
        self.concat_conv = nn.Conv2d(2*hidden_channels, hidden_channels, 3, 1, 1)
        self.drop = nn.Dropout()
        # self.sim = ECA(64)
        self.fc = nn.Linear(self.hidden_size, self.hidden_size)
    def eudis(self, x, y, dim,  fea_att_score=None):
        if fea_att_score is None:
            return (torch.pow(x - y, 2)).sum(dim)
        else:
            return (torch.pow(x - y, 2) * fea_att_score).sum(dim)


    def test_mode(self, x_support, x_query):
        n = x_support.shape[0]
        k = x_support.shape[1]
        q = x_query.shape[1]
        nq = n * q
        c = self.hidden_channels

        x_proto = self.encoder(x_support)  # (n, k, embed_dim)

        # # feature-level attention
        # fea_att_score = x_proto.view(n, 1, k, x_proto.shape[2])  # (N, 1, K, D)
        # fea_att_score = F.relu(self.conv1(fea_att_score))  # (N, 32, K, D)
        # fea_att_score = F.relu(self.conv2(fea_att_score))  # (N, 64, K, D)
        # fea_att_score = self.drop(fea_att_score)
        # fea_att_score = self.conv_final(fea_att_score)  # (N, 1, 1, D)
        # fea_att_score = F.relu(fea_att_score)
        # fea_att_score = fea_att_score.view(n, x_proto.shape[-1]).unsqueeze(0)  # (1, N, D)

        # h = w = int(math.sqrt(x_proto.shape[2] // c))
        # x_proto = x_proto.contiguous().view(n * k, c, w, h)
        # x_proto = self.sim(x_proto).view(n, k, -1)
        x_q = self.encoder(x_query)  # (n, q, embed_dim)
        # x_q = x_q.contiguous().view(n * q, c, w, h)
        # x_q = self.sim(x_q).view(n, q, -1)
        x_q = x_q.view(-1, x_q.shape[-1])  # (n*q, embed_dim)
        x_proto = x_proto.unsqueeze(0).expand(nq, -1, -1, -1)  # nq n k d
        x_proto_att = self.fc(x_proto.contiguous().view(-1, self.hidden_size))  # nq n k d
        x_q_2 = x_q.view(-1, x_q.shape[-1])  # (n*q, embed_dim)
        query_att = self.fc(x_q_2.unsqueeze(1).unsqueeze(2).expand(-1, n, k, -1).contiguous().view(-1, self.hidden_size))  # nq n k d
        x_proto_att = x_proto_att.view(nq, n, k, -1)
        query_att = query_att.view(nq, n, k, -1)
        ins_att_score = F.softmax(torch.tanh(x_proto_att * query_att).sum(-1), dim=-1)  # (nq, n, k)
        support_proto = (x_proto * ins_att_score.unsqueeze(3).expand(-1, -1, -1, self.hidden_size)).sum(2)  # (NQ, N, D)
        x_proto = support_proto.squeeze(0)  # n d
        # x_proto = x_proto.mean(1)  # (n, embed_dim)


        # sim_result = self.similarity(x_q, x_proto[0])  # (n*q, n)
        sim_result = -self.eudis(support_proto, x_q_2.unsqueeze(1), 2)

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
        n = x_support.shape[0]
        k = x_support.shape[1]
        q = x_query.shape[1]
        c = self.hidden_channels
        nq = n*q
        x_support = x_support.view(-1, x_support.shape[2], x_support.shape[3], x_support.shape[4])
        x_query = x_query.view(-1, x_query.shape[2], x_query.shape[3], x_query.shape[4])
        x_proto = self.encoder(x_support)  # (n*k, d)
        x_query = self.encoder(x_query)# (n*q, d)
        x_proto = x_proto.view(n, k,-1).mean(1) # (n d)
        x_query = x_query.view(n*q, -1) # (n*q, d)


        sim_result = self.similarity(x_query, x_proto)  # (n*q, n)



        log_p_y = F.log_softmax(sim_result, dim=1)

        # print('log_p_y:', log_p_y)

        return log_p_y  # (n*q, n)
        # return log_p_y

    @staticmethod
    def similarity(a, b, sim_type='euclidean'):
        methods = {'euclidean': euclidean_dist_similarity, 'cosine': cosine_similarity}
        assert sim_type in methods.keys(), f'type must be in {methods.keys()}'
        return methods[sim_type](a, b)  # 值越大相似度越高