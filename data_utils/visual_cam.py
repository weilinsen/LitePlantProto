import numpy as np
import cv2
import os
import torchvision.models as models
import torchvision.transforms as transforms
import torch
import pytorch_grad_cam
from pytorch_grad_cam.utils.image import show_cam_on_image
from PIL import Image
from shufflenet import *
from protonet import Resnet18SE, PVT, EfficientNet, PVT_tSF
from pvtv2 import pvt_v2_b2
import math
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
# 1.定义模型结构，选取要可视化的层
# model = models.resnet18(pretrained=True)
# print(model)
# model.eval()
# # traget_layers = [resnet18.bn1]
# traget_layers = [model.layer4[1].bn2]

#####这里是ShuffleNet的网络权重######
# model = shufflenet_v2_x2_0()
# net_dict = model.state_dict()
# predict_model = torch.load('shufflenetv2_x2_0-8be3c8ee.pth')
# state_dict = {k: v for k, v in predict_model.items() if k in net_dict.keys()}  # 寻找网络中公共层，并保留预训练参数
# net_dict.update(state_dict)  # 将预训练参数更新到新的网络层
# model.load_state_dict(net_dict)
# print(model)
#################################

# #####这里是LPP的网络权重######
# model = EfficientNet()
# model_dict = model.state_dict()
# checkpoint = torch.load('./runs/exp846/checkpoint/proto_best.pth')
# model_list1 = list(checkpoint.keys())
# model_list2 = list(model_dict.keys())
# len1 = len(model_list1)
# len2 = len(model_list2)
# m, n = 0, 0
# while True:
#     if m >= len1 or n >= len2:
#         break
#     layername1, layername2 = model_list1[m], model_list2[n]
#     w1, w2 = checkpoint[layername1], model_dict[layername2]
#     if w1.shape != w2.shape:
#         continue
#     model_dict[layername2] = checkpoint[layername1]
#     m += 1
#     n += 1
# model.load_state_dict(model_dict)
# print(model)
#################################


#####这里用来生成tSF的图 PV2下 5 845
# model = PVT()
#
# model_dict = model.state_dict()
# checkpoint = torch.load('./runs/exp845/checkpoint/proto_best.pth')
# model_list1 = list(checkpoint.keys())
# model_list2 = list(model_dict.keys())
# len1 = len(model_list1)
# len2 = len(model_list2)
# m, n = 0, 0
# while True:
#     if m >= len1 or n >= len2:
#         break
#     layername1, layername2 = model_list1[m], model_list2[n]
#     w1, w2 = checkpoint[layername1], model_dict[layername2]
#     if w1.shape != w2.shape:
#         continue
#     model_dict[layername2] = checkpoint[layername1]
#     m += 1
#     n += 1
# model.load_state_dict(model_dict)
# print(model)

# resnet18.load_state_dict(checkpoint)
# resnet18.eval()
# print(resnet18)
# predict_model = torch.load('./runs/exp845/checkpoint/proto_best.pth')
# state_dict = {k: v for k, v in predict_model.items() if k in resnet18.keys()}  # 寻找网络中公共层，并保留预训练参数
# net_dict.update(state_dict)  # 将预训练参数更新到新的网络层
# model.load_state_dict(net_dict)
# checkpoint = torch.load('./runs/exp169/checkpoint/proto_best.pth')



# model = Resnet18SE(64)
# model_dict = model.state_dict()
# checkpoint = torch.load('./runs/exp847/checkpoint/proto_best.pth')
# model_list1 = list(checkpoint.keys())
# model_list2 = list(model_dict.keys())
# # print(model_list1)
# # print('-----')
# # print(model_list2)
# len1 = len(model_list1)
# len2 = len(model_list2)
#
# print(len1)
# # print('-----')
# print(len2)
# m, n = 0, 0
#
# while True:
#     if m >= len1 or n >= len2:
#         break
#     layername1, layername2 = model_list1[m], model_list2[n]
#     w1, w2 = checkpoint[layername1], model_dict[layername2]
#     print(w1.shape, w2.shape)
#     if w1.shape != w2.shape:
#         continue
#     model_dict[layername2] = checkpoint[layername1]
#     m += 1
#     n += 1
# model.load_state_dict(model_dict)
# model.eval()

# PVT-V2 原始
model = PVT()
#
path = 'pvt_v2_b2.pth'
save_model = torch.load(path)
model_dict = model.backbone.state_dict()
# print(len(save_model.keys()))
# print("---------------------")
# print(len(model_dict.keys()))

state_dict = {k: v for k, v in save_model.items() if k in model_dict.keys()}
print(state_dict)
model_dict.update(state_dict)
model.backbone.load_state_dict(model_dict)
model.eval()
print(model)

# net_dict = model.state_dict()
# predict_model = torch.load('./runs/exp847/checkpoint/proto_best.pth')
# state_dict = {k: v for k, v in predict_model.items() if k in model.keys()}  # 寻找网络中公共层，并保留预训练参数
# net_dict.update(state_dict)  # 将预训练参数更新到新的网络层
# model.load_state_dict(net_dict)
# model.eval()
# print(model)
# checkpoint = torch.load('./runs/exp169/checkpoint/proto_best.pth')
#
#
# model_dict = resnet18.state_dict()
# model_list1 = list(checkpoint.keys())
# model_list2 = list(model_dict.keys())
# len1 = len(model_list1)
# len2 = len(model_list2)
# m, n = 0, 0
# while True:
#     if m >= len1 or n >= len2:
#         break
#     layername1, layername2 = model_list1[m], model_list2[n]
#     w1, w2 = checkpoint[layername1], model_dict[layername2]
#     if w1.shape != w2.shape:
#         continue
#     model_dict[layername2] = checkpoint[layername1]
#     m += 1
#     n += 1
# resnet18.load_state_dict(model_dict)
#
# # resnet18.load_state_dict(checkpoint)
# resnet18.eval()
# print(resnet18)

# model = PVT_tSF()
# model_dict = model.state_dict()
# checkpoint = torch.load('./PVT-tsf/0/model_best.pth.tar')["state_dict"]
# model_list1 = list(checkpoint.keys())
# model_list2 = list(model_dict.keys())
# # print(model_list1)
# # print('-----')
# # print(model_list2)
# len1 = len(model_list1)
# len2 = len(model_list2)
# new_model_list1 = []
# for i in range(len1):
#     item = model_list1[i]
#     if(item.split(".")[-1] != 'total_ops' and item.split(".")[-1] != 'total_params'):
#         new_model_list1.append(item)
#
# print(len(new_model_list1))
# # print(len1)
# # print('-----')
# print(len2)
# m, n = 0, 0
#
# while True:
#     if m >= len1 or n >= len2:
#         break
#     layername1, layername2 = new_model_list1[m], model_list2[n]
#     w1, w2 = checkpoint[layername1], model_dict[layername2]
#     print(w1.shape, w2.shape)
#     if w1.shape != w2.shape:
#         continue
#     model_dict[layername2] = checkpoint[layername1]
#     m += 1
#     n += 1
# model.load_state_dict(model_dict)
# model.eval()
# print(model)
def reshape_transform(tensor, height=8, width=8):
    print(tensor.shape)
    # 去掉cls token
    result = tensor[:, :, :].reshape(tensor.size(0),
    int(math.sqrt(tensor.size(1))), int(math.sqrt(tensor.size(1))), tensor.size(2))

    # 将通道维度放到第一个位置
    result = result.transpose(2, 3).transpose(1, 2)
    return result

traget_layers = [model.backbone.norm4]
# traget_layers = [resnet18.backbone.layer4[1].bn2]

# 2.读取图片，将图片转为RGB
img_path = 'visual_pics/totamto1.jpg'
rgb_img = Image.open(img_path).convert('RGB')

# 3.图片预处理：resize、裁剪、归一化
trans = transforms.Compose([
    # transforms.Resize(224),
    # transforms.CenterCrop(224),
    transforms.ToTensor(),
])
crop_img = trans(rgb_img)
net_input = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))(crop_img).unsqueeze(0)

# 4.将裁剪后的Tensor格式的图像转为numpy格式，便于可视化
canvas_img = (crop_img*255).byte().numpy().transpose(1, 2, 0)
canvas_img = cv2.cvtColor(canvas_img, cv2.COLOR_RGB2BGR)

# 5.实例化cam
cam = pytorch_grad_cam.GradCAMPlusPlus(model=model, target_layers=traget_layers, reshape_transform=reshape_transform)
grayscale_cam = cam(net_input)
grayscale_cam = grayscale_cam[0, :]

# 6.将feature map与原图叠加并可视化
src_img = np.float32(canvas_img) / 255
visualization_img = show_cam_on_image(src_img, grayscale_cam, use_rgb=False)

cv2.imwrite('./visual_pics/totamto1_pvtv2_test.jpg', visualization_img)
