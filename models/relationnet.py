import torch.nn as nn
import torch.nn.functional as F
import torch
class CNNEncoder(nn.Module):
    """docstring for ClassName"""
    def __init__(self):
        super(CNNEncoder, self).__init__()
        self.layer1 = nn.Sequential(
                        nn.Conv2d(3,64,kernel_size=3,padding=0),
                        nn.BatchNorm2d(64, momentum=1, affine=True),
                        nn.ReLU(),
                        nn.MaxPool2d(2))
        self.layer2 = nn.Sequential(
                        nn.Conv2d(64,64,kernel_size=3,padding=0),
                        nn.BatchNorm2d(64, momentum=1, affine=True),
                        nn.ReLU(),
                        nn.MaxPool2d(2))
        self.layer3 = nn.Sequential(
                        nn.Conv2d(64,64,kernel_size=3,padding=1),
                        nn.BatchNorm2d(64, momentum=1, affine=True),
                        nn.ReLU())
        self.layer4 = nn.Sequential(
                        nn.Conv2d(64,64,kernel_size=3,padding=1),
                        nn.BatchNorm2d(64, momentum=1, affine=True),
                        nn.ReLU())

    def forward(self,x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        #out = out.view(out.size(0),-1)
        return out # 64

class RelationNetwork(nn.Module):
    """docstring for RelationNetwork"""
    def __init__(self,input_size,hidden_size):
        super(RelationNetwork, self).__init__()
        self.layer1 = nn.Sequential(
                        nn.Conv2d(128,64,kernel_size=3,padding=1),
                        nn.BatchNorm2d(64, momentum=1, affine=True),
                        nn.ReLU(),
                        nn.MaxPool2d(2))
        self.layer2 = nn.Sequential(
                        nn.Conv2d(64,64,kernel_size=3,padding=1),
                        nn.BatchNorm2d(64, momentum=1, affine=True),
                        nn.ReLU(),
                        nn.MaxPool2d(2))
        self.fc1 = nn.Linear(input_size,hidden_size)
        self.fc2 = nn.Linear(hidden_size,1)

    def forward(self,x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.view(out.size(0),-1)
        out = F.relu(self.fc1(out))
        out = F.sigmoid(self.fc2(out))
        return out

class RelationNet(nn.Module):
    """docstring for RelationNetwork"""
    def __init__(self,input_size,hidden_size):
        super(RelationNet, self).__init__()
        self.feature_encoder = CNNEncoder()
        self.RelationNetwork = RelationNetwork(10816, hidden_size)

    def forward(self, x, q):
        n_num = x.shape[0]
        k_num = x.shape[1]
        query_n = q.shape[1]
        x = x.view(-1, x.shape[2],x.shape[3],x.shape[4])
        q = q.view(-1, q.shape[2], q.shape[3], q.shape[4])
        x_f = self.feature_encoder(x)
        q_f = self.feature_encoder(q)


        x_f = x_f.view(n_num, k_num, x_f.shape[1], x_f.shape[2], x_f.shape[3])
        q_f = q_f.view(n_num, query_n, q_f.shape[1], q_f.shape[2], q_f.shape[3])
        x_f = x_f.repeat(1, query_n, 1, 1, 1)

        relation_map = torch.cat((x_f, q_f), dim=2)
        relation_pair = relation_map.view(-1, relation_map.shape[2], relation_map.shape[3], relation_map.shape[4])
        relations = self.RelationNetwork(relation_pair)
        return relations

