import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
# from layers.gcn import TreeGCN
from model.gcn import TreeGCN

from math import ceil

class Discriminator(nn.Module):
    def __init__(self, features,version=0):
        self.layer_num = len(features)-1
        super(Discriminator, self).__init__()

        self.fc_layers = nn.ModuleList([])
        for inx in range(self.layer_num):
            self.fc_layers.append(nn.Conv1d(features[inx], features[inx+1], kernel_size=1, stride=1))

        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2)

        self.final_layer = nn.Sequential(
                    nn.Linear(features[-1], 128),
                    nn.LeakyReLU(negative_slope=0.2),
                    nn.Linear(128, 64),
                    nn.LeakyReLU(negative_slope=0.2),
                    nn.Linear(64, 1),
                    nn.Sigmoid())

    def forward(self, f):
        
        feat = f.transpose(1,2)
        vertex_num = feat.size(2)

        for inx in range(self.layer_num):
            feat = self.fc_layers[inx](feat)
            feat = self.leaky_relu(feat)
        out1 = F.max_pool1d(input=feat, kernel_size=vertex_num).squeeze(-1)
        out = self.final_layer(out1) # (B, 1)
        return out,out1


class Generator(nn.Module):
    def __init__(self,features,degrees,support,args=None):
        self.layer_num = len(features)-1
        assert self.layer_num == len(degrees), "Number of features should be one more than number of degrees."
        self.pointcloud = None
        super(Generator, self).__init__()
        
        vertex_num = 1
        self.gcn = nn.Sequential()
        for inx in range(self.layer_num):
            # NOTE last layer activation False
            if inx == self.layer_num-1:
                self.gcn.add_module('TreeGCN_'+str(inx),
                                    TreeGCN(inx, features, degrees, 
                                            support=support, node=vertex_num, upsample=True, activation=False,args=args))
            else:
                self.gcn.add_module('TreeGCN_'+str(inx),
                                    TreeGCN(inx, features, degrees, 
                                            support=support, node=vertex_num, upsample=True, activation=True,args=args))
            vertex_num = int(vertex_num * degrees[inx])

    def forward(self, tree):
        tree = torch.unsqueeze(tree, 1)
        tree = [tree]
        feat = self.gcn(tree)
        
        self.pointcloud = feat[-1]
        
        return self.pointcloud

    def getPointcloud(self):
        return self.pointcloud[-1] # return a single point cloud (2048,3)

    def get_params(self,index):
        
        if index < 7:
            for param in self.gcn[index].parameters():
                yield param
        else:
            raise ValueError('Index out of range')

class Encoder_Generator(nn.Module):
    def __init__(self,features,degrees,support,args=None):
        self.layer_num = len(features)-1
        assert self.layer_num == len(degrees), "Number of features should be one more than number of degrees."
        self.pointcloud = None
        super(Encoder_Generator, self).__init__()
        
        #Encoder
        self.encoder = nn.Sequential()
        self.encoder.add_module('encoder_layer1', nn.Conv1d(3, 64, 1))
        self.encoder.add_module('encoder_layer2', nn.Conv1d(64, 128, 1))
        self.encoder.add_module('encoder_layer3', nn.Conv1d(128, 128, 1))
        self.encoder.add_module('encoder_layer4', nn.Conv1d(128, 96, 1))
        #Generator
        vertex_num = 1
        self.gcn = nn.Sequential()
        for inx in range(self.layer_num):
            # NOTE last layer activation False
            if inx == self.layer_num-1:
                self.gcn.add_module('TreeGCN_'+str(inx),
                                    TreeGCN(inx, features, degrees, 
                                            support=support, node=vertex_num, upsample=True, activation=False,args=args))
            else:
                self.gcn.add_module('TreeGCN_'+str(inx),
                                    TreeGCN(inx, features, degrees, 
                                            support=support, node=vertex_num, upsample=True, activation=True,args=args))
            vertex_num = int(vertex_num * degrees[inx])

    def forward(self, tree):
        tree = tree[0].permute([0,2,1])
        tree = self.encoder(tree)
        tree = tree.permute([0,2,1])
        tree = torch.max(tree, dim=1, keepdim=True)[0]
        hidden_z = tree
        tree = [tree]
        feat = self.gcn(tree)
        
        self.pointcloud = feat[-1]
        
        return self.pointcloud, hidden_z

    def getPointcloud(self):
        return self.pointcloud[-1] # return a single point cloud (2048,3)

    def get_gcn_params(self,index):
        
        if index < 7:
            for param in self.gcn[index].parameters():
                yield param
        else:
            raise ValueError('Index out of range')
    def get_encoder_params(self,index):
        if index < 4:
            for param in self.encoder[index].parameters():
                yield param
        else:
            raise ValueError('Index out of range')


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()

        # first shared mlp
        self.conv1 = nn.Conv1d(3, 128, 1)
        self.conv2 = nn.Conv1d(128, 256, 1)
        self.bn1 = nn.BatchNorm1d(128)
        self.bn2 = nn.BatchNorm1d(256)

        # second shared mlp
        self.conv3 = nn.Conv1d(512, 512, 1)
        self.conv4 = nn.Conv1d(512, 1024, 1)
        self.bn3 = nn.BatchNorm1d(512)
        self.bn4 = nn.BatchNorm1d(1024)
    
    def forward(self, x):
        x = x[0].permute([0,2,1])
        n = x.size()[2]

        # first shared mlp
        x = F.relu(self.bn1(self.conv1(x)))           # (B, 128, N)
        f = self.bn2(self.conv2(x))                   # (B, 256, N)
        
        # point-wise maxpool
        g = torch.max(f, dim=2, keepdim=True)[0]      # (B, 256, 1)
        
        # expand and concat
        x = torch.cat([g.repeat(1, 1, n), f], dim=1)  # (B, 512, N)

        # second shared mlp
        x = F.relu(self.bn3(self.conv3(x)))           # (B, 512, N)
        x = self.bn4(self.conv4(x))                   # (B, 1024, N)
        
        # point-wise maxpool
        v = torch.max(x, dim=-1)[0]                   # (B, 1024)
        
        return v


class Decoder(nn.Module):
    def __init__(self, num_coarse=1024, num_dense=2048):
        super(Decoder, self).__init__()

        self.num_coarse = num_coarse
        
        # fully connected layers
        self.linear1 = nn.Linear(1024, 1024)
        self.linear2 = nn.Linear(1024, 1024)
        self.linear3 = nn.Linear(1024, 3 * num_coarse)
        self.bn1 = nn.BatchNorm1d(1024)
        self.bn2 = nn.BatchNorm1d(1024)

        # shared mlp
        self.conv1 = nn.Conv1d(3+2+1024, 512, 1)
        self.conv2 = nn.Conv1d(512, 512, 1)
        self.conv3 = nn.Conv1d(512, 3, 1)
        self.bn3 = nn.BatchNorm1d(512)
        self.bn4 = nn.BatchNorm1d(512)

        # 2D grid
        grids = np.meshgrid(np.linspace(-0.05, 0.05, 2, dtype=np.float32),
                            np.linspace(-0.05, 0.05, 1, dtype=np.float32))                               # (2, 4, 44)
        self.grids = torch.Tensor(grids).view(2, -1)  # (2, 4, 4) -> (2, 16)
    
    def forward(self, x):
        b = x.size()[0]
        # global features
        v = x  # (B, 1024)

        # fully connected layers to generate the coarse output
        x = F.relu(self.bn1(self.linear1(x)))
        x = F.relu(self.bn2(self.linear2(x)))
        x = self.linear3(x)
        y_coarse = x.view(-1, 3, self.num_coarse)  # (B, 3, 1024)

        repeated_centers = y_coarse.unsqueeze(3).repeat(1, 1, 1, 2).view(b, 3, -1)  # (B, 3, 16x1024)
        repeated_v = v.unsqueeze(2).repeat(1, 1, 2 * self.num_coarse)               # (B, 1024, 16x1024)
        grids = self.grids.to(x.device)  # (2, 16)
        grids = grids.unsqueeze(0).repeat(b, 1, self.num_coarse)                     # (B, 2, 16x1024)

        x = torch.cat([repeated_v, grids, repeated_centers], dim=1)                  # (B, 2+3+1024, 16x1024)
        x = F.relu(self.bn3(self.conv1(x)))
        x = F.relu(self.bn4(self.conv2(x)))
        x = self.conv3(x)                # (B, 3, 16x1024)
        y_detail = x + repeated_centers  # (B, 3, 16x1024)
        y_detail = y_detail.permute([0,2,1])

        return y_detail

class Disentangler(nn.Module):
    def __init__(self, f_dims=512):
        super(Disentangler, self).__init__()
        # first fc
        self.fc1 = nn.Linear(1024, 1024)
        self.bn1_fc = nn.BatchNorm1d(1024)
        # second fc
        self.fc2 = nn.Linear(1024, f_dims)
        self.bn2_fc = nn.BatchNorm1d(f_dims)
    def forward(self, x):
        x = F.relu(self.bn1_fc(self.fc1(x)))
        x = F.relu(self.bn2_fc(self.fc2(x)))
        return x

class Z_Mapper(nn.Module):
    def __init__(self, f_dims=512):
        super(Z_Mapper, self).__init__()
        #first fc
        self.fc1 = nn.Linear(f_dims, f_dims)
        self.bn1_fc = nn.BatchNorm1d(f_dims)
        #second fc
        self.fc2 = nn.Linear(f_dims, f_dims)
        self.bn2_fc = nn.BatchNorm1d(f_dims)
    def forward(self, x):
        x = F.relu(self.bn1_fc(self.fc1(x)))
        x = F.relu(self.bn2_fc(self.fc2(x)))
        return x

class Classifier(nn.Module):
    def __init__(self, f_dims=512):
        super(Classifier, self).__init__()
        #first fc
        self.fc1 = nn.Linear(f_dims, f_dims//4)
        self.bn1_fc = nn.BatchNorm1d(f_dims//4)
        #second fc
        self.fc2 = nn.Linear(f_dims//4, 2) 
    def forward(self, x):
        x = F.relu(self.bn1_fc(self.fc1(x)))
        x = self.fc2(x)
        return x

class ViewPredictor(nn.Module):
    def __init__(self, f_dims=12, out_dims=2):
        super(ViewPredictor, self).__init__()
        #first fc
        self.fc1 = nn.Linear(f_dims, f_dims)
        self.bn1_fc = nn.BatchNorm1d(f_dims)
        #second fc
        self.fc2 = nn.Linear(f_dims, out_dims)
    def forward(self, x):
        x = F.relu(self.bn1_fc(self.fc1(x)))
        x = self.fc2(x)
        return x

class PCN(nn.Module):
    def __init__(self):
        super(PCN, self).__init__()
        self.pointcloud = None

        self.encoder = Encoder()
        self.decoder = Decoder()
    
    def forward(self, x):
        v = self.encoder(x)
        y = self.decoder(v)
        self.pointcloud = y 
        return y, v

    def getPointcloud(self):
        return self.pointcloud[-1] # return a single point cloud (2048,3)
