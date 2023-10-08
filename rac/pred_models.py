import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import Dataset, TensorDataset
import numpy as np

import torch
from rac.utils.models import TwoLayerNet, ThreeLayerNet, CifarNet, ResNet18, VGG, MnistNet

# Define the Contrastive Loss Function
class ContrastiveLoss(torch.nn.Module):
    def __init__(self, margin=2.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        # Calculate the euclidean distance and calculate the contrastive loss
        euclidean_distance = F.pairwise_distance(output1, output2, keepdim = True)
        loss_contrastive = torch.mean((1-label) * torch.pow(euclidean_distance, 2) +
                                    (label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))


        return loss_contrastive

class CustomTensorDataset(Dataset):
    """TensorDataset with support of transforms.
    """
    def __init__(self, tensors1, tensors2, labels=None, train=False, transform=None):
        self.tensors1 = tensors1
        self.tensors2 = tensors2
        self.labels = labels
        self.train = train
        self.transform = transform
        if self.train:
            num_neg, num_pos = int(np.unique(labels, return_counts=True)[1][0]), int(np.unique(labels, return_counts=True)[1][1])
            self.num_minority = num_pos if num_pos < num_neg else num_neg
            self.num_majority = num_pos if num_pos > num_neg else num_neg

    def __getitem__(self, index):
        if self.train:
            index = index if index < self.num_majority else self.num_majority + ((index - self.num_majority) % self.num_minority)
        x1 = self.tensors1[index]
        x2 = self.tensors2[index]

        if self.transform:
            x1 = self.transform(x1)
            x2 = self.transform(x2)

        if self.labels is not None:
            y = self.labels[index]
            return x1, x2, y

        return x1, x2

    def __len__(self):
        if self.train:
            return 2*self.num_majority
        else:
            return len(self.tensors1)


#self.net = ACCNet(TwoLayerNet(input_dim, 512, 1024), TwoLayerNet(input_dim, 512, 1024)).to(self.device)
class ACCNet(nn.Module):
    def __init__(self, base_net="resnet", siamese=False, output="", input_dim=128):
        super(ACCNet, self).__init__()
        self.base_net = base_net
        self.siamese = siamese
        #self.net = ACCNet(TwoLayerNet(input_dim, 512, 1024), TwoLayerNet(input_dim, 512, 1024)).to(self.device)
        #self.net = ACCNet(TwoLayerNet(input_dim, 512, 1024), TwoLayerNet(input_dim, 512, 1024)).to(self.device)
        #self.net = ACCNet(CifarNet(), CifarNet()).to(self.device)
        #self.net = ACCNet(ResNet18(), ResNet18()).to(self.device)
        kwargs = {}
        if base_net == "resnet":
            base_net_model = ResNet18
            emb_dim = 512
        elif base_net == "vgg16":
            base_net_model = VGG
            kwargs = {"vgg_name": "VGG16"}
            emb_dim = 512
        elif base_net == "mnistnet":
            base_net_model = MnistNet
            emb_dim = 128
        elif base_net == "cifarnet":
            base_net_model = CifarNet
            emb_dim = 256
        elif base_net == "three_layer_net":
            base_net_model = ThreeLayerNet
            kwargs = {"input_dim": input_dim, "num_classes": 256, "h1": 1024, "h2": 512}
            emb_dim = 256
        else:
            raise ValueError("WRONG BASE NET")
        
        if siamese:
            self.net1 = base_net_model(**kwargs)
            self.combined_net1 = ThreeLayerNet(emb_dim*2, 256, 2048, 512)
            self.combined_net2 = ThreeLayerNet(256, 1, 128, 32)
        else:
            self.net1 = base_net_model(**kwargs)
            self.net2 = base_net_model(**kwargs)
            self.combined_net1 = ThreeLayerNet(emb_dim*2, 256, 2048, 512)
            self.combined_net2 = ThreeLayerNet(256, 1, 128, 32)

    def forward(self, X1, X2):
        #if X1.shape[1] != 2:
            #raise ValueError("WRONG INPUT DIM")
        
        if self.siamese:
            if self.base_net == "three_layer_net":
                out1 = F.relu(self.net1(X1))
                out2 = F.relu(self.net1(X2))
            else:
                _, out1 = self.net1(X1, last=True)
                _, out2 = self.net1(X2, last=True)
        else:
            if self.base_net == "three_layer_net":
                out1 = F.relu(self.net1(X1))
                out2 = F.relu(self.net2(X2))
            else:
                _, out1 = self.net1(X1, last=True)
                _, out2 = self.net2(X2, last=True)
        combined = torch.cat((out1, out2), 1)

        #print("OUT1: ", e1.shape)
        #print("OUT2: ", e2.shape)
        res1 = F.relu(self.combined_net1(combined))
        #return torch.clip(self.combined_net(combined), min=-1, max=1)
        return self.combined_net2(res1)