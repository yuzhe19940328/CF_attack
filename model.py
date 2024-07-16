import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd

import numpy as np
import networks 


class MNIST_model(torch.nn.Module):
    def __init__(self,args):
        super(MNIST_model, self).__init__()
        self.args = args
        self.backbone = networks.MNIST_CNN(args.input_shape)
        self.classifier=networks.Classifier(self.backbone.n_outputs,args.n_classes)
        self.network = nn.Sequential(self.backbone,self.classifier)
        self.loss = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=args.lr)

    def forward(self, x):
        return self.network(x)
    
    def update(self, x, y):
        self.optimizer.zero_grad()
        loss = self.loss(self.forward(x), y)
        loss.backward()
        self.optimizer.step()
        return loss.item()


class CelebA_model(torch.nn.Module):
    def __init__(self,args):
        super(CelebA_model, self).__init__()
        self.args = args
        self.backbone = networks.ResNet(args.input_shape,type='resnet18')
        self.classifiers=[]        
        for i in range(args.n_tasks):
            self.classifiers.append(networks.Classifier(self.backbone.n_outputs,2))
        self.loss = nn.CrossEntropyLoss()
        self.optimizer_feature_extractor = torch.optim.Adam(self.backbone.parameters(), lr=args.lr)
        
        self.head_optimizers = []
        for i in range(args.n_tasks):
            self.head_optimizers.append(torch.optim.Adam(self.classifiers[i].parameters(), lr=args.lr))


    def forward(self, x, haed_idx):
        head=self.classifiers[haed_idx]
        head.cuda()
        feature=self.backbone(x)
        return head(feature)

    def get_feature(self, x):
        z = self.backbone(x)
        return z


    def updata_head(self, x, y, head_idx):
        self.head_optimizers[head_idx].zero_grad()
        loss = self.loss(self.forward(x,head_idx), y)
        loss.backward()
        self.head_optimizers[head_idx].step()
        return loss.item()

    def update_feature_extractor(self, x, y, head_idx):
        self.optimizer_feature_extractor.zero_grad()
        loss = self.loss(self.forward(x,head_idx), y)
        loss.backward()
        self.optimizer_feature_extractor.step()
        return loss.item()

    def save(self, path):
        checkpoint={'backbone':self.backbone.state_dict()}
        for i in range(self.args.n_tasks):
            checkpoint[f'head_{i}']=self.classifiers[i].state_dict()
        torch.save(checkpoint, path)
        return


    def load(self,path):
        checkpoint=torch.load(path)
        self.backbone.load_state_dict(checkpoint['backbone'])
        for i in range(self.args.n_tasks):
            self.classifiers[i].load_state_dict(checkpoint[f'head_{i}'])
        return



class DeepFashion_model(torch.nn.Module):
    def __init__(self,args):
        super(DeepFashion_model, self).__init__()
        self.args = args
        self.backbone = networks.ResNet(args.input_shape)
        self.classifier=networks.Classifier(self.backbone.n_outputs,args.n_classes)
        self.network = nn.Sequential(self.backbone,self.classifier)
        self.loss = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=args.lr)

    def forward(self, x):
        return self.network(x)
    
    def update(self, x, y):
        self.optimizer.zero_grad()
        loss = self.loss(self.forward(x), y)
        loss.backward()
        self.optimizer.step()
        return loss.item()

