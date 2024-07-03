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
