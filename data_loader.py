import torch
from torchvision import datasets
from torchvision import transforms
from torch.utils import data
from torch.utils.data import Dataset
import argparse
import pandas as pd
import os



def get_train_loader(args):

    """Builds and returns Dataloader for MNIST and SVHN dataset."""
    
    transform_for_1channel = transforms.Compose([
                    transforms.Resize(args.image_size),
                    transforms.ToTensor(),
                    ])
    mnist = datasets.MNIST(root='./mnist', train=True, download=True, transform=transform_for_1channel)

    loader=data.DataLoader(dataset=mnist,
                                            batch_size=args.batch_size,
                                            shuffle=True,
                                            num_workers=1)
    return loader





def get_test_loader(args):

    """Builds and returns Dataloader for MNIST and SVHN dataset."""
    
    transform_for_1channel = transforms.Compose([
                    transforms.Resize(args.image_size),
                    transforms.ToTensor(),
                    ])
    mnist = datasets.MNIST(root='./mnist', train=False, download=True, transform=transform_for_1channel)

    loader=data.DataLoader(dataset=mnist,
                                            batch_size=1000,
                                            shuffle=False,
                                            num_workers=1)
    return loader
