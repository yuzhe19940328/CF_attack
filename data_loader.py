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


# Get attribute of Male, Heavy Makeup, Wearing Lipstick, Smiling, Black Hair
def celebA_transform(target):
        return target[[20, 21, 31, 39, 9]]

def get_train_loader_celeba(args):

    """Builds and returns Dataloader for CelebA dataset."""
    
    transform = transforms.Compose([
                    transforms.Resize((args.image_size, args.image_size)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
                ])

    celebA = datasets.CelebA(root='./celeba', split='train', download=True, transform=transform)


    celebA.target_transform = celebA_transform

    loader = data.DataLoader(dataset=celebA,
                             batch_size=args.batch_size,
                             shuffle=True,
                             num_workers=1)
    return loader

def get_test_loader_celeba(args):

    """Builds and returns Dataloader for CelebA dataset."""
    
    transform = transforms.Compose([
                    transforms.Resize((args.image_size, args.image_size)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
                ])

    celebA = datasets.CelebA(root='./celeba', split='test', download=True, transform=transform)

    celebA.target_transform = celebA_transform

    loader = data.DataLoader(dataset=celebA,
                             batch_size=1000,
                             shuffle=False,
                             num_workers=1)
    return loader


def deepfashion_transform(target):
    category = target['category']  # カテゴリー
    sleeve = target['sleeve_length']  # 袖の長さ
    neckline = target['neckline']  # ネックライン
    return category, sleeve, neckline



def get_train_loader_deepfashion(args):

    """Builds and returns Dataloader for DeepFashion dataset."""
    
    transform = transforms.Compose([
                    transforms.Resize((args.image_size, args.image_size)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
                ])

    deepfashion_train = datasets.ImageFolder(root='./deepfashion/train', transform=transform)


    deepfashion_train.target_transform = deepfashion_transform

    loader = data.DataLoader(dataset=deepfashion_train,
                             batch_size=args.batch_size,
                             shuffle=True,
                             num_workers=1)
    return loader

def get_test_loader_deepfashion(args):

    """Builds and returns Dataloader for DeepFashion dataset."""
    
    transform = transforms.Compose([
                    transforms.Resize((args.image_size, args.image_size)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
                ])

    deepfashion_test = datasets.ImageFolder(root='./deepfashion/test', transform=transform)



    deepfashion_test.target_transform = deepfashion_transform

    loader = data.DataLoader(dataset=deepfashion_test,
                             batch_size=1000,
                             shuffle=False,
                             num_workers=1)
    return loader
