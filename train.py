import argparse
import os
import random

import numpy as np
import torch
import torchvision
import torch.utils.data

from model import MNIST_model, CelebA_model, DeepFashion_model
from data_loader import get_train_loader



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='MNIST Classification')
    parser.add_argument('--dataset_kind', type=str, default='MNIST', help='kind of dataset & model (default: MNIST)')
    parser.add_argument('--batch_size', type=int, default=64, help='input batch size for training (default: 64)')
    parser.add_argument('--epochs', type=int, default=10, help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.01, help='learning rate (default: 0.01)')
    parser.add_argument('--image_size', type=int, default=32, help='image size (default: 32)')
    parser.add_argument('--n_classes', type=int, default=10, help='number of classes (default: 10)')
    parser.add_argument('--seed', type=int, default=0, help='random seed (default: 0)')
    parser.add_argument('--save_dir', type=str, default='checkpoint', help='directory to save checkpoints')
    parser.add_argument('--resnet18', type=bool, default=True, help='Whether to use ResNet18. False means to use ResNet50 (default: True)')
    parser.add_argument('--resnet_dropout', type=float, default=0.5, help=' (ResNet drop-out rate (default : 0.5)')
    args = parser.parse_args()


    args.input_shape = (1, args.image_size, args.image_size)
    args.resnet_hparam = {
        'resnet18': args.resnet18,
        'resnet_dropout': args.resnet_dropout
    }
    # Set seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    # Load data
    train_loader = get_train_loader(args)

    # Load model
    if(args.dataset_kind != 'MNIST'):
        args.save_dir += '_' + args.dataset_kind
    if(args.dataset_kind == 'CelebA'):
        model = CelebA_model(args)
    elif (args.dataset_kind == 'DeepFashion'):
        model = DeepFashion_model(args)
    else:
        model = MNIST_model(args)
    model.cuda()
    model.train()

    # Train
    for epoch in range(args.epochs):
        for i, (x, y) in enumerate(train_loader):
            x,y = x.cuda(),y.cuda()
            loss = model.update(x, y)

        # save checkpoint
        if not os.path.exists(args.save_dir):
            os.makedirs(args.save_dir)
        checkpoint_path = os.path.join(args.save_dir, f'epoch_{epoch}.pt')
        torch.save(model.state_dict(), checkpoint_path)
    
    print('Training is done!')
    



