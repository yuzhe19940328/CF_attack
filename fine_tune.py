import argparse
import os
import random

import numpy as np
import torch
import torchvision
import torch.utils.data

from model import MNIST_model, CelebA_model, DeepFashion_model
from data_loader import *



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='CelebA Classification')
    parser.add_argument('--dataset', type=str, default='celeba', help='kind of dataset & model ')
    parser.add_argument('--batch_size', type=int, default=32, help='input batch size for training (default: 64)')
    parser.add_argument('--epochs', type=int, default=2, help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.01, help='learning rate (default: 0.01)')
    parser.add_argument('--image_size', type=int, default=224, help='image size (default: 32)')
    parser.add_argument('--n_classes', type=int, default=10, help='number of classes (default: 10)')
    parser.add_argument('--n_tasks', type=int, default=5, help='how many tasks in this dataset')
    parser.add_argument('--target_task', type=int, default=0, help='target task we want to attack')


    parser.add_argument('--seed', type=int, default=0, help='random seed (default: 0)')
    parser.add_argument('--load_dir', type=str, default='checkpoint_all', help='directory to load checkpoints')

    parser.add_argument('--save_dir', type=str, default='fine_tune_checkpoint', help='directory to save checkpoints')
    args = parser.parse_args()


    args.input_shape = (3, args.image_size, args.image_size)

    # Set seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    # Load data

    if args.dataset == 'celeba':
        train_loader = get_train_loader_celeba(args)
        test_loader = get_test_loader_celeba(args)
    elif args.dataset == 'deepfashion':
        train_loader = get_train_loader_deepfashion(args)
        test_loader = get_test_loader_deepfashion(args)
    else:
        raise ValueError('Invalid dataset')


    # Load model
    if(args.dataset == 'celeba'):
        model = CelebA_model(args)
    elif (args.dataset == 'DeepFashion'):
        model = DeepFashion_model(args)


    load_path=os.path.join(args.load_dir, f'epoch_{19}.pt')

    model.load(load_path)



    model.cuda()
    model.train()

    # Train
    for epoch in range(args.epochs):
        for i, (x, y) in enumerate(train_loader):
            x,y = x.cuda(),y.cuda()

            for i in range(args.n_tasks):

                if i == args.target_task:
                    continue
                else:
                    temp_y = y[:,i]
                    temp_y = temp_y.cuda()
                    #loss = model.updata_head(x, temp_y, i)
                    loss = model.update_feature_extractor(x, temp_y, i)



        # save checkpoint
        save_dir=os.path.join(args.save_dir, f'target_task_{args.target_task}')
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)


        checkpoint_path = os.path.join(save_dir, f'epoch_{epoch}.pt')
        model.save( checkpoint_path)
    
    print('Finetune is done!')
    



