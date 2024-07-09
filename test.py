#test checkpoint performance
import os
import numpy as np
import torch
import torchvision
import torch.utils.data
import argparse
from model import MNIST_model, CelebA_model, DeepFashion_model
from data_loader import get_test_loader
from utils import measure_test_accuracy
if __name__=='__main__':
    parser = argparse.ArgumentParser(description='MNIST Classification')
    parser.add_argument('--dataset_kind', type=str, default='MNIST', help='kind of dataset & model (default: MNIST)')
    parser.add_argument('--image_size', type=int, default=32, help='image size (default: 32)')
    parser.add_argument('--n_classes', type=int, default=10, help='number of classes (default: 10)')
    parser.add_argument('--seed', type=int, default=0, help='random seed (default: 0)')
    parser.add_argument('--lr', type=float, default=0.01, help='learning rate (default: 0.01)')

    parser.add_argument('--save_dir', type=str, default='checkpoint', help='directory to save checkpoints')
    args = parser.parse_args()

    # For MNIST set channel as 1, for otehr dataset, set channel as 3
    args.input_shape = (1, args.image_size, args.image_size)

    # Load data
    test_loader = get_test_loader(args)
    
    # Load model
    if(args.dataset_kind == 'CelebA'):
        model = CelebA_model(args)
    elif (args.dataset_kind == 'DeepFashion'):
        model = DeepFashion_model(args)
    else:
        model = MNIST_model(args)
    model.cuda()
    model.train()
    model.cuda()
    model.eval()
    
    # Load checkpoint
    for epoch in range(10):
        checkpoint_path = os.path.join(args.save_dir, f'epoch_{epoch}.pt')
        model.load_state_dict(torch.load(checkpoint_path))
    
        # Test
        measure_test_accuracy(model,test_loader)
    
