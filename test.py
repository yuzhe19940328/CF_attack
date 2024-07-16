#test checkpoint performance
import os
import numpy as np
import torch
import torchvision
import torch.utils.data
import argparse
from model import MNIST_model, CelebA_model, DeepFashion_model
from data_loader import get_test_loader_celeba
from utils import measure_multi_task_accuracy


if __name__=='__main__':
    parser = argparse.ArgumentParser(description='MNIST Classification')
    parser.add_argument('--dataset', type=str, default='celeba', help='kind of dataset & model (default: MNIST)')
    parser.add_argument('--image_size', type=int, default=224, help='image size (default: 32)')
    parser.add_argument('--n_classes', type=int, default=2, help='number of classes (default: 10)')
    parser.add_argument('--seed', type=int, default=0, help='random seed (default: 0)')
    parser.add_argument('--lr', type=float, default=0.01, help='learning rate (default: 0.01)')
    parser.add_argument('--n_tasks', type=int, default=5, help='how many tasks in this dataset')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size')

    parser.add_argument('--save_dir', type=str, default='fine_tune_checkpoint/target_task_0', help='directory to save checkpoints')
    args = parser.parse_args()

    # For MNIST set channel as 1, for otehr dataset, set channel as 3
    args.input_shape = (3, args.image_size, args.image_size)

    # Load data
    test_loader = get_test_loader_celeba(args)
    
    # Load model
    if(args.dataset == 'celeba'):
        model = CelebA_model(args)
    elif (args.dataset == 'DeepFashion'):
        model = DeepFashion_model(args)

    model.cuda()
    model.eval()
    
    # Load checkpoint
    checkpoint_path = os.path.join(args.save_dir, f'epoch_0.pt')
    model.load(checkpoint_path)

    # Test
    measure_multi_task_accuracy(model,test_loader,args)

