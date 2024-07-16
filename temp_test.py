import os
import numpy as np
import torch
import torchvision
import torch.utils.data
import argparse
from model import MNIST_model, CelebA_model, DeepFashion_model
from data_loader import get_test_loader
from utils import measure_multi_task_accuracy



if __name__=='__main__':
    parser = argparse.ArgumentParser(description='MNIST Classification')
    parser.add_argument('--dataset_kind', type=str, default='MNIST', help='kind of dataset & model (default: MNIST)')
    parser.add_argument('--image_size', type=int, default=32, help='image size (default: 32)')
    parser.add_argument('--n_classes', type=int, default=10, help='number of classes (default: 10)')
    parser.add_argument('--seed', type=int, default=0, help='random seed (default: 0)')
    parser.add_argument('--lr', type=float, default=0.01, help='learning rate (default: 0.01)')
    parser.add_argument('--n_tasks', type=int, default=5, help='how many tasks in this dataset')

    parser.add_argument('--save_dir', type=str, default='checkpoint', help='directory to save checkpoints')
    args = parser.parse_args()

    args.input_shape = (3, args.image_size, args.image_size)
    model = CelebA_model(args)
    model_file_path="./checkpoint/epoch_19.pt"

    model.cuda()
    model.eval()

    checkpoint=torch.load(model_file_path)
    for key in checkpoint.keys():
        print(key) 








