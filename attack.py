import torch
from torch.autograd import Variable
import torchvision
from model import MNIST_model, CelebA_model, DeepFashion_model
from data_loader import *
import argparse
import os
import random




def attack(model, fine_tuned_model, x, epsilon, criterion):
    torch.cuda.empty_cache()

    x_adv = x.clone().detach().requires_grad_(True)
    perturbed_img=Variable(x_adv, requires_grad=True)


    for i in range(10):
        original_feature=model.get_feature(perturbed_img)

        desired_feature=fine_tuned_model.get_feature(x)

        loss = criterion(original_feature, desired_feature)

        temp_grad=torch.autograd.grad(loss, perturbed_img, create_graph=True)[0]
        eta = epsilon * temp_grad.sign()*(-1)
        perturbed_img = Variable(perturbed_img + eta,requires_grad=True)

        eta = torch.clamp(perturbed_img - x, min=-epsilon, max=epsilon)
        perturbed_img = Variable(x + eta,requires_grad=True)
        perturbed_img = torch.clamp(perturbed_img, 0, 1)

    return perturbed_img


def generate_adv_examples(model, fine_tuned_model, test_loader, epsilon, criterion):
    #generate 1000 adversarial examples, and save as PNG images
    model.eval()
    fine_tuned_model.eval()
    torch.cuda.empty_cache()

    idx=0
    for i, (x, y) in enumerate(test_loader):
        x, y = x.cuda(), y.cuda()
        x_adv = attack(model, fine_tuned_model, x, epsilon, criterion)
        for j in range(x_adv.size(0)): 
            torchvision.utils.save_image(x_adv[j], f'adv_examples/{idx}.png')
            #save labels
            temp_np=y[j].cpu().numpy()
            print(temp_np)
            with open('adv_examples/labels.txt', 'a') as f:
                label_str=str(temp_np)
                f.write(label_str + '\n')
            idx+=1
        if idx>=1000:
            break
    return


if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Generate AEs')
    parser.add_argument('--dataset', type=str, default='celeba', help='kind of dataset & model (default: MNIST)')
    parser.add_argument('--image_size', type=int, default=224, help='image size (default: 32)')
    parser.add_argument('--n_classes', type=int, default=2, help='number of classes (default: 10)')
    parser.add_argument('--seed', type=int, default=0, help='random seed (default: 0)')
    parser.add_argument('--lr', type=float, default=0.01, help='learning rate (default: 0.01)')
    parser.add_argument('--n_tasks', type=int, default=5, help='how many tasks in this dataset')
    parser.add_argument('--epsilon', type=float, default=0.1, help='size of adversarial examples')
    parser.add_argument('--batch_size', type=int, default=16, help='batch size')

    parser.add_argument('--save_dir', type=str, default='checkpoint_all', help='directory to save checkpoints')
    args = parser.parse_args()

    args.input_shape = (3, args.image_size, args.image_size)
    # Load data
    test_loader = get_test_loader_celeba(args)

    model = CelebA_model(args)

    fine_tuned_model = CelebA_model(args)

    model_path=os.path.join(args.save_dir, f'epoch_19.pt')
    model.load(model_path)

    fine_tuned_model_path=os.path.join('fine_tune_checkpoint', 'target_task_0', f'epoch_0.pt')
    fine_tuned_model.load(fine_tuned_model_path)

    model.cuda()
    fine_tuned_model.cuda()

    criterion = torch.nn.MSELoss()
    epsilon=args.epsilon   # size of adversarial examples
    generate_adv_examples(model, fine_tuned_model, test_loader, epsilon, criterion)














