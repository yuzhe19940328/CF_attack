import torch
import torch.utils.data


def measure_test_accuracy(model,test_loader):
    model.eval()
    correct=0
    with torch.no_grad():
        for x,y in test_loader:
            x,y = x.cuda(),y.cuda()
            output=model(x)
            pred=output.argmax(dim=1,keepdim=True)
            correct += pred.eq(y.view_as(pred)).sum().item()
    print('test accuracy is {}'.format(100.*correct/len(test_loader.dataset)))
    return correct / len(test_loader.dataset)