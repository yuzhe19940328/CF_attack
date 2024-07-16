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




def measure_multi_task_accuracy(model,test_loader,args):
    model.eval()
    correct={}
    with torch.no_grad():
        for x,y in test_loader:
            x,y = x.cuda(),y.cuda()
            for i in range(args.n_tasks):
                temp_y = y[:,i]
                output=model(x,i)
                pred=output.argmax(dim=1,keepdim=True)
                if i not in correct:
                    correct[i]=0
                correct[i] += pred.eq(temp_y.view_as(pred)).sum().item()
    #print the accuracy of each task
    for i in range(args.n_tasks):
        print('task {} accuracy is {}'.format(i,100.*correct[i]/(len(test_loader.dataset))))



    return 