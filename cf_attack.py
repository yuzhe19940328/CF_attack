import io
from torch.autograd import Variable
from torch import nn
import torch
# from torchvision.models.feature_extraction import create_feature_extractor
import torchvision
import numpy as np

import torch.nn.functional as F

from tqdm import tqdm
def CF_attack_mtask(x, y, mask, model_attacking, model_targeting, criterion, task_name, epsilon, steps, step_size, args, comet=None,
                     batch_index = None,strategy=None,using_noise=True,norm = 'Linf' ,encoder_names=None,comet_suffix=[""],layer_grad_each_encoder_dict=None,dr=True):
    # if args.arch == "resnet-18":
    if args.arch == "":
        if encoder_names is None:
            names = torchvision.models.feature_extraction.get_graph_node_names(model_attacking)
            # print(names)
            encoder_all_names = []
            for name in names[0]:
                if "encoder.layer" in name and "conv" in name :
                    encoder_all_names.append(name)

            if attack_layer_idx is None:
                attack_layer_idx = list(range(encoder_all_names))

            if type(attack_layer_idx) is not list:
                attack_layer_idx = [attack_layer_idx]

            encoder_names = []
            for idx in attack_layer_idx:
                encoder_names.append(encoder_all_names[idx])

        extractor = torchvision.models.feature_extraction.create_feature_extractor(
                        model_attacking, encoder_names
                    )
    else:

        def extract(model_attacking,target, inputs):
            feature = None
            

            def forward_hook(module, inputs, outputs):
                # 順伝搬の出力を features というグローバル変数に記録する
                global features__
                # 1. detach でグラフから切り離す。
                # 2. clone() でテンソルを複製する。モデルのレイヤーで ReLU(inplace=True) のように
                #    inplace で行う層があると、値がその後のレイヤーで書き換えられてまい、
                #    指定した層の出力が取得できない可能性があるため、clone() が必要。
                
                # features__ = outputs.clone()
                features__ = outputs
                features__.retain_grad()
                # print(features__.is_leaf)
                
                # features__.requires_grad = True

            # コールバック関数を登録する。
            handle = target.register_forward_hook(forward_hook)

            # 推論する
            model_attacking.eval()
            # output = model_attacking(inputs)
            output = model_attacking.encoder(inputs)
            # if args.arch == "xception-full":
            #     output = model_attacking.final_conv(output)
            #     output = model_attacking.final_conv_bn(output)

            # コールバック関数を解除する。
            handle.remove()

            return features__, output
        

    x = x.cuda()

    if model_targeting is not None:
        model_targeting.eval()
    model_attacking.eval()
    

    # if batch_index % batch_log ==0:
    #     print("attacking batch {batch_index} attacking {task_name} targeting {target_task_name} with strategy {strategy}".format(batch_index=batch_index, task_name=task_name, target_task_name=args.target_task_set,strategy=strategy))
    # tensor_std = get_torch_std(info)
    if epsilon == 0:
        return Variable(x, requires_grad=False)

    GPU_flag = False
    if torch.cuda.is_available():
        GPU_flag=True

    if None: #args.dataset == "cifar10":
        rescale_term = 1
    else:
        rescale_term = 1./255
    epsilon = epsilon * rescale_term
    step_size = step_size * rescale_term

    # print('epsilon', epsilon, epsilon / rescale_term)
    GRID_WEIGTHS = [10**i for i in range(-9,9)]
    weights = {k:[] for k in task_name}

    search_iters = np.power(len(GRID_WEIGTHS), len(task_name)) if strategy == "GRID_SEARCH" else 1
    # best_advs = {"score": -np.inf, "adv": None, "metrics":[]}
    # strtg = strategy.split("_")

    # best_advs["metric"] = task_name[int(strtg[-1])] if len(strtg)>2 else task_name[0]
    task_metrics = {}
    x_advs = []
    # for k in tqdm(range(len(encoder_names))):
    for k in range(len(encoder_names)):

        # print("attacking :",encoder_names[k])
        for j in range(search_iters):
            x_adv = x.clone()

            pert_upper = x_adv + epsilon
            pert_lower = x_adv - epsilon

            upper_bound = torch.ones_like(x_adv)
            lower_bound = -torch.ones_like(x_adv)

            upper_bound = torch.min(upper_bound, pert_upper)
            lower_bound = torch.max(lower_bound, pert_lower)

            if args.debug:
                print(x_adv.max())
                print(x_adv.min())
                print(upper_bound.max())
                print(lower_bound.max())



            ones_x = torch.ones_like(x).float()
            if GPU_flag:

                x_adv = x_adv.cuda()
                upper_bound = upper_bound.cuda()
                lower_bound = lower_bound.cuda()
                # for keys, m in mask.items():
                #     mask[keys] = m.cuda()
                # for keys, tar in y.items():
                #     y[keys] = tar.cuda()



          
            target_module = model_attacking
                    # 抽出対象の層
            for attr in encoder_names[k].split("."):
                # print(attr)
                if "[" in attr:
                    attr, attr_idx = attr.split("[")
                    attr_idx = int(attr_idx.replace("]",""))
                    target_module = getattr(target_module,attr)[int(attr_idx)]
                else:
                    target_module = getattr(target_module,attr)

            feature_forgetting, _ = extract(model_attacking,target_module,x_adv)
            feature_forgetting = feature_forgetting.clone().detach()

            if not y[list(y.keys())[0]].dim() > 2 and k == len(encoder_names) - 1:
                    feature_forgetting = F.adaptive_avg_pool2d(feature_forgetting, (1, 1))
                    feature_forgetting = feature_forgetting.view(feature_forgetting.size(0), -1)

            # target_module = model_targeting
            #         # 抽出対象の層
            # for attr in encoder_names[k].split("."):
            #     # print(attr)
            #     if "[" in attr:
            #         attr, attr_idx = attr.split("[")
            #         attr_idx = int(attr_idx.replace("]",""))
            #         target_module = getattr(target_module,attr)[int(attr_idx)]
            #     else:
            #         target_module = getattr(target_module,attr)

            # feature_forgetting, _ = extract(model_targeting,target_module,x_adv)
            # feature_forgetting = feature_forgetting.detach().clone()

            # if not y[list(y.keys())[0]].dim() > 2 and k == len(encoder_names) - 1:
            #     feature_forgetting = F.adaptive_avg_pool2d(feature_forgetting, (1, 1))
            #     feature_forgetting = feature_forgetting.view(feature_forgetting.size(0), -1)

            #noiseを加えないと、step=0のlossが0となってしまう
            # if using_noise:
            # noise = torch.FloatTensor(x.size()).uniform_(-epsilon, epsilon)
            # if GPU_flag:
            #     noise = noise.cuda()
            # x_adv = x_adv + noise
            # x_adv = clamp_tensor(x_adv, lower_bound, upper_bound)

            x_adv = Variable(x_adv, requires_grad=True)
            # del _,target_module
            model_targeting.zero_grad()

            if x_adv.grad is not None:
                x_adv.grad.data.fill_(0)
            for i in range(steps):

                target_module = model_targeting
                    # 抽出対象の層
                for attr in encoder_names[k].split("."):
                    # print(attr)
                    if "[" in attr:
                        attr, attr_idx = attr.split("[")
                        attr_idx = int(attr_idx.replace("]",""))
                        target_module = getattr(target_module,attr)[int(attr_idx)]
                    else:
                        target_module = getattr(target_module,attr)

                feature, _ = extract(model_targeting,target_module,x_adv)
                if not y[list(y.keys())[0]].dim() > 2 and k == len(encoder_names) - 1:
                    feature = F.adaptive_avg_pool2d(feature, (1, 1))
                    feature = feature.view(feature.size(0), -1)
                    # del _

                # for k,feature in enumerate(features):

                
                model_targeting.zero_grad()

                if x_adv.grad is not None:
                    x_adv.grad.data.fill_(0)

                if norm == 'Linf':
                    # print(feature.shape)
                    loss = 0
     
                    loss = - torch.norm(feature - feature_forgetting)
                    # loss = torch.norm(feature - feature_forgetting)


                    loss.backward()

                    x_adv = x_adv + step_size * x_adv.grad.sign()

                    if comet is not None and k == len(encoder_names) - 1:
                        comet.log_metric(f"MSE feature - feature_forgetting {encoder_names[k]}",- loss.item(),step=i)
                        comet.log_metric(f"MAE feature - feature_forgetting {encoder_names[k]}",torch.norm(feature - feature_forgetting,p=1).item(),step=i)


                    
                    #     comet.log_metric("Nom of pixel greater than upper_bound_{}".format(batch_index),torch.where(x_adv > upper_bound, 1, 0).sum(),step=i)
                    #     comet.log_metric("Nom of pixel less than lower_bound_{}".format(batch_index),torch.where(x_adv < lower_bound, 1, 0).sum(),step=i)
                    x_adv = clamp_tensor(x_adv, upper_bound, lower_bound)
                else:

                    loss = -1 * feature.std()
                    loss.backward()
                    x_adv = x_adv + x_adv.grad * epsilon
                    x_delta = x_adv - x.cuda()
                    x_delta_normalized = x_delta / torch.norm(x_delta, 2)

                    x_adv = x.cuda() + x_delta_normalized * epsilon


                x_adv = Variable(x_adv.data, requires_grad=True)  #TODO: optimize, remove this variable init each
                #TODO: volatile option for backward, check later
            x_advs.append(x_adv.cpu().detach())

            # if strategy[:11]!="GRID_SEARCH":
                #Cosine 実験

                # x_adv = x_adv.view(len(encoder_names),x.shape[0],x.shape[1],x.shape[2],x.shape[3])

        # cs, cs_rep = calculate_cs(args, x, y, mask, model_attacking, model_targeting, criterion, encoder_names)

    # return x_advs, task_metrics,cs,cs_rep
    return x_advs, task_metrics,None,None
