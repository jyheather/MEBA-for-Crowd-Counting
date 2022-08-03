# Best model
import torch
import os
import numpy as np
from datasets.crowd_sh import Crowd
from models.vgg import vgg19, Stack, Adapter1, Adapter2, Adapter3
import argparse

args = None


def parse_args():
    parser = argparse.ArgumentParser(description='Test ')
    parser.add_argument('--data-dir', default='/content/drive/MyDrive/Bayesian-Crowd-Counting-master-shb/shb-Train-Val-Test',
                        help='training data directory')
    parser.add_argument('--save-dir', default='/content/drive/MyDrive/Bayesian-Crowd-Counting-master-shb/result/best--0720-055733(bg=1.0, mae=8.0, mse=13.7)',
                        help='model directory')
    parser.add_argument('--device', default='0', help='assign device')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.device.strip()  # set vis gpu

    datasets = Crowd(os.path.join(args.data_dir, 'test'), 512, 8, is_gray=False, method='val')
    dataloader = torch.utils.data.DataLoader(datasets, 1, shuffle=False,
                                             num_workers=8, pin_memory=False)

    device = torch.device('cuda')
    model_dic = torch.load(os.path.join(args.save_dir, 'best_model.pth'), device)
    model_0 = vgg19(flag=0).to(device)
    adapter_0 = Adapter1().to(device)
    model_1 = vgg19(flag=1).to(device)
    adapter_1 = Adapter2().to(device)
    model_2 = vgg19(flag=2).to(device)
    adapter_2 = Adapter3().to(device)
    stack = Stack().to(device)

    model_0.load_state_dict(model_dic[0])
    adapter_0.load_state_dict(model_dic[1])
    model_1.load_state_dict(model_dic[2])
    adapter_1.load_state_dict(model_dic[3])
    model_2.load_state_dict(model_dic[4])
    adapter_2.load_state_dict(model_dic[5])
    stack.load_state_dict(model_dic[6])


    epoch_minus = []
    for inputs, count, name in dataloader:
        inputs = inputs.to(device)
        assert inputs.size(0) == 1, 'the batch size should equal to 1'
        with torch.set_grad_enabled(False):
            outputs_0 = model_0(inputs)
            outputs_0 = adapter_0(outputs_0)
            outputs_1 = model_1(inputs)
            outputs_1 = adapter_1(outputs_1)
            outputs_2 = model_2(inputs)
            outputs_2 = adapter_2(outputs_2)
            answers = torch.cat([outputs_0, outputs_1, outputs_2], dim=1)
            outputs = stack(answers)
            temp_minu = count[0].item() - torch.sum(outputs).item()
            print(name, temp_minu, count[0].item(), torch.sum(outputs).item())
            epoch_minus.append(temp_minu)    


    epoch_minus = np.array(epoch_minus)
    mse = np.sqrt(np.mean(np.square(epoch_minus)))
    mae = np.mean(np.abs(epoch_minus))
    log_str = 'Final Test: mae {}, mse {}'.format(mae, mse)
    print(log_str)






# # Ablation_only adapter
# import torch
# import os
# import numpy as np
# from datasets.crowd_sh import Crowd
# from models.vgg import vgg19, Stack, Adapter1, Adapter2, Adapter3
# import argparse

# args = None


# def parse_args():
#     parser = argparse.ArgumentParser(description='Test ')
#     parser.add_argument('--data-dir', default='/content/drive/MyDrive/Bayesian-Crowd-Counting-master-shb/shb-Train-Val-Test',
#                         help='training data directory')
#     parser.add_argument('--save-dir', default='/content/drive/MyDrive/Bayesian-Crowd-Counting-master-shb/result/0720-055733(best model)',
#                         help='model directory')
#     parser.add_argument('--device', default='0', help='assign device')
#     args = parser.parse_args()
#     return args

# if __name__ == '__main__':
#     args = parse_args()
#     os.environ['CUDA_VISIBLE_DEVICES'] = args.device.strip()  # set vis gpu

#     datasets = Crowd(os.path.join(args.data_dir, 'test'), 512, 8, is_gray=False, method='val')
#     dataloader = torch.utils.data.DataLoader(datasets, 1, shuffle=False,
#                                              num_workers=8, pin_memory=False)

#     device = torch.device('cuda')
#     model_dic = torch.load(os.path.join(args.save_dir, 'best_model.pth'), device)
#     model_0 = vgg19().to(device)
#     adapter_0 = Adapter1().to(device)
#     model_1 = vgg19().to(device)
#     adapter_1 = Adapter2().to(device)
#     model_2 = vgg19().to(device)
#     adapter_2 = Adapter3().to(device)
#     stack = Stack().to(device)

#     model_0.load_state_dict(model_dic[0])
#     adapter_0.load_state_dict(model_dic[1])
#     model_1.load_state_dict(model_dic[2])
#     adapter_1.load_state_dict(model_dic[3])
#     model_2.load_state_dict(model_dic[4])
#     adapter_2.load_state_dict(model_dic[5])
#     stack.load_state_dict(model_dic[6])


#     epoch_minus = []
#     for inputs, count, name in dataloader:
#         inputs = inputs.to(device)
#         assert inputs.size(0) == 1, 'the batch size should equal to 1'
#         with torch.set_grad_enabled(False):
#             outputs_0 = model_0(inputs)
#             outputs_0 = adapter_0(outputs_0)
#             outputs_1 = model_1(inputs)
#             outputs_1 = adapter_1(outputs_1)
#             outputs_2 = model_2(inputs)
#             outputs_2 = adapter_2(outputs_2)
#             answers = torch.cat([outputs_0, outputs_1, outputs_2], dim=1)
#             outputs = stack(answers)
#             temp_minu = count[0].item() - torch.sum(outputs).item()
#             print(name, temp_minu, count[0].item(), torch.sum(outputs).item())
#             epoch_minus.append(temp_minu)    


#     epoch_minus = np.array(epoch_minus)
#     mse = np.sqrt(np.mean(np.square(epoch_minus)))
#     mae = np.mean(np.abs(epoch_minus))
#     log_str = 'Final Test: mae {}, mse {}'.format(mae, mse)
#     print(log_str)



# # Ablation_only decoder
# import torch
# import os
# import numpy as np
# from datasets.crowd_sh import Crowd
# from models.vgg import vgg19, Stack, Adapter1, Adapter2, Adapter3
# import argparse

# args = None


# def parse_args():
#     parser = argparse.ArgumentParser(description='Test ')
#     parser.add_argument('--data-dir', default='/content/drive/MyDrive/Bayesian-Crowd-Counting-master-shb/shb-Train-Val-Test',
#                         help='training data directory')
#     parser.add_argument('--save-dir', default='/content/drive/MyDrive/Bayesian-Crowd-Counting-master-shb/ablation/0720-165724',
#                         help='model directory')
#     parser.add_argument('--device', default='0', help='assign device')
#     args = parser.parse_args()
#     return args

# if __name__ == '__main__':
#     args = parse_args()
#     os.environ['CUDA_VISIBLE_DEVICES'] = args.device.strip()  # set vis gpu

#     datasets = Crowd(os.path.join(args.data_dir, 'test'), 512, 8, is_gray=False, method='val')
#     dataloader = torch.utils.data.DataLoader(datasets, 1, shuffle=False,
#                                              num_workers=8, pin_memory=False)

#     device = torch.device('cuda')
#     model_dic = torch.load(os.path.join(args.save_dir, 'best_model.pth'), device)
#     model_0 = vgg19(flag=0).to(device)
#     model_1 = vgg19(flag=1).to(device)
#     model_2 = vgg19(flag=2).to(device)
#     stack = Stack().to(device)

#     model_0.load_state_dict(model_dic[0])
#     model_1.load_state_dict(model_dic[1])
#     model_2.load_state_dict(model_dic[2])
#     stack.load_state_dict(model_dic[3])


#     epoch_minus = []
#     for inputs, count, name in dataloader:
#         inputs = inputs.to(device)
#         assert inputs.size(0) == 1, 'the batch size should equal to 1'
#         with torch.set_grad_enabled(False):
#             outputs_0 = model_0(inputs)
#             outputs_1 = model_1(inputs)
#             outputs_2 = model_2(inputs)
#             answers = torch.cat([outputs_0, outputs_1, outputs_2], dim=1)
#             outputs = stack(answers)
#             temp_minu = count[0].item() - torch.sum(outputs).item()
#             print(name, temp_minu, count[0].item(), torch.sum(outputs).item())
#             epoch_minus.append(temp_minu)    


#     epoch_minus = np.array(epoch_minus)
#     mse = np.sqrt(np.mean(np.square(epoch_minus)))
#     mae = np.mean(np.abs(epoch_minus))
#     log_str = 'Final Test: mae {}, mse {}'.format(mae, mse)
#     print(log_str)





# # baseline
# import torch
# import os
# import numpy as np
# from datasets.crowd_sh import Crowd
# from models.vgg import vgg19
# import argparse

# args = None


# def parse_args():
#     parser = argparse.ArgumentParser(description='Test ')
#     parser.add_argument('--data-dir', default='/content/drive/MyDrive/Bayesian-Crowd-Counting-master-shb/shb-Train-Val-Test',
#                         help='training data directory')
#     parser.add_argument('--save-dir', default='/content/drive/MyDrive/Bayesian-Crowd-Counting-master-shb/baseline(val=test)/0712-083141',
#                         help='model directory')
#     parser.add_argument('--device', default='0', help='assign device')
#     args = parser.parse_args()
#     return args


# if __name__ == '__main__':
#     args = parse_args()
#     os.environ['CUDA_VISIBLE_DEVICES'] = args.device.strip()  # set vis gpu

#     datasets = Crowd(os.path.join(args.data_dir, 'test'), 512, 8, is_gray=False, method='val')
#     dataloader = torch.utils.data.DataLoader(datasets, 1, shuffle=False,
#                                              num_workers=8, pin_memory=False)
#     model = vgg19()
#     device = torch.device('cuda')
#     model.to(device)
#     model.load_state_dict(torch.load(os.path.join(args.save_dir, 'best_model.pth'), device))
#     epoch_minus = []

#     for inputs, count, name in dataloader:
#         inputs = inputs.to(device)
#         assert inputs.size(0) == 1, 'the batch size should equal to 1'
#         with torch.set_grad_enabled(False):
#             outputs = model(inputs)
#             temp_minu = count[0].item() - torch.sum(outputs).item()
#             print(name, temp_minu, count[0].item(), torch.sum(outputs).item())
#             epoch_minus.append(temp_minu)

#     epoch_minus = np.array(epoch_minus)
#     mse = np.sqrt(np.mean(np.square(epoch_minus)))
#     mae = np.mean(np.abs(epoch_minus))
#     log_str = 'Final Test: mae {}, mse {}'.format(mae, mse)
#     print(log_str)




