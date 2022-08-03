# import torch
# import os
# import numpy as np
# from datasets.crowd_sh import Crowd
# from models.vgg import vgg19
# import argparse
# import matplotlib.pyplot as plt

# args = None


# def parse_args():
#     parser = argparse.ArgumentParser(description='Test ')
#     parser.add_argument('--data-dir', default='/content/drive/MyDrive/Bayesian-Crowd-Counting-master-sha/sha-Train-Val-Test',
#                         help='training data directory')
#     parser.add_argument('--save-dir', default='/content/drive/MyDrive/Bayesian-Crowd-Counting-master-sha/baseline(val=test)/baseline--0711-065147(mae=69.9, mse=115.6)',
#                         help='model directory')
#     parser.add_argument('--device', default='0', help='assign device')
#     args = parser.parse_args()
#     return args


# if __name__ == '__main__':
#     args = parse_args()
#     os.environ['CUDA_VISIBLE_DEVICES'] = args.device.strip()  # set vis gpu

#     datasets = Crowd(os.path.join(args.data_dir, 'plot_173'), 512, 8, is_gray=False, method='val')
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

#             fig = plt.figure()
#             temp = outputs[0, 0, :, :].cpu().numpy()
#             temp_mean = np.mean(temp)
#             temp_var = np.std(temp)
#             print(temp_mean, temp_var)
#             temp_0 = (temp - temp_mean) / temp_var
#             plt.imshow(temp, cmap=plt.cm.jet)
#             plt.show()
#             fig.savefig("173_baseline.png")

#             temp_minu = count[0].item() - torch.sum(outputs).item()
#             print(name, temp_minu, count[0].item(), torch.sum(outputs).item())
#             epoch_minus.append(temp_minu)

#     epoch_minus = np.array(epoch_minus)
#     mse = np.sqrt(np.mean(np.square(epoch_minus)))
#     mae = np.mean(np.abs(epoch_minus))
#     log_str = 'Final Test: mae {}, mse {}'.format(mae, mse)
#     print(log_str)


# # Visualization (Best model)
# from re import A
# import torch
# import os
# import numpy as np
# from datasets.crowd_sh import Crowd
# from models.vgg import vgg19, Stack, Adapter1, Adapter2, Adapter3
# import argparse
# import matplotlib.pyplot as plt

# args = None


# def parse_args():
#     parser = argparse.ArgumentParser(description='Test ')
#     parser.add_argument('--data-dir', default='/content/drive/MyDrive/Bayesian-Crowd-Counting-master-sha/sha-Train-Val-Test',
#                         help='training data directory')
#     parser.add_argument('--save-dir', default='/content/drive/MyDrive/Bayesian-Crowd-Counting-master-sha/result/best--0716-151840 (mae 67.1, mse 109.2)',
#                         help='model directory')
#     parser.add_argument('--device', default='0', help='assign device')
#     args = parser.parse_args()
#     return args

# if __name__ == '__main__':
#     args = parse_args()
#     os.environ['CUDA_VISIBLE_DEVICES'] = args.device.strip()  # set vis gpu

#     datasets = Crowd(os.path.join(args.data_dir, 'plot_173'), 512, 8, is_gray=False, method='val')
#     dataloader = torch.utils.data.DataLoader(datasets, 1, shuffle=False,
#                                              num_workers=8, pin_memory=False)
    

#     device = torch.device('cuda')
#     model_dic = torch.load(os.path.join(args.save_dir, 'best_model.pth'), device)
#     model_0 = vgg19(flag=0).to(device)
#     adapter_0 = Adapter1().to(device)
#     model_1 = vgg19(flag=1).to(device)
#     adapter_1 = Adapter2().to(device)
#     model_2 = vgg19(flag=2).to(device)
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


#             fig = plt.figure()
#             temp_0 = outputs_0[0, 0, :, :].cpu().numpy()
#             temp_0_mean = np.mean(temp_0)
#             temp_0_var = np.std(temp_0)
#             print(temp_0_mean, temp_0_var)
#             temp_0 = (temp_0 - temp_0_mean) / temp_0_var
#             plt.imshow(temp_0, cmap=plt.cm.jet)
#             plt.show()
#             # fig.savefig("157_outputs_0.png")
#             fig.savefig("173_outputs_0.png")

#             fig.clear()
#             fig = plt.figure()
#             temp_1 = outputs_1[0, 0, :, :].cpu().numpy()
#             temp_1_mean = np.mean(temp_1)
#             temp_1_var = np.std(temp_1)
#             print(temp_1_mean, temp_1_var)
#             temp_1 = (temp_1 - temp_1_mean) / temp_1_var
#             plt.imshow(temp_1, cmap=plt.cm.jet)
#             plt.show()
#             # fig.savefig("157_outputs_1.png")
#             fig.savefig("173_outputs_1.png")

#             fig.clear()
#             fig = plt.figure()
#             temp_2 = outputs_2[0, 0, :, :].cpu().numpy()
#             temp_2_mean = np.mean(temp_2)
#             temp_2_var = np.std(temp_2)
#             print(temp_2_mean, temp_2_var)
#             temp_2 = (temp_2 - temp_2_mean) / temp_2_var
#             plt.imshow(temp_2, cmap=plt.cm.jet)
#             plt.show()
#             # fig.savefig("157_outputs_2.png")
#             fig.savefig("173_outputs_2.png")


#             answers = torch.cat([outputs_0, outputs_1, outputs_2], dim=1)
#             outputs = stack(answers)

#             fig.clear()
#             fig = plt.figure()
#             temp_s = outputs[0, 0, :, :].cpu().numpy()
#             temp_s_mean = np.mean(temp_s)
#             temp_s_var = np.std(temp_s)
#             print(temp_s_mean, temp_s_var)
#             temp_s = (temp_s - temp_s_mean) / temp_s_var
#             plt.imshow(temp_s, cmap=plt.cm.jet)
#             plt.show()
#             # fig.savefig("157_stack.png")
#             fig.savefig("173_stack.png")


#             temp_minu = count[0].item() - torch.sum(outputs).item()
#             print(name, temp_minu, count[0].item(), torch.sum(outputs).item())
#             epoch_minus.append(temp_minu)    


#     epoch_minus = np.array(epoch_minus)
#     mse = np.sqrt(np.mean(np.square(epoch_minus)))
#     mae = np.mean(np.abs(epoch_minus))
#     log_str = 'Final Test: mae {}, mse {}'.format(mae, mse)
#     print(log_str)


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
    parser.add_argument('--data-dir', default='/content/drive/MyDrive/Bayesian-Crowd-Counting-master-sha/sha-Train-Val-Test',
                        help='training data directory')
    parser.add_argument('--save-dir', default='/content/drive/MyDrive/Bayesian-Crowd-Counting-master-sha/result/best--0716-151840 (mae 67.1, mse 109.2)',
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
#     parser.add_argument('--data-dir', default='/content/drive/MyDrive/Bayesian-Crowd-Counting-master-sha/sha-Train-Val-Test',
#                         help='training data directory')
#     parser.add_argument('--save-dir', default='/content/drive/MyDrive/Bayesian-Crowd-Counting-master-shb/ablation',
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
#     parser.add_argument('--data-dir', default='/content/drive/MyDrive/Bayesian-Crowd-Counting-master-sha/sha-Train-Val-Test',
#                         help='training data directory')
#     parser.add_argument('--save-dir', default='/content/drive/MyDrive/Bayesian-Crowd-Counting-master-sha/ablation',
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
#     parser.add_argument('--data-dir', default='/content/drive/MyDrive/Bayesian-Crowd-Counting-master-sha/sha-Train-Val-Test',
#                         help='training data directory')
#     parser.add_argument('--save-dir', default='/content/drive/MyDrive/Bayesian-Crowd-Counting-master-sha/baseline(val=test)',
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

