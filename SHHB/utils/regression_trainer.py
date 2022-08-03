from utils.trainer import Trainer
from utils.helper import Save_Handle, AverageMeter
import os
import sys
import time
import torch
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate
import logging
import numpy as np
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from  models.vgg import vgg19, Stack, Adapter1, Adapter2, Adapter3
from datasets.crowd_sh import Crowd
from losses.bay_loss import Bay_Loss
from losses.post_prob import Post_Prob
import itertools


def train_collate(batch):
    transposed_batch = list(zip(*batch))
    images = torch.stack(transposed_batch[0], 0)
    points = transposed_batch[1]  # the number of points is not fixed, keep it as a list of tensor
    targets = transposed_batch[2]
    st_sizes = torch.FloatTensor(transposed_batch[3])
    return images, points, targets, st_sizes


class RegTrainer(Trainer):
    def setup(self):
        """initial the datasets, model, loss and optimizer"""
        args = self.args
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            self.device_count = torch.cuda.device_count()
            # for code conciseness, we release the single gpu version
            assert self.device_count == 1
            logging.info('using {} gpus'.format(self.device_count))
        else:
            raise Exception("gpu is not available")

        self.downsample_ratio = args.downsample_ratio
        self.datasets = {x: Crowd(os.path.join(args.data_dir, x),
                                  args.crop_size,
                                  args.downsample_ratio,
                                  args.is_gray, x) for x in ['train', 'val']}

        self.dataloaders = {x: DataLoader(self.datasets[x],
                                          collate_fn=(train_collate
                                                      if x == 'train' else default_collate),
                                          batch_size=(args.batch_size
                                          if x == 'train' else 1),
                                          shuffle=(True if x == 'train' else False),
                                          num_workers=args.num_workers*self.device_count,
                                          pin_memory=(True if x == 'train' else False))
                            for x in ['train', 'val']}

        global Adapter1, Adapter2, Adapter3
        Adapter1, Adapter2, Adapter3 = Adapter1(), Adapter2(), Adapter3()
        self.adapters = [Adapter1, Adapter2, Adapter3]
        
        self.models = []
        self.optimizers = []
        for _ in range(3):
            self.models.append(vgg19(flag=_))
            self.models[-1].to(self.device)
            self.optimizers.append(optim.Adam(itertools.chain(self.models[-1].parameters(), self.adapters[_].parameters()),
                     lr=args.lr, weight_decay=args.weight_decay))
        self.stack = Stack()
        self.opt = optim.Adam(itertools.chain(self.stack.parameters(), 
                              self.models[0].parameters(), self.models[1].parameters(), self.models[2].parameters(),
                              self.adapters[0].parameters(), self.adapters[1].parameters(), self.adapters[2].parameters()), 
                              lr=args.lr, weight_decay=args.weight_decay)

        # device = torch.device('cuda')
        # model_dic = torch.load(os.path.join('/content/drive/MyDrive/Bayesian-Crowd-Counting-master-shb/result', 
        #                       'best_model.pth'), device) 
        # self.models[0].load_state_dict(model_dic[0])
        # self.adapters[0].load_state_dict(model_dic[1])
        # self.models[1].load_state_dict(model_dic[2])
        # self.adapters[1].load_state_dict(model_dic[3])
        # self.models[2].load_state_dict(model_dic[4])
        # self.adapters[2].load_state_dict(model_dic[5])
        # self.stack.load_state_dict(model_dic[6])

        self.start_epoch = 0
        if args.resume:
            suf = args.resume.rsplit('.', 1)[-1]
            if suf == 'tar':
                checkpoint = torch.load(args.resume, self.device)
                self.model.load_state_dict(checkpoint['model_state_dict'])
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                self.start_epoch = checkpoint['epoch'] + 1
            elif suf == 'pth':
                self.model.load_state_dict(torch.load(args.resume, self.device))

        self.post_prob = Post_Prob(args.sigma,
                                   args.crop_size,
                                   args.downsample_ratio,
                                   args.background_ratio,
                                   args.use_background,
                                   self.device)  
        self.criterion = Bay_Loss(args.use_background, self.device)
        self.save_list = Save_Handle(max_num=args.max_model_num)
        self.best_mae = np.inf
        self.best_mse = np.inf
        self.best_count = 0

    def train(self):
        """training process"""
        args = self.args
        for epoch in range(self.start_epoch, args.max_epoch):
            logging.info('-'*5 + 'Epoch {}/{}'.format(epoch, args.max_epoch - 1) + '-'*5)
            self.epoch = epoch
            self.train_eopch()
            if epoch % args.val_epoch == 0 and epoch >= args.val_start:
                self.val_epoch()

    def train_eopch(self):
        epoch_loss = AverageMeter()
        epoch_mae = AverageMeter()
        epoch_mse = AverageMeter()
        epoch_start = time.time()
        for idx in range(3):
            self.models[idx].train()  # Set model to train mode
            self.adapters[idx].train()
        self.stack.train()
        
        # Iterate over data.
        for step, (inputs, points, targets, st_sizes) in enumerate(self.dataloaders['train']):
            inputs = inputs.to(self.device)
            st_sizes = st_sizes.to(self.device)
            gd_count = np.array([len(p) for p in points], dtype=np.float32)
            points = [p.to(self.device) for p in points]
            targets = [t.to(self.device) for t in targets]

            with torch.set_grad_enabled(True):
                outputs_0 = self.models[0](inputs)
                outputs_0 = self.adapters[0](outputs_0)
                outputs_1 = self.models[1](inputs)
                outputs_1 = self.adapters[1](outputs_1)
                outputs_2 = self.models[2](inputs)
                outputs_2 = self.adapters[2](outputs_2)
                answers = torch.cat([outputs_0, outputs_1, outputs_2], dim=1)
                outputs = self.stack(answers)

                prob_list = self.post_prob(points, st_sizes)
                loss = self.criterion(prob_list, targets, outputs)

                self.opt.zero_grad()
                loss.backward()
                self.opt.step()

                N = inputs.size(0)
                pre_count = torch.sum(outputs.view(N, -1), dim=1).detach().cpu().numpy()
                res = pre_count - gd_count

                epoch_loss.update(loss.item(), N)
                epoch_mse.update(np.mean(res * res), N)
                epoch_mae.update(np.mean(abs(res)), N)

        logging.info('Epoch {} Train, Loss: {:.2f}, MSE: {:.2f} MAE: {:.2f}, Cost {:.1f} sec'
                      .format(self.epoch, epoch_loss.get_avg(), np.sqrt(epoch_mse.get_avg()), epoch_mae.get_avg(),
                              time.time()-epoch_start))

    def val_epoch(self):
        epoch_start = time.time()
        for idx in range(3):
            self.models[idx].eval()  # Set model to evaluate mode
            self.adapters[idx].eval()
        self.stack.eval()
        epoch_res = []

        # Iterate over data.
        for inputs, count, name in self.dataloaders['val']:
            inputs = inputs.to(self.device)
            # inputs are images with different sizes
            assert inputs.size(0) == 1, 'the batch size should equal to 1 in validation mode'
            with torch.set_grad_enabled(False):
                outputs_0 = self.models[0](inputs)
                outputs_0 = self.adapters[0](outputs_0)
                outputs_1 = self.models[1](inputs)
                outputs_1 = self.adapters[1](outputs_1)
                outputs_2 = self.models[2](inputs)
                outputs_2 = self.adapters[2](outputs_2)
                answers = torch.cat([outputs_0, outputs_1, outputs_2], dim=1)
                outputs = self.stack(answers)

                res = count[0].item() - torch.sum(outputs).item()
                epoch_res.append(res)

        epoch_res = np.array(epoch_res)
        mse = np.sqrt(np.mean(np.square(epoch_res)))
        mae = np.mean(np.abs(epoch_res))
        logging.info('Epoch {} Val, MSE: {:.2f} MAE: {:.2f}, Cost {:.1f} sec'
                     .format(self.epoch, mse, mae, time.time()-epoch_start))

        model_state_dic = [self.models[0].state_dict(), self.adapters[0].state_dict(),
                           self.models[1].state_dict(), self.adapters[1].state_dict(),
                           self.models[2].state_dict(), self.adapters[2].state_dict(),
                           self.stack.state_dict()]
        if (2.0 * mse + mae) < (2.0 * self.best_mse + self.best_mae):
            self.best_mse = mse
            self.best_mae = mae
            logging.info("save best mse {:.2f} mae {:.2f} model epoch {}".format(self.best_mse, self.best_mae, self.epoch))
            torch.save(model_state_dic, os.path.join(self.save_dir, 'best_model.pth'))

