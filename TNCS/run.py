import math

import pandas as pd


from TNCS.model import *

import torch.optim as optim
import warnings

from utils import utility
from utils.utility import progress_bar
import torch.backends.cudnn as cudnn
import torch
from torch import nn
from torchvision.models.vgg import vgg19


import csv

warnings.filterwarnings("ignore")

vgg = vgg19(pretrained=True)
loss_network = nn.Sequential(*list(vgg.features)[:36]).eval()
loss_network.cuda()
for param in loss_network.parameters():
    param.requires_grad = False

a = []
results = {'d_loss': [], 'g_loss': [], 'd_score': [], 'g_score': [], 'psnr': [], 'ssim': []}


class CSGAN_Trainer(object):
    def __init__(self, config, training_loader, testing_loader):
        super(CSGAN_Trainer, self).__init__()
        self.CUDA = torch.cuda.is_available()
        self.device = torch.device('cuda' if self.CUDA else 'cpu')
        self.netG = None

        self.lr = config.lr
        self.nEpochs = config.nEpochs
        self.epoch_pretrain = 0
        self.sensing_rate = config.sensing_rate

        self.layer_num = config.layer_num
        self.output_path = './epochs'
        self.criterionG = None  # optimizer 负责更新模型参数 ，而 criterion 负责计算损失 （计算梯度）

        self.optimizerG = None
        self.warm_epochs = config.warm_epochs
        self.last_lr = config.last_lr

        self.feature_extractor = None
        self.scheduler = None
        self.seed = config.seed
        self.batchsize = config.batchSize
        self.training_loader = training_loader
        self.testing_loader = testing_loader
        self.criterionG = torch.nn.L1Loss()  # L1 损失是像素级别的损失
        # self.criterionG =F.mse_loss
        self.criterionMSE = torch.nn.MSELoss()  # 可用于计算模型的预测与实际目标之间的均方误差

    def build_model(self):
        self.netG = tncs(LayerNo=self.layer_num, sensing_rate=self.sensing_rate).to(self.device)
        self.netG.weight_init(mean=0.0, std=0.2)
        torch.manual_seed(self.seed)

        if self.CUDA:
            torch.cuda.manual_seed(self.seed)
            cudnn.benchmark = True
            self.criterionG.cuda()

        self.optimizerG = torch.optim.Adam(self.netG.parameters(), lr=self.lr, betas=(0.9, 0.999))

        # 这个优化器将根据损失函数的梯度信息来调整判别器的权重，以改善其性能。

        # self.schedulerG = torch.optim.lr_scheduler.StepLR(self.optimizerG, step_size=400, gamma=0.5)

        # scheduler_cosine = optim.lr_scheduler.CosineAnnealingLR(self.optimizerG, self.nEpochs- self.warm_epochs,
        #                                                         eta_min=self.last_lr)
        # self.schedulerG = GradualWarmupScheduler(self.optimizerG, multiplier=1, total_epoch=self.warm_epochs,
        #                                    after_scheduler=scheduler_cosine)

        self.schedulerG = torch.optim.lr_scheduler.StepLR(self.optimizerG, step_size=400, gamma=0.5)

    # 使用 StepLR 学习率调度器允许你在训练过程中逐渐降低生成器网络的学习率。这有助于稳定训练，尤其是在训练过程中
    # 损失函数的变化不再明显或训练靠近收敛点时。
    # 通过减小学习率，你可以更小心地进行权重更新，以获得更好的模型性能。

    def train(self):
        # models setup
        self.netG.train()
        # print('generator parameters:', sum(param.numel() for param in self.netG.parameters()))

        train_loss = 0

        for batch_num, (data, target) in enumerate(self.training_loader):
            data, target = data.to(self.device), target.to(self.device)
            self.optimizerG.zero_grad()
            real = self.netG(data)

            loss = self.criterionG(real, data)

            # g_real = torch.cat([real, real, real], dim=1, out=None)
            # data = torch.cat([data, data, data], dim=1, out=None)
            # vgg19_loss = self.criterionMSE(loss_network(g_real), loss_network(data))
            #
            # loss = loss + 0.006 * vgg19_loss
            train_loss += loss.item()
            loss.backward()
            self.optimizerG.step()

            progress_bar(batch_num, len(self.training_loader), 'Loss: %.4f' % (train_loss / (batch_num + 1)))
        return format(train_loss / len(self.training_loader))

    def test(self):
        self.netG.eval()
        sum_psnr = 0
        with torch.no_grad():
            for batch_num, (data, target) in enumerate(self.testing_loader):
                data, target = data.to(self.device), target.to(self.device)
                prediction = self.netG(data)
                mse1 = self.criterionMSE(prediction, data)
                psnr1 = 10 * math.log10(1 / mse1.item())
                sum_psnr += psnr1

                progress_bar(batch_num, len(self.testing_loader), 'PSNR: %.4f' % (sum_psnr / (batch_num + 1)))
        avg_psnr = sum_psnr / len(self.testing_loader)

        return avg_psnr

    def run(self):
        self.build_model()
        # log = "./epochs/README.txt"
        # file = open(log, 'a')

        # fo = open("D:\\CSProject\\TNCS\\paper test\\number\\5.csv", 'w', newline='')
        # header = ["Epoch", "PSNR"]
        # writer = csv.writer(fo)
        # writer.writerow(header)

        for epoch in range(1, self.nEpochs + 1):
            print("\n===> Epoch {} starts:".format(epoch))
            loss = self.train()
            avg_psnr = self.test()
            results['psnr'].append(avg_psnr)

            # writer.writerow([epoch, avg_psnr])

            self.schedulerG.step()
            model_out_path = self.output_path + "/model_path" + str(epoch) + ".pth"
            if (epoch>0):
                torch.save(self.netG, model_out_path)
            a.append(str(avg_psnr))

        max_index = 0
        list_index = 0
        for num in a:
            if num > a[max_index]:
                max_index = list_index
            list_index += 1
        print("max_avg_psnr: Epoch " + str(max_index + 1) + " , " + str(a[max_index]))
        out = str("max_avg_psnr: Epoch " + str(max_index + 1) + " , " + str(a[max_index]))
        # file.write(out)
        #         # file.close()
        # fo.close()
        # out_path = 'data_results/'
        # data_frame = pd.DataFrame(
        #     data={'Loss_G': results['g_loss'], 'Loss_D': results['d_loss'], 'PSNR': results['psnr']},
        #     index=range(1, epoch + 1))
        # data_frame.to_csv(out_path + 'MR-CSGAN_' + 'train_results_' + str(self.sensing_rate) + '.csv',
        #                   index_label='Epoch')
