# import torch
# import torch.nn as nn
# from PIL.Image import Image
# from torch.autograd import Variable
# import torch.nn.functional as F
# from einops import rearrange
# import numbers
# import argparse
# import warnings
# from torch.nn import init
#
# import numpy as np
# from PIL import Image
# import torchvision.transforms as transforms
# import sys  # 导入sys模块
#
#
# sys.setrecursionlimit(2000)  # 将默认的递归深度修改为3000
#
#
# def conv_layer(in_channels, out_channels, kernel_size, stride=1, dilation=1, groups=1):
#     padding = int((kernel_size - 1) / 2) * dilation
#     return nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=padding, bias=True, dilation=dilation,
#                      groups=groups)
# # 默认卷积实现
# def default_conv(in_channels, out_channels, kernel_size, bias=True):
#     return nn.Conv2d(in_channels, out_channels, kernel_size, padding=(kernel_size // 2), bias=bias)
#
#
#
# def to_3d(x):
#     return rearrange(x, 'b c h w -> b (h w) c')
#
#
# def to_4d(x, h, w):
#     return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)
#
#
# class BiasFree_LayerNorm(nn.Module):
#     def __init__(self, normalized_shape):
#         super(BiasFree_LayerNorm, self).__init__()
#         if isinstance(normalized_shape, numbers.Integral):
#             normalized_shape = (normalized_shape,)
#         normalized_shape = torch.Size(normalized_shape)
#         assert len(normalized_shape) == 1
#         self.weight = nn.Parameter(torch.ones(normalized_shape))
#         self.normalized_shape = normalized_shape
#
#     def forward(self, x):
#         sigma = x.var(-1, keepdim=True, unbiased=False)
#         return x / torch.sqrt(sigma + 1e-5) * self.weight
#
#
# class WithBias_LayerNorm(nn.Module):
#     def __init__(self, normalized_shape):
#         super(WithBias_LayerNorm, self).__init__()
#         if isinstance(normalized_shape, numbers.Integral):
#             normalized_shape = (normalized_shape,)
#         normalized_shape = torch.Size(normalized_shape)
#         assert len(normalized_shape) == 1
#         self.weight = nn.Parameter(torch.ones(normalized_shape))
#         self.bias = nn.Parameter(torch.zeros(normalized_shape))
#         self.normalized_shape = normalized_shape
#
#     def forward(self, x):
#         mu = x.mean(-1, keepdim=True)
#         sigma = x.var(-1, keepdim=True, unbiased=False)
#         return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias
#
#
# class LayerNorm(nn.Module):
#     def __init__(self, dim, LayerNorm_type):
#         super(LayerNorm, self).__init__()
#         if LayerNorm_type == 'BiasFree':
#             self.body = BiasFree_LayerNorm(dim)
#         else:
#             self.body = WithBias_LayerNorm(dim)
#
#     def forward(self, x):
#         h, w = x.shape[-2:]
#         return to_4d(self.body(to_3d(x)), h, w)
#
#
# class LayerNorm(nn.Module):
#
#     def __init__(self, dim, LayerNorm_type):
#         super(LayerNorm, self).__init__()
#         if LayerNorm_type == 'BiasFree':
#             self.body = BiasFree_LayerNorm(dim)
#         else:
#             self.body = WithBias_LayerNorm(dim)
#
#     def forward(self, x):
#         h, w = x.shape[-2:]
#         return to_4d(self.body(to_3d(x)), h, w)
#
#
# #
# # class ESA(nn.Module):        #256*256*64
# #     def __init__(self, n_feats, conv):
# #         super(ESA, self).__init__()
# #         f = n_feats // 4         # 16
# #         self.conv1 = conv(n_feats, f, kernel_size=1)
# #         self.conv_f = conv(f, f, kernel_size=1)
# #         self.conv_max = conv(f, f, kernel_size=3, padding=1)
# #         self.conv2 = conv(f, f, kernel_size=3, stride=2, padding=0)
# #         self.conv3 = conv(f, f, kernel_size=3, padding=1)
# #         self.conv3_ = conv(f, f, kernel_size=3, padding=1)
# #         self.conv4 = conv(f, n_feats, kernel_size=1)
# #         self.sigmoid = nn.Sigmoid()
# #         self.relu = nn.ReLU(inplace=True)
# #
# #     def forward(self, x):
# #         c1_ = (self.conv1(x))        #256*256*16
# #         c1 = self.conv2(c1_)         #126*126*16
# #         v_max = F.max_pool2d(c1, kernel_size=7, stride=3)  #40x40x16
# #         v_range = self.relu(self.conv_max(v_max))
# #         c3 = self.relu(self.conv3(v_range))
# #         c3 = self.conv3_(c3)   #40x40x16
# #         c3 = F.interpolate(c3, (x.size(2), x.size(3)), mode='bilinear', align_corners=False)  #256x256x16
# #         cf = self.conv_f(c1_)   #256x256x16
# #         c4 = self.conv4(c3 + cf)
# #         m = self.sigmoid(c4)    #256x256x64
# #
# #         return x * m
#
#
# # class LNCT(nn.Module):
# #     def __init__(self, in_channels=64, distillation_rate=0.25):
# #         super(LNCT, self).__init__()
# #         self.rc = self.remaining_channels = in_channels
# #         self.c1_r = conv_layer(in_channels, self.rc, 3)
# #         self.esa = ESA(in_channels, nn.Conv2d)
# #         self.esa2 = ESA(in_channels, nn.Conv2d)
# #         self.atten = CrossAtten()
# #
# #
# #     def forward(self, input):
# #
# #         input = self.esa2(input)
# #         input=self.atten(input,input)
# #         out_fused = self.esa(self.c1_r(input))
# #
# #         return out_fused
#
#
# # # 输出高度 = (输入高度 - 卷积核高度 + 2 * 填充) / 步幅 + 1
# # # 输出宽度 = (输入宽度 - 卷积核宽度 + 2 * 填充) / 步幅 + 1
# # #定义网络多尺度残差模块
# class MSRB(nn.Module):
#     def __init__(self):
#         super(MSRB, self).__init__()
#         self.conv_3_1 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=True)
#         self.conv_3_2 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1, bias=True)
#         self.conv_5_1 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=5, stride=1, padding=2, bias=True)
#         self.conv_5_2 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=5, stride=1, padding=2, bias=True)
#         self.confusion = nn.Conv2d(in_channels=256, out_channels=64, kernel_size=1, stride=1, padding=0, bias=False)
#         self.relu = nn.ReLU(inplace=True)
#
#     def forward(self, x):
#         identity_data = x  # 254*254*64
#         output_3_1 = self.relu(self.conv_3_1(x))
#         output_5_1 = self.relu(self.conv_5_1(x))
#         input_2 = torch.cat([output_3_1, output_5_1], 1)  # 256*256*64 +252*252*64
#         output_3_2 = self.relu(self.conv_3_2(input_2))
#         output_5_2 = self.relu(self.conv_5_2(input_2))
#         output = torch.cat([output_3_2, output_5_2], 1)  # 特征图数量变多 256*256*128 +256*256*128
#
#         output = self.confusion(output)  # 256*256*64
#         output = torch.add(output, identity_data)
#         return output
#
#
# class CrossAtten(torch.nn.Module):
#     def __init__(self):
#         super(CrossAtten, self).__init__()
#         self.channels =128
#         self.softmax = nn.Softmax(dim=-1)  # Softmax 操作将输入转换成概率分布，其中每个元素的值都被映射到 (0, 1) 的范围，并且所有元素的和为 1。
#         self.norm1 = LayerNorm(self.channels, 'WithBias')
#         self.norm2 = LayerNorm(self.channels, 'WithBias')
#         self.conv_q = nn.Sequential(
#             nn.Conv2d(in_channels=self.channels, out_channels=self.channels, kernel_size=1, stride=1, bias=True),
#             nn.Conv2d(self.channels, self.channels, kernel_size=3, stride=1, padding=1, groups=self.channels, bias=True)
#         )  # 分组卷积使每个通道组独立计算，增强了通道之间的特征表达。
#         self.conv_k = nn.Sequential(
#             nn.Conv2d(in_channels=self.channels, out_channels=self.channels * 2, kernel_size=1, stride=1, bias=True),
#             nn.Conv2d(self.channels * 2, self.channels * 2, kernel_size=3, stride=1, padding=1,
#                       groups=self.channels * 2, bias=True)
#         )
#         self.conv_kv = nn.Sequential(
#             nn.Conv2d(in_channels=self.channels, out_channels=self.channels * 2, kernel_size=1, stride=1, bias=True),
#             nn.Conv2d(self.channels * 2, self.channels * 2, kernel_size=3, stride=1, padding=1,
#                       groups=self.channels * 2, bias=True)
#         )
#
#         self.conv_v = nn.Sequential(
#             nn.Conv2d(in_channels=self.channels, out_channels=self.channels * 2, kernel_size=1, stride=1, bias=True),
#             nn.Conv2d(self.channels * 2, self.channels * 2, kernel_size=3, stride=1, padding=1,
#                       groups=self.channels * 2, bias=True)
#         )
#
#         self.conv_out = nn.Conv2d(in_channels=self.channels, out_channels=self.channels, kernel_size=1, stride=1,
#                                   bias=True)
#
#     def forward(self, pre, cur):  # 标准化 分组卷积  计算VKQ 注意力权重
#         b, c, h, w = pre.shape  # 1×31×96×96
#         pre_ln = self.norm1(pre)  # 每个通道上进行标准化，使得每个通道的均值为零，标准差为一，从 而有助于训练的稳定性。
#         cur_ln = self.norm2(cur)
#         q = self.conv_q(cur_ln)  # 1×31×96×96
#         q = q.view(b, c, -1)  # 1×31×9216
#         k, v = self.conv_kv(pre_ln).chunk(2, dim=1)  # 结果是 1×62×96×96 经过chunk(2, dim=1)分割 k和 v 1×31×96×96
#         k = k.view(b, c, -1)
#         v = v.view(b, c, -1)  # k 和 v 的大小均为  # 1×31×96×96
#         q = torch.nn.functional.normalize(q, dim=-1)  # 是对输入的张量 q 进行标准化操作
#         k = torch.nn.functional.normalize(k, dim=-1)
#         att = torch.matmul(q, k.permute(0, 2, 1))  # 1×31×96×96  计算注意力权重
#         att = self.softmax(att)  # 其中每个元素表示相应位置的权重
#         out = torch.matmul(att, v).view(b, c, h, w)  # 其中输入张量 v 的不同通道的信息受到了根据注意力权重进行的调整
#         out = self.conv_out(out) + cur
#         return out
#
#
# class CAB(nn.Module):
#     def __init__(self, nc, reduction=8, bias=False):
#         super(CAB, self).__init__()
#         self.avg_pool = nn.AdaptiveAvgPool2d(1)
#         self.conv_du = nn.Sequential(
#             nn.Conv2d(nc, nc // reduction, kernel_size=1, padding=0, bias=bias),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(nc // reduction, nc, kernel_size=1, padding=0, bias=bias),
#             nn.Sigmoid()
#         )
#
#     def forward(self, x):
#         y = self.avg_pool(x)
#         y = self.conv_du(y)
#         return x * y
#
#
#
#
# class SplitChannelAttention(nn.Module):
#     def __init__(self, channel, ratio=16):
#         super(SplitChannelAttention, self).__init__()
#         # 通道压缩模块
#         self.layer = nn.Sequential(
#             nn.AdaptiveAvgPool2d((1, 1)),  # 全局平均池化，输出形状 [B, C, 1, 1]
#             nn.Conv2d(channel, channel // ratio, kernel_size=1, bias=False),  # 降维
#             nn.LeakyReLU(0.2, inplace=True)  # 激活函数
#         )
#         # 通道频率注意力模块
#         self.layer_frequency = nn.Sequential(
#             nn.Conv2d(channel // ratio, channel, kernel_size=1, bias=False),  # 恢复通道数
#             nn.Sigmoid()  # 归一化
#         )
#         # 通道空间注意力模块
#         self.layer_spatial = nn.Sequential(
#             nn.Conv2d(channel // ratio, channel, kernel_size=1, bias=False),  # 恢复通道数
#             nn.Sigmoid()  # 归一化
#         )
#
#     def forward(self, x):
#         # 通道压缩后得到 squeeze
#         squeeze = self.layer(x)  # 输出形状 [B, C // ratio, 1, 1]
#         # 通道频率注意力
#         freq_attention = self.layer_frequency(squeeze)  # 输出形状 [B, C, 1, 1]
#         # 通道空间注意力
#         spatial_attention = self.layer_spatial(squeeze)  # 输出形状 [B, C, 1, 1]
#         # 返回两个结果：分别对应频率和空间注意力加权后的输出
#         return x * freq_attention, x * spatial_attention
#
#
#
# class FourierUnit(nn.Module):
#
#     def __init__(self, in_channels, out_channels):
#         super(FourierUnit, self).__init__()
#         self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False)
#         self.bn1 = nn.BatchNorm2d(out_channels)
#         self.act1 = nn.LeakyReLU(0.2, inplace=True)
#
#         self.conv2 = nn.Conv2d(out_channels * 2, out_channels * 2, kernel_size=1, stride=1, padding=0, bias=False)
#         self.bn2 = nn.BatchNorm2d(out_channels * 2)
#         self.act2 = nn.LeakyReLU(0.2, inplace=True)
#
#     def forward(self, x):
#         x = self.act1(self.bn1(self.conv1(x)))
#
#         batch = x.shape[0]
#
#         # (batch, c, h, w/2+1, 2)
#         fft_dim = (-2, -1)
#         ffted = torch.fft.rfftn(x, dim=fft_dim, norm='ortho')
#         ffted = torch.stack((ffted.real, ffted.imag), dim=-1)
#         ffted = ffted.permute(0, 1, 4, 2, 3).contiguous()  # (batch, c, 2, h, w/2+1)
#         ffted = ffted.view((batch, -1,) + ffted.size()[3:])
#
#         ffted = self.conv2(ffted)  # (batch, c*2, h, w/2+1)
#         ffted = self.act2(self.bn2(ffted))
#
#         ffted = ffted.view((batch, -1, 2,) + ffted.size()[2:]).permute(
#             0, 1, 3, 4, 2).contiguous()  # (batch,c, t, h, w/2+1, 2)
#         ffted = torch.complex(ffted[..., 0], ffted[..., 1])
#
#         ifft_shape_slice = x.shape[-2:]
#         output = torch.fft.irfftn(ffted, s=ifft_shape_slice, dim=fft_dim, norm='ortho')
#
#         return output + x
#
# # 8×128×16×16
# class AMC(nn.Module):
#     def __init__(self, in_channel=128, out_channel=128, kernel_size=3, stride=1, padding_mode='reflection',
#                  need_bias=True):
#         super(AMC, self).__init__()
#         self.sca = SplitChannelAttention(channel=in_channel, ratio=16)
#
#
#         self.fu = FourierUnit(in_channel, out_channel)
#         self.relu = nn.ReLU(inplace=True)
#
#         self.padding_mode = padding_mode
#         to_pad = int((kernel_size - 1) / 2)
#         if padding_mode == 'reflection':
#             self.pad1 = nn.ReflectionPad2d(to_pad)
#             self.pad2 = nn.ReflectionPad2d(to_pad)
#             to_pad = 0
#
#         self.spatial_branch1 = nn.Sequential(
#             nn.Conv2d(in_channel, in_channel, kernel_size, stride=stride, padding=to_pad, dilation=1, bias=need_bias),
#             nn.BatchNorm2d(in_channel),
#             nn.LeakyReLU(0.2, inplace=True)
#         )
#         self.spatial_branch2 = nn.Sequential(
#             nn.Conv2d(in_channel, out_channel, kernel_size, stride=stride, padding=to_pad, dilation=1, bias=need_bias),
#             nn.BatchNorm2d(out_channel),
#             nn.LeakyReLU(0.2, inplace=True)
#         )
#
#     def forward(self, x):
#         x_f, x_s = self.sca(x)
#         if self.padding_mode == 'reflection':
#             x_s = self.pad1(x_s)
#         x_s = self.spatial_branch1(x_s)
#         if self.padding_mode == 'reflection':
#             x_s = self.pad2(x_s)
#         x_s = self.spatial_branch2(x_s)
#
#
#         return self.fu(x_f) + x_s
#
#
# ## Supervised Attention Module
# # class SAM(nn.Module):
# #     def __init__(self, n_feat, kernel_size, bias):
# #         super(SAM, self).__init__()
# #         self.conv1 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=True)
# #
# #         # 增加自适应调整层
# #         self.resize_conv = nn.Conv2d(64, 1, kernel_size=1, bias=False)
# #
# #         self.conv3 = nn.Conv2d(1, n_feat, kernel_size=3, stride=1, padding=1, bias=True)
# #
# #     def forward(self, x, x_img):
# #         x1 = self.conv1(x)
# #
# #         # 自适应调整img到x_img大小
# #         img = self.resize_conv(x)  # 使用1x1卷积调整通道数
# #         img = F.interpolate(img, size=(x_img.shape[2], x_img.shape[3]),
# #                             mode='bilinear', align_corners=False)  # 调整尺寸
# #
# #         img = img + x_img
# #
# #         x2 = torch.sigmoid(self.conv3(img))
# #         # x1 = x1 * x2                    #Todo
# #         # x1 = x1 + x
# #         x1 = F.interpolate(
# #             x1,
# #             size=(x2.shape[2], x2.shape[3]),
# #             mode='bilinear',
# #             align_corners=False
# #         )
# #
# #         # 应用注意力权重 - 根据图中373行(高亮蓝色)
# #         x1 = x1 * x2  # 尺寸现在相同
# #
# #         # 残差连接 - 根据图中375行(粉色)
# #         # 注意：需要确保x尺寸匹配
# #         x = F.interpolate(
# #             x,
# #             size=(x1.shape[2], x1.shape[3]),
# #             mode='bilinear',
# #             align_corners=False
# #         )
# #         x1 = x1 + x
# #
# #
# #         return x1, img
#
#
# class SAM(nn.Module):
#     def __init__(self, n_feat, kernel_size, bias):
#         super(SAM, self).__init__()
#         self.conv1 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=True)
#         self.conv2 = nn.Conv2d(n_feat, 1, kernel_size=1, stride=1, padding=0, bias=False)
#         self.conv3 = nn.Conv2d(1, n_feat, kernel_size=3, stride=1, padding=1, bias=True)
#
#
#     def forward(self, x, x_img):
#         x1 = self.conv1(x)
#         img = self.conv2(x)
#         img=img+ x_img
#         x2 = torch.sigmoid(self.conv3(img))
#         x1 = x1 * x2
#         x1 = x1 + x
#         return x1, img
#
#
#
#
# # 8×128×16×16
# class cross_orthogonal(nn.Module):
#     def __init__(self):
#         super(cross_orthogonal, self).__init__()
#         self.conv_3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=True)
#         self.conv_1 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=1, stride=1, padding=0, bias=False)
#
#         self.conv_5 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=5, stride=1, padding=2, bias=True)
#         self.confusion = nn.Conv2d(in_channels=192, out_channels=64, kernel_size=1, stride=1, padding=0, bias=False)
#         self.relu = nn.ReLU(inplace=True)
#
#         self.CrossAtten = CrossAtten()
#
#     def forward(self, x):
#
#         x =  (self.CrossAtten(x, x))
#
#         return x
#
#     ##原
#     # class BasicBlock(torch.nn.Module):
#     #     def __init__(self,n):
#     #         super(BasicBlock, self).__init__()
#     #
#     #         self.getFactor = nn.Sequential(
#     #             nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, stride=1, padding=1, bias=True),
#     #             nn.ReLU(inplace=True),
#     #         )
#     #
#     #
#     #         # self.CrossAtten=CrossAtten()
#     #         # self.msrb=MSRB()
#     #         # self.LNCT=LNCT()
#     #         self.cross_orthogonal=cross_orthogonal()
#     #         # self.lnar = self.make_layer(SwinT)
#     #         self.c = nn.Conv2d(in_channels=1, out_channels=n, kernel_size=32, stride=32)
#     #
#     #         self.relu = nn.ReLU(inplace=True)
#     #         self.amc=AMC()
#     #
#     #         self.recover = nn.Sequential(
#     #             nn.Conv2d(in_channels=192, out_channels=64, kernel_size=1, stride=1, padding=0, bias=False),
#     #             nn.ReLU(inplace=True),
#     #             nn.Conv2d(in_channels=64, out_channels=1, kernel_size=3, stride=1, padding=1, bias=False),
#     #             nn.ReLU(inplace=True),
#     #         )
#     #
#     #     #x, self.Phiweight,z
#     #     def forward(self, x, PhiWeight,z,q):
#     #
#     #         #  z  (16, 102, 3, 3)   PhiWeight (102,1,32,32)    x  8, 1, 64, 64
#     #         r = F.conv_transpose2d(z, PhiWeight, stride=32) + x  # 8*1*64*64
#     #
#     #         # max_r_abs = torch.max(r, dim=0)[0]
#     #         # # 计算 epsilon
#     #         # epsilon = torch.maximum(0.001 * max_r_abs, torch.tensor(0.00001))
#     #         # # 生成独立同分布 b
#     #         # b = torch.randn_like(r, dtype=torch.float32)
#     #         # r=r+epsilon*b
#     #
#     #         r_out = self.getFactor(r)
#     #         LR = r_out+q
#     #
#     #         out = self.cross_orthogonal(r_out)
#     #         concat1 = out
#     #         # out = self.cross_orthogonal(out)
#     #         # concat2 = out
#     #
#     #         q = self.amc(out)
#     #         concat3 = out
#     #         out = torch.cat([LR, concat1, concat3], 1)
#     #         out = self.recover(out)
#     #
#     #
#     #         # 消融
#     #         # out = self.amc(r_out)
#     #         # concat3 = out
#     #         # out = torch.cat([LR], 1)
#     #         # out = self.recover(out)
#     #
#     #         # 返回 divD
#     #         # divD=self.c(b*out)
#     #
#     #
#     #         # x1 = self.c(out)
#     #         # x1 = x1 * x1
#     #
#     #
#     #         # b, c, h, w = x1.shape
#     #         # x1=x1.view(-1,h,w)
#     #         # x1=torch.matmul(x1,x1)
#     #         # x1=x1.view(b,c,h,w)
#     #
#     #         return out+r,q
#
#
#
# # 残差块
# class ResBlock(nn.Module):
#     def __init__(self, conv, n_feats, kernel_size, bias=True, bn=False, act=nn.PReLU(), res_scale=1):
#         super(ResBlock, self).__init__()
#         m = []
#         for i in range(2):
#             if i == 0:
#                 m.append(conv(n_feats, 64, kernel_size, bias=bias))  # 第一层卷积
#             else:
#                 m.append(conv(64, n_feats, kernel_size, bias=bias))  # 第二层卷积
#             if bn:
#                 m.append(nn.BatchNorm2d(n_feats))
#             if i == 0:
#                 m.append(act)
#
#         self.body = nn.Sequential(*m)
#         self.res_scale = res_scale
#
#     def forward(self, x):
#         res = self.body(x).mul(self.res_scale)
#         res += x
#         return res
#
#
# # 上采样模块
# class UpSample(nn.Module):
#     def __init__(self, in_channels, s_factor):
#         super(UpSample, self).__init__()
#         self.resblock = ResBlock(default_conv, in_channels, kernel_size=3)
#
#         # 修改1: 将上采样倍率从2改为4（实现16x16→64x64）
#         # 修改2: 调整卷积输出通道数为64（实现128→64）
#         self.up = nn.Sequential(
#             nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False),
#             nn.Conv2d(in_channels, 64, kernel_size=1, stride=1, padding=0, bias=False)  # 直接指定输出通道为64
#         )
#
#     def forward(self, x):
#         x = self.resblock(x)
#         x = self.up(x)
#         return x
#
#
#
# # class UpSample(nn.Module):
# #     def __init__(self, in_channels, s_factor):
# #         super(UpSample, self).__init__()
# #         self.resblock = ResBlock(default_conv, in_channels, kernel_size=3)  # 残差块
# #         self.up = nn.Sequential(
# #             nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),  # 尺寸扩大2倍
# #             nn.Conv2d(in_channels, in_channels - s_factor, kernel_size=1, stride=1, padding=0, bias=False)  # 调整通道数
# #         )
# #
# #     def forward(self, x):
# #         x = self.resblock(x)  # 通过残差块提取特征
# #         x = self.up(x)  # 上采样
# #         return x
#
#
# # 下采样模块
# class DownSample(nn.Module):
#     def __init__(self, in_channels, s_factor):
#         super(DownSample, self).__init__()
#         self.resblock = ResBlock(default_conv, in_channels, kernel_size=3)  # 残差块
#         self.down = nn.Sequential(
#             nn.Upsample(scale_factor=0.5, mode='bilinear', align_corners=False),
#             nn.Conv2d(in_channels, in_channels + s_factor, kernel_size=1, stride=1, padding=0, bias=False)
#         )
#
#     def forward(self, x):
#         x = self.resblock(x)  # 残差块
#         x = self.down(x)  # 下采样
#         return x
#
#
# class TextureTransformer(nn.Module):
#     """纹理转换器"""
#
#     def __init__(self):
#         super().__init__()
#         # 深度可分离纹理提取器
#         self.lte_l0_down = DS_LTE(128, 128)  # 处理L0↓
#         self.lte_l0 = DS_LTE(64, 128)  # 处理原始L0 (得到V)
#         self.lte_m1_up = DS_LTE(96, 128)  # 处理M1↑ (得到Q)
#
#         # 硬注意力模块
#         self.hard_attention = HardAttention()
#
#         # 软注意力模块
#         self.soft_attention = SoftAttention()
#
#         # F模块特征提取 (简单卷积)
#         self.f_extractor = nn.Conv2d(64, 128, kernel_size=3, padding=1)
#
#         # 新增下采样层
#         self.downsample = nn.AdaptiveAvgPool2d((16, 16))
#
#     def forward(self, l0, m1_up, l0_down, m1):
#         """
#         输入尺寸:
#           l0: [8, 64, 64, 64]      原始高分辨率参考图
#           m1_up: [8, 96, 32, 32]   上采样目标图像
#           l0_down: [8, 128, 16, 16] 下采样参考图
#           m1: [8, 64, 64, 64]      原始目标图像
#         输出:
#           H1: [8, 128, 64, 64]     重建的高分辨率图像
#         """
#         # 1. 生成Q/K/V特征
#         V = self.lte_l0(l0)  # [8,128,64,64]
#         K = self.lte_l0_down(l0_down)  # [8,128,16,16]
#         Q = self.lte_m1_up(m1_up)  # [8,128,32,32]
#
#         # 2. 调整Q尺寸到16x16 (匹配K)
#         Q = F.interpolate(Q, size=(16, 16), mode='bilinear', align_corners=True)
#
#         # 3. 硬注意力 (V和K作为输入)
#         R = self.correlation(Q, K)  # 相关性计算  [8, 256，256]
#         hard_feat = self.hard_attention(R, V)  # [8,128,16,16]
#
#         # 4. 软注意力 (硬注意力和Q作为输入)
#         soft_feat = self.soft_attention(hard_feat, Q)  # [8,128,16,16]
#
#         # 5. F特征提取 (原始M1输入)
#         F_main = self.f_extractor(m1)  # [8,128,16,16]
#
#         # 6. 下采样F_main到16×16
#         F_main_down = self.downsample(F_main)  # [8,128,16,16]
#
#         # 7. 最终输出: F + 上采样后的软注意力结果
#         H1 = F_main_down + soft_feat
#
#         return H1  # [8,128,16,16]
#
#
#
#     def correlation(self, Q, K):
#         """极简点积相关性计算"""
#         # 展平特征
#         Q_flat = Q.flatten(2)  # [8,128,256]
#         K_flat = K.flatten(2)  # [8,128,256]
#
#         # 直接计算点积相似度
#         return torch.bmm(Q_flat.permute(0, 2, 1), K_flat)  # [8,256,256]
#
#
#
# class DS_LTE(nn.Module):
#     """深度可分离纹理提取器"""
#
#     def __init__(self, in_ch, out_ch):
#         super().__init__()
#         # 深度卷积(空间纹理提取)
#         self.depthwise = nn.Conv2d(in_ch, in_ch, 3, padding=1, groups=in_ch)
#         # 点卷积(通道融合)
#         self.pointwise = nn.Conv2d(in_ch, out_ch, 1)
#         # 激活函数
#         self.activation = nn.ReLU()
#
#     def forward(self, x):
#         x = self.depthwise(x)
#         x = self.pointwise(x)
#         return self.activation(x)
#
#
# class HardAttention(nn.Module):
#     """硬注意力模块"""
#
#     def forward(self, R, V):
#         """输入: R(相关性图), V(原始高分辨率特征)"""
#         # 1. 获取每个query的最大相关位置索引 (公式4.5)
#         max_indices = R.argmax(dim=-1)  # [8, 256]
#
#         # 2. 转换V到向量形式: [8, 128, 4096]
#         V_flat = V.view(V.size(0), V.size(1), -1)  # 64 * 64=4096
#
#         # 3. 按索引选择特征
#         batch_idx = torch.arange(V.size(0)).view(-1, 1).expand_as(max_indices)
#         selected = V_flat[batch_idx, :, max_indices]
#
#         # 4. 重塑为空间特征图: [8, 128, 16, 16]
#         return selected.view(V.size(0), V.size(1), 16, 16)
#
#
# class SoftAttention(nn.Module):
#     """软注意力模块"""
#
#     def __init__(self):
#         super().__init__()
#         # 注意力权重生成
#         self.attention = nn.Sequential(
#             nn.Conv2d(256, 128, 1),  # 合并硬注意力和Q后的通道数
#             nn.ReLU(),
#             nn.Conv2d(128, 1, 1),
#             nn.Sigmoid()
#         )
#
#     def forward(self, hard_feat, Q):
#         """输入: hard_feat(硬注意力输出), Q(纹理查询)"""
#         # 1. 连接硬注意力输出和Q (通道维度)
#         concat_feat = torch.cat([hard_feat, Q], dim=1)  # [8,256,16,16]
#
#         # 2. 生成注意力权重图
#         attn_weights = self.attention(concat_feat)  # [8,1,16,16]
#
#         # 3. 加权融合
#         return attn_weights * hard_feat  # [8,128,16,16]
#
#
#
#
#
#
# class BasicBlock(torch.nn.Module):
#     def __init__(self, n):
#         super(BasicBlock, self).__init__()
#
#         self.getFactor = nn.Sequential(
#             nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, stride=1, padding=1, bias=True),
#             nn.ReLU(inplace=True),
#         )
#
#         in_channels = 64
#         s_factor = 32 # 通道变化因子
#         # 定义模块
#         # 定义下采样和上采样模块
#         self.down1 = DownSample(in_channels, s_factor)
#         self.down2 = DownSample(in_channels + s_factor, s_factor)
#         # self.up1 = UpSample(in_channels + 2 * s_factor, s_factor)
#         # 初始化时传入参数示例
#         self.up1 = UpSample(in_channels=128, s_factor=64)  # in_channels=128对应输入通道数
#         self.up2 = UpSample(in_channels + s_factor, s_factor)
#         # self.CrossAtten=CrossAtten()
#         self.msrb=MSRB()
#         # self.LNCT=LNCT()
#         self.cross_orthogonal = cross_orthogonal()
#         self.ttf=TextureTransformer()
#         # self.lnar = self.make_layer(SwinT)
#         self.c = nn.Conv2d(in_channels=1, out_channels=n, kernel_size=32, stride=32)
#
#         self.relu = nn.ReLU(inplace=True)
#         self.amc = AMC()
#         self.sam = SAM(64,3,False)
#         self.outc= nn.Conv2d(in_channels=64, out_channels=1, kernel_size=3, stride=1, padding=1, bias=False),
#         self.c3 = nn.Conv2d(64, 1, 1, 1, bias=False)
#
#
#
#     def forward(self, x, PhiWeight, z, q):
#         #  z  (16, 102, 3, 3)   PhiWeight (102,1,32,32)    x  8, 1, 64, 64
#         r = F.conv_transpose2d(z, PhiWeight, stride=32) + x+q  # 8*1*64*64
#
#         r_out = self.getFactor(r)
#
#         out1 = self.down1(r_out)  # 第一次下采样 8 96 32 32
#         out2 = self.down2(out1)  # 第二次下采样 8 128 16 16
#
#         # out3 = self.amc(out2)  # 8 128 64 64
#
#         # out4 = self.relu(self.ttf(r_out, out1, out3, r_out))
#
#         out5 = self.cross_orthogonal(out2) # 8 128 16 16
#
#         # out5 = (self.ttf(r_out,out1,out4,r_out)) + out4  # 8 128 16 16
#
#         out6 = (self.amc(out5))+out5 # 8 128 16 16
#
#         out7 = self.up1(out6)  # 上采样结果  8  64 64 64
#         # out8 = self.up2(out7)+r_out # 上采样  8  64 64 64
#
#
#         out,q = (self.sam(out7,r))
#         out= (self.c3(out))
#
#
#         return out , q
#
#
#
#
#     def make_layer(self, block):
#         layers = []
#         layers.append(block())
#         return nn.Sequential(*layers)
#
#
# class tncs(torch.nn.Module):
#     def __init__(self, LayerNo, sensing_rate):
#
#         super(tncs, self).__init__()
#         onelayer = []
#         self.LayerNo = LayerNo
#         self.patch_size = 32
#         self.n_input = int(sensing_rate * 1024)  # 截尾
#
#         for i in range(LayerNo):
#             onelayer.append(BasicBlock(self.n_input))
#
#             # 初始化权重  (102,1,32,32)
#         self.Phiweight = nn.Parameter(
#             init.xavier_normal_(torch.Tensor(self.n_input, 1, self.patch_size, self.patch_size)))
#
#         # self.Phiweight_b = nn.Parameter(
#         #     init.xavier_normal_(torch.Tensor(self.n, 1, 64, 64)))
#
#         self.fcs = nn.ModuleList(onelayer)
#         self.var_step = nn.Parameter(torch.tensor(0.1, dtype=torch.float32))
#
#         #
#         self.c1 = (nn.Conv2d(1, 64, 1, 1, bias=False))
#         self.c2 = (nn.Conv2d(64, 64, 3, 1, 1, bias=False, groups=64))
#         self.c3 = nn.Conv2d(64, 1, 1, 1, bias=False)
#         self.relu = nn.ReLU(inplace=True)
#
#
#     def forward(self, x):
#         # y1=self.relu(self.c1(x))
#         # y2 = self.relu(self.c2(y1))
#         # y2 = (self.c3(y2))
#         # y = F.conv2d(y2, self.Phiweight, stride=self.patch_size, padding=0, bias=None)
#
#         y1 = F.conv2d(x, self.Phiweight, stride=self.patch_size, padding=0, bias=None)
#         y2 = self.relu(self.c2(self.relu(self.c1(x))))
#         y2 = (self.c3(y2))
#         y = F.conv2d(y2, self.Phiweight, stride=self.patch_size, padding=0, bias=None)+y1
#
#         # y = F.conv2d(x, self.Phiweight, stride=self.patch_size, padding=0, bias=None)
#         PhiTb = F.conv_transpose2d(y, self.Phiweight, stride=self.patch_size)  # 反卷积 16, 1, 64, 64
#         x = PhiTb
#         q = 0
#         x1 = F.conv2d(x, self.Phiweight, stride=self.patch_size, padding=0, bias=None)
#         z = y - x1  # (8, 102, 2, 2)
#         osger = 0
#         for i in range(self.LayerNo):
#             x, q = self.fcs[i](x, self.Phiweight, z, q)
#             # if i<self.LayerNo-1:
#             #    # osger =osger+self.var_step*z*r
#             z = y - F.conv2d(x, self.Phiweight, stride=self.patch_size, padding=0, bias=None)
#         x_final = x
#
#         return x_final
#
#     def weight_init(self, mean, std):
#         for m in self._modules:
#             if isinstance(m, (nn.Conv2d, nn.Linear)):
#                 nn.init.kaiming_normal_(m.weight, mode='fan_in')
#
# #     def weight_init(self, mean=0.0, std=0.02):
# #         for m in self._modules:
# #             normal_init(self._modules[m], mean, std)
# #
# # def normal_init(m, mean, std):
# #     if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d):
# #         m.weight.data.normal_(mean, std)
# #         m.bias.data.zero_()





import torch
import torch.nn as nn
from PIL.Image import Image
from torch.autograd import Variable
import torch.nn.functional as F
from einops import rearrange
import numbers
import argparse
import warnings
from torch.nn import init

import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import sys  # 导入sys模块


sys.setrecursionlimit(2000)  # 将默认的递归深度修改为3000


def conv_layer(in_channels, out_channels, kernel_size, stride=1, dilation=1, groups=1):
    padding = int((kernel_size - 1) / 2) * dilation
    return nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=padding, bias=True, dilation=dilation,
                     groups=groups)
# 默认卷积实现
def default_conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv2d(in_channels, out_channels, kernel_size, padding=(kernel_size // 2), bias=bias)



def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')


def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)


class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)
        assert len(normalized_shape) == 1
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma + 1e-5) * self.weight


class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)
        assert len(normalized_shape) == 1
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias


class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type == 'BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)


class LayerNorm(nn.Module):

    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type == 'BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)


#
# class ESA(nn.Module):        #256*256*64
#     def __init__(self, n_feats, conv):
#         super(ESA, self).__init__()
#         f = n_feats // 4         # 16
#         self.conv1 = conv(n_feats, f, kernel_size=1)
#         self.conv_f = conv(f, f, kernel_size=1)
#         self.conv_max = conv(f, f, kernel_size=3, padding=1)
#         self.conv2 = conv(f, f, kernel_size=3, stride=2, padding=0)
#         self.conv3 = conv(f, f, kernel_size=3, padding=1)
#         self.conv3_ = conv(f, f, kernel_size=3, padding=1)
#         self.conv4 = conv(f, n_feats, kernel_size=1)
#         self.sigmoid = nn.Sigmoid()
#         self.relu = nn.ReLU(inplace=True)
#
#     def forward(self, x):
#         c1_ = (self.conv1(x))        #256*256*16
#         c1 = self.conv2(c1_)         #126*126*16
#         v_max = F.max_pool2d(c1, kernel_size=7, stride=3)  #40x40x16
#         v_range = self.relu(self.conv_max(v_max))
#         c3 = self.relu(self.conv3(v_range))
#         c3 = self.conv3_(c3)   #40x40x16
#         c3 = F.interpolate(c3, (x.size(2), x.size(3)), mode='bilinear', align_corners=False)  #256x256x16
#         cf = self.conv_f(c1_)   #256x256x16
#         c4 = self.conv4(c3 + cf)
#         m = self.sigmoid(c4)    #256x256x64
#
#         return x * m


# class LNCT(nn.Module):
#     def __init__(self, in_channels=64, distillation_rate=0.25):
#         super(LNCT, self).__init__()
#         self.rc = self.remaining_channels = in_channels
#         self.c1_r = conv_layer(in_channels, self.rc, 3)
#         self.esa = ESA(in_channels, nn.Conv2d)
#         self.esa2 = ESA(in_channels, nn.Conv2d)
#         self.atten = CrossAtten()
#
#
#     def forward(self, input):
#
#         input = self.esa2(input)
#         input=self.atten(input,input)
#         out_fused = self.esa(self.c1_r(input))
#
#         return out_fused


# # 输出高度 = (输入高度 - 卷积核高度 + 2 * 填充) / 步幅 + 1
# # 输出宽度 = (输入宽度 - 卷积核宽度 + 2 * 填充) / 步幅 + 1
# #定义网络多尺度残差模块
class MSRB(nn.Module):
    def __init__(self):
        super(MSRB, self).__init__()
        self.conv_3_1 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv_3_2 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv_5_1 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=5, stride=1, padding=2, bias=True)
        self.conv_5_2 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=5, stride=1, padding=2, bias=True)
        self.confusion = nn.Conv2d(in_channels=256, out_channels=64, kernel_size=1, stride=1, padding=0, bias=False)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity_data = x  # 254*254*64
        output_3_1 = self.relu(self.conv_3_1(x))
        output_5_1 = self.relu(self.conv_5_1(x))
        input_2 = torch.cat([output_3_1, output_5_1], 1)  # 256*256*64 +252*252*64
        output_3_2 = self.relu(self.conv_3_2(input_2))
        output_5_2 = self.relu(self.conv_5_2(input_2))
        output = torch.cat([output_3_2, output_5_2], 1)  # 特征图数量变多 256*256*128 +256*256*128

        output = self.confusion(output)  # 256*256*64
        output = torch.add(output, identity_data)
        return output


class CrossAtten(torch.nn.Module):
    def __init__(self):
        super(CrossAtten, self).__init__()
        self.channels =128
        self.softmax = nn.Softmax(dim=-1)  # Softmax 操作将输入转换成概率分布，其中每个元素的值都被映射到 (0, 1) 的范围，并且所有元素的和为 1。
        self.norm1 = LayerNorm(self.channels, 'WithBias')
        self.norm2 = LayerNorm(self.channels, 'WithBias')
        self.conv_q = nn.Sequential(
            nn.Conv2d(in_channels=self.channels, out_channels=self.channels, kernel_size=1, stride=1, bias=True),
            nn.Conv2d(self.channels, self.channels, kernel_size=3, stride=1, padding=1, groups=self.channels, bias=True)
        )  # 分组卷积使每个通道组独立计算，增强了通道之间的特征表达。
        self.conv_k = nn.Sequential(
            nn.Conv2d(in_channels=self.channels, out_channels=self.channels * 2, kernel_size=1, stride=1, bias=True),
            nn.Conv2d(self.channels * 2, self.channels * 2, kernel_size=3, stride=1, padding=1,
                      groups=self.channels * 2, bias=True)
        )
        self.conv_kv = nn.Sequential(
            nn.Conv2d(in_channels=self.channels, out_channels=self.channels * 2, kernel_size=1, stride=1, bias=True),
            nn.Conv2d(self.channels * 2, self.channels * 2, kernel_size=3, stride=1, padding=1,
                      groups=self.channels * 2, bias=True)
        )

        self.conv_v = nn.Sequential(
            nn.Conv2d(in_channels=self.channels, out_channels=self.channels * 2, kernel_size=1, stride=1, bias=True),
            nn.Conv2d(self.channels * 2, self.channels * 2, kernel_size=3, stride=1, padding=1,
                      groups=self.channels * 2, bias=True)
        )

        self.conv_out = nn.Conv2d(in_channels=self.channels, out_channels=self.channels, kernel_size=1, stride=1,
                                  bias=True)

    def forward(self, pre, cur):  # 标准化 分组卷积  计算VKQ 注意力权重
        b, c, h, w = pre.shape  # 1×31×96×96
        pre_ln = self.norm1(pre)  # 每个通道上进行标准化，使得每个通道的均值为零，标准差为一，从 而有助于训练的稳定性。
        cur_ln = self.norm2(cur)
        q = self.conv_q(cur_ln)  # 1×31×96×96
        q = q.view(b, c, -1)  # 1×31×9216
        k, v = self.conv_kv(pre_ln).chunk(2, dim=1)  # 结果是 1×62×96×96 经过chunk(2, dim=1)分割 k和 v 1×31×96×96
        k = k.view(b, c, -1)
        v = v.view(b, c, -1)  # k 和 v 的大小均为  # 1×31×96×96
        q = torch.nn.functional.normalize(q, dim=-1)  # 是对输入的张量 q 进行标准化操作
        k = torch.nn.functional.normalize(k, dim=-1)
        att = torch.matmul(q, k.permute(0, 2, 1))  # 1×31×96×96  计算注意力权重
        att = self.softmax(att)  # 其中每个元素表示相应位置的权重
        out = torch.matmul(att, v).view(b, c, h, w)  # 其中输入张量 v 的不同通道的信息受到了根据注意力权重进行的调整
        out = self.conv_out(out) + cur
        return out


class CAB(nn.Module):
    def __init__(self, nc, reduction=8, bias=False):
        super(CAB, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_du = nn.Sequential(
            nn.Conv2d(nc, nc // reduction, kernel_size=1, padding=0, bias=bias),
            nn.ReLU(inplace=True),
            nn.Conv2d(nc // reduction, nc, kernel_size=1, padding=0, bias=bias),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y




class SplitChannelAttention(nn.Module):
    def __init__(self, channel, ratio=16):
        super(SplitChannelAttention, self).__init__()
        # 通道压缩模块
        self.layer = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),  # 全局平均池化，输出形状 [B, C, 1, 1]
            nn.Conv2d(channel, channel // ratio, kernel_size=1, bias=False),  # 降维
            nn.LeakyReLU(0.2, inplace=True)  # 激活函数
        )
        # 通道频率注意力模块
        self.layer_frequency = nn.Sequential(
            nn.Conv2d(channel // ratio, channel, kernel_size=1, bias=False),  # 恢复通道数
            nn.Sigmoid()  # 归一化
        )
        # 通道空间注意力模块
        self.layer_spatial = nn.Sequential(
            nn.Conv2d(channel // ratio, channel, kernel_size=1, bias=False),  # 恢复通道数
            nn.Sigmoid()  # 归一化
        )

    def forward(self, x):
        # 通道压缩后得到 squeeze
        squeeze = self.layer(x)  # 输出形状 [B, C // ratio, 1, 1]
        # 通道频率注意力
        freq_attention = self.layer_frequency(squeeze)  # 输出形状 [B, C, 1, 1]
        # 通道空间注意力
        spatial_attention = self.layer_spatial(squeeze)  # 输出形状 [B, C, 1, 1]
        # 返回两个结果：分别对应频率和空间注意力加权后的输出
        return x * freq_attention, x * spatial_attention



class FourierUnit(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(FourierUnit, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.act1 = nn.LeakyReLU(0.2, inplace=True)

        self.conv2 = nn.Conv2d(out_channels * 2, out_channels * 2, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels * 2)
        self.act2 = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        x = self.act1(self.bn1(self.conv1(x)))

        batch = x.shape[0]

        # (batch, c, h, w/2+1, 2)
        fft_dim = (-2, -1)
        ffted = torch.fft.rfftn(x, dim=fft_dim, norm='ortho')
        ffted = torch.stack((ffted.real, ffted.imag), dim=-1)
        ffted = ffted.permute(0, 1, 4, 2, 3).contiguous()  # (batch, c, 2, h, w/2+1)
        ffted = ffted.view((batch, -1,) + ffted.size()[3:])

        ffted = self.conv2(ffted)  # (batch, c*2, h, w/2+1)
        ffted = self.act2(self.bn2(ffted))

        ffted = ffted.view((batch, -1, 2,) + ffted.size()[2:]).permute(
            0, 1, 3, 4, 2).contiguous()  # (batch,c, t, h, w/2+1, 2)
        ffted = torch.complex(ffted[..., 0], ffted[..., 1])

        ifft_shape_slice = x.shape[-2:]
        output = torch.fft.irfftn(ffted, s=ifft_shape_slice, dim=fft_dim, norm='ortho')

        return output + x

# 8×128×16×16
class AMC(nn.Module):
    def __init__(self, in_channel=128, out_channel=128, kernel_size=3, stride=1, padding_mode='reflection',
                 need_bias=True):
        super(AMC, self).__init__()
        self.sca = SplitChannelAttention(channel=in_channel, ratio=16)


        self.fu = FourierUnit(in_channel, out_channel)
        self.relu = nn.ReLU(inplace=True)

        self.padding_mode = padding_mode
        to_pad = int((kernel_size - 1) / 2)
        if padding_mode == 'reflection':
            self.pad1 = nn.ReflectionPad2d(to_pad)
            self.pad2 = nn.ReflectionPad2d(to_pad)
            to_pad = 0

        self.spatial_branch1 = nn.Sequential(
            nn.Conv2d(in_channel, in_channel, kernel_size, stride=stride, padding=to_pad, dilation=1, bias=need_bias),
            nn.BatchNorm2d(in_channel),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.spatial_branch2 = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size, stride=stride, padding=to_pad, dilation=1, bias=need_bias),
            nn.BatchNorm2d(out_channel),
            nn.LeakyReLU(0.2, inplace=True)
        )

    def forward(self, x):
        x_f, x_s = self.sca(x)
        if self.padding_mode == 'reflection':
            x_s = self.pad1(x_s)
        x_s = self.spatial_branch1(x_s)
        if self.padding_mode == 'reflection':
            x_s = self.pad2(x_s)
        x_s = self.spatial_branch2(x_s)


        return self.fu(x_f) + x_s


## Supervised Attention Module
class SAM(nn.Module):
    def __init__(self, n_feat, kernel_size, bias):
        super(SAM, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=True)

        self.conv2 = nn.Conv2d(n_feat, 1, kernel_size=1, stride=1, padding=0, bias=False)

        self.conv3 = nn.Conv2d(1, n_feat, kernel_size=3, stride=1, padding=1, bias=True)

    def forward(self, x, x_img):
        x1 = self.conv1(x)
        img = self.conv2(x)
        img=img+ x_img
        x2 = torch.sigmoid(self.conv3(img))
        x1 = x1 * x2
        x1 = x1 + x
        return x1, img


# 8×128×16×16
class cross_orthogonal(nn.Module):
    def __init__(self):
        super(cross_orthogonal, self).__init__()
        self.conv_3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv_1 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=1, stride=1, padding=0, bias=False)

        self.conv_5 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=5, stride=1, padding=2, bias=True)
        self.confusion = nn.Conv2d(in_channels=192, out_channels=64, kernel_size=1, stride=1, padding=0, bias=False)
        self.relu = nn.ReLU(inplace=True)

        self.CrossAtten = CrossAtten()

    def forward(self, x):


        x =  (self.CrossAtten(x, x))

        return x

    ##原
    # class BasicBlock(torch.nn.Module):
    #     def __init__(self,n):
    #         super(BasicBlock, self).__init__()
    #
    #         self.getFactor = nn.Sequential(
    #             nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, stride=1, padding=1, bias=True),
    #             nn.ReLU(inplace=True),
    #         )
    #
    #
    #         # self.CrossAtten=CrossAtten()
    #         # self.msrb=MSRB()
    #         # self.LNCT=LNCT()
    #         self.cross_orthogonal=cross_orthogonal()
    #         # self.lnar = self.make_layer(SwinT)
    #         self.c = nn.Conv2d(in_channels=1, out_channels=n, kernel_size=32, stride=32)
    #
    #         self.relu = nn.ReLU(inplace=True)
    #         self.amc=AMC()
    #
    #         self.recover = nn.Sequential(
    #             nn.Conv2d(in_channels=192, out_channels=64, kernel_size=1, stride=1, padding=0, bias=False),
    #             nn.ReLU(inplace=True),
    #             nn.Conv2d(in_channels=64, out_channels=1, kernel_size=3, stride=1, padding=1, bias=False),
    #             nn.ReLU(inplace=True),
    #         )
    #
    #     #x, self.Phiweight,z
    #     def forward(self, x, PhiWeight,z,q):
    #
    #         #  z  (16, 102, 3, 3)   PhiWeight (102,1,32,32)    x  8, 1, 64, 64
    #         r = F.conv_transpose2d(z, PhiWeight, stride=32) + x  # 8*1*64*64
    #
    #         # max_r_abs = torch.max(r, dim=0)[0]
    #         # # 计算 epsilon
    #         # epsilon = torch.maximum(0.001 * max_r_abs, torch.tensor(0.00001))
    #         # # 生成独立同分布 b
    #         # b = torch.randn_like(r, dtype=torch.float32)
    #         # r=r+epsilon*b
    #
    #         r_out = self.getFactor(r)
    #         LR = r_out+q
    #
    #         out = self.cross_orthogonal(r_out)
    #         concat1 = out
    #         # out = self.cross_orthogonal(out)
    #         # concat2 = out
    #
    #         q = self.amc(out)
    #         concat3 = out
    #         out = torch.cat([LR, concat1, concat3], 1)
    #         out = self.recover(out)
    #
    #
    #         # 消融
    #         # out = self.amc(r_out)
    #         # concat3 = out
    #         # out = torch.cat([LR], 1)
    #         # out = self.recover(out)
    #
    #         # 返回 divD
    #         # divD=self.c(b*out)
    #
    #
    #         # x1 = self.c(out)
    #         # x1 = x1 * x1
    #
    #
    #         # b, c, h, w = x1.shape
    #         # x1=x1.view(-1,h,w)
    #         # x1=torch.matmul(x1,x1)
    #         # x1=x1.view(b,c,h,w)
    #
    #         return out+r,q



# 残差块
class ResBlock(nn.Module):
    def __init__(self, conv, n_feats, kernel_size, bias=True, bn=False, act=nn.PReLU(), res_scale=1):
        super(ResBlock, self).__init__()
        m = []
        for i in range(2):
            if i == 0:
                m.append(conv(n_feats, 64, kernel_size, bias=bias))  # 第一层卷积
            else:
                m.append(conv(64, n_feats, kernel_size, bias=bias))  # 第二层卷积
            if bn:
                m.append(nn.BatchNorm2d(n_feats))
            if i == 0:
                m.append(act)

        self.body = nn.Sequential(*m)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x).mul(self.res_scale)
        res += x
        return res


# 上采样模块
class UpSample(nn.Module):
    def __init__(self, in_channels, s_factor):
        super(UpSample, self).__init__()
        self.resblock = ResBlock(default_conv, in_channels, kernel_size=3)

        # 修改1: 将上采样倍率从2改为4（实现16x16→64x64）
        # 修改2: 调整卷积输出通道数为64（实现128→64）
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False),
            nn.Conv2d(in_channels, 64, kernel_size=1, stride=1, padding=0, bias=False)  # 直接指定输出通道为64
        )

    def forward(self, x):
        x = self.resblock(x)
        x = self.up(x)
        return x



# class UpSample(nn.Module):
#     def __init__(self, in_channels, s_factor):
#         super(UpSample, self).__init__()
#         self.resblock = ResBlock(default_conv, in_channels, kernel_size=3)  # 残差块
#         self.up = nn.Sequential(
#             nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),  # 尺寸扩大2倍
#             nn.Conv2d(in_channels, in_channels - s_factor, kernel_size=1, stride=1, padding=0, bias=False)  # 调整通道数
#         )
#
#     def forward(self, x):
#         x = self.resblock(x)  # 通过残差块提取特征
#         x = self.up(x)  # 上采样
#         return x


# 下采样模块
class DownSample(nn.Module):
    def __init__(self, in_channels, s_factor):
        super(DownSample, self).__init__()
        self.resblock = ResBlock(default_conv, in_channels, kernel_size=3)  # 残差块
        self.down = nn.Sequential(
            nn.Upsample(scale_factor=0.5, mode='bilinear', align_corners=False),
            nn.Conv2d(in_channels, in_channels + s_factor, kernel_size=1, stride=1, padding=0, bias=False)
        )

    def forward(self, x):
        x = self.resblock(x)  # 残差块
        x = self.down(x)  # 下采样
        return x




class BasicBlock(torch.nn.Module):
    def __init__(self, n):
        super(BasicBlock, self).__init__()

        self.getFactor = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(inplace=True),
        )

        in_channels = 64
        s_factor = 32 # 通道变化因子
        # 定义模块
        # 定义下采样和上采样模块
        self.down1 = DownSample(in_channels, s_factor)
        self.down2 = DownSample(in_channels + s_factor, s_factor)
        # self.up1 = UpSample(in_channels + 2 * s_factor, s_factor)
        # 初始化时传入参数示例
        self.up1 = UpSample(in_channels=128, s_factor=64)  # in_channels=128对应输入通道数
        self.up2 = UpSample(in_channels + s_factor, s_factor)
        # self.CrossAtten=CrossAtten()
        # self.msrb=MSRB()
        # self.LNCT=LNCT()
        self.cross_orthogonal = cross_orthogonal()
        # self.lnar = self.make_layer(SwinT)
        self.c = nn.Conv2d(in_channels=1, out_channels=n, kernel_size=32, stride=32)

        self.relu = nn.ReLU(inplace=True)
        self.amc = AMC()
        self.sam = SAM(64,3,False)
        self.outc= nn.Conv2d(in_channels=64, out_channels=1, kernel_size=3, stride=1, padding=1, bias=False),
        self.c3 = nn.Conv2d(64, 1, 1, 1, bias=False)
        # x, self.Phiweight,z

    def forward(self, x, PhiWeight, z, q):
        #  z  (16, 102, 3, 3)   PhiWeight (102,1,32,32)    x  8, 1, 64, 64
        r = F.conv_transpose2d(z, PhiWeight, stride=32) + x+q  # 8*1*64*64

        r_out = self.getFactor(r)

        # 通过两次下采样
        out1 = self.down1(r_out)  # 第一次下采样 8 96 32 32
        out2 = self.down2(out1)  # 第二次下采样 8 128 16 16

        out3 = self.amc(out2) # 8 128 16 16

        out4 = self.cross_orthogonal(out3) # 8 128 16 16

        out5 = self.relu((self.cross_orthogonal(out4))+out4) # 8 128 16 16

        # out6 = (self.amc(out5))+out3  # 8 128 16 16

        out7 = self.up1(out5)+r_out # 上采样  8  96  32  32
        # out8 = self.up2(out7)+r_out # 上采样  8  64 64 64


        out,q = (self.sam(out7,r))
        out= (self.c3(out))



        return out , q




    def make_layer(self, block):
        layers = []
        layers.append(block())
        return nn.Sequential(*layers)


class tncs(torch.nn.Module):
    def __init__(self, LayerNo, sensing_rate):

        super(tncs, self).__init__()
        onelayer = []
        self.LayerNo = LayerNo
        self.patch_size = 32
        self.n_input = int(sensing_rate * 1024)  # 截尾

        for i in range(LayerNo):
            onelayer.append(BasicBlock(self.n_input))

            # 初始化权重  (102,1,32,32)
        self.Phiweight = nn.Parameter(
            init.xavier_normal_(torch.Tensor(self.n_input, 1, self.patch_size, self.patch_size)))

        # self.Phiweight_b = nn.Parameter(
        #     init.xavier_normal_(torch.Tensor(self.n, 1, 64, 64)))

        self.fcs = nn.ModuleList(onelayer)
        self.var_step = nn.Parameter(torch.tensor(0.1, dtype=torch.float32))

        #
        self.c1 = (nn.Conv2d(1, 64, 1, 1, bias=False))
        self.c2 = (nn.Conv2d(64, 64, 3, 1, 1, bias=False, groups=64))
        self.c3 = nn.Conv2d(64, 1, 1, 1, bias=False)
        self.relu = nn.ReLU(inplace=True)


    def forward(self, x):
        # y1=self.relu(self.c1(x))
        # y2 = self.relu(self.c2(y1))
        # y2 = (self.c3(y2))
        # y = F.conv2d(y2, self.Phiweight, stride=self.patch_size, padding=0, bias=None)

        y1 = F.conv2d(x, self.Phiweight, stride=self.patch_size, padding=0, bias=None)
        y2 = self.relu(self.c2(self.relu(self.c1(x))))
        y2 = (self.c3(y2))
        y = F.conv2d(y2, self.Phiweight, stride=self.patch_size, padding=0, bias=None)+y1

        # y = F.conv2d(x, self.Phiweight, stride=self.patch_size, padding=0, bias=None)
        PhiTb = F.conv_transpose2d(y, self.Phiweight, stride=self.patch_size)  # 反卷积 16, 1, 64, 64
        x = PhiTb
        q = 0
        x1 = F.conv2d(x, self.Phiweight, stride=self.patch_size, padding=0, bias=None)
        z = y - x1  # (8, 102, 2, 2)
        osger = 0
        for i in range(self.LayerNo):
            x, q = self.fcs[i](x, self.Phiweight, z, q)
            # if i<self.LayerNo-1:
            #    # osger =osger+self.var_step*z*r
            z = y - F.conv2d(x, self.Phiweight, stride=self.patch_size, padding=0, bias=None)
        x_final = x

        return x_final

    def weight_init(self, mean, std):
        for m in self._modules:
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, mode='fan_in')

#     def weight_init(self, mean=0.0, std=0.02):
#         for m in self._modules:
#             normal_init(self._modules[m], mean, std)
#
# def normal_init(m, mean, std):
#     if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d):
#         m.weight.data.normal_(mean, std)
#         m.bias.data.zero_()
