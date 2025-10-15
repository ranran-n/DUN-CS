import torch

import TNCS.model
from thop import profile
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = TNCS.model.tncs(2,0.1)
model = model.to('cuda')
input = torch.randn(1, 1, 256, 256) #模型输入的形状,batch_size=1
input = input.to('cuda')
flops, params = profile(model, inputs=(input, ))
print('flops: ', flops, 'params: ', params)
print('flops: %.4f G, params: %.4f M' % (flops / 1048576.0/1024.0, params / 1e6))

