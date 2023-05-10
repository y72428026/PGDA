import torch
a = torch.ones([99,9999,9999]).to('cuda:4')
b = torch.ones([99,9999,9999]).to('cuda:5')
c = torch.ones([99,9999,9999]).to('cuda:6')
input('fixgpu')
