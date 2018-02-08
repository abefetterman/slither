import torch

def HWC_to_BCHW(arr, cuda=False):
    Tensor = torch.FloatTensor
    if cuda:
        Tensor = torch.cuda.FloatTensor
    return Tensor(arr.transpose((2,0,1))).unsqueeze(0)
