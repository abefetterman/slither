from torch.autograd import Variable
from torch import FloatTensor

def HWC_to_BCHW(arr):
    return FloatTensor(arr.transpose((2,0,1))).unsqueeze(0)
