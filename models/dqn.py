import torch.nn as nn
import torch.nn.functional as F
import math

def conv_size(h,w,conv):
    stride = conv.stride
    padding = conv.padding
    kernel_size = conv.kernel_size
    dilation = conv.dilation
    h_out = math.floor((h + 2*padding[0] - dilation[0] * (kernel_size[0]-1) -1)/stride[0] + 1)
    w_out = math.floor((w + 2*padding[1] - dilation[1] * (kernel_size[1]-1) -1)/stride[1] + 1)
    return (h_out, w_out)

class DQN(nn.Module):

    def __init__(self, h=15, w=15):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        conv1_h, conv1_w = conv_size(h,w,self.conv1)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=4, stride=2)
        conv2_h, conv2_w = conv_size(conv1_h, conv1_w, self.conv2)
        #print('h:{}, w:{}'.format(conv1_h, conv1_w))
        self.head = nn.Linear(conv2_h*conv2_w*32, 4)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        #print(x.size())
        return self.head(x.view(x.size(0), -1))
