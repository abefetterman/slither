import torch.nn as nn
import torch.nn.functional as F
import math
from methods.utils import conv_chain

class DQN(nn.Module):

    def __init__(self, h, w, batch_norm=False):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)

        conv_out_h, conv_out_w = conv_chain((h,w), [self.conv1, self.conv2, self.conv3])

        self.hidden = nn.Linear(conv_out_h*conv_out_w*64, 512)
        self.head = nn.Linear(512, 4)
        if batch_norm:
            self.bn1 = nn.BatchNorm2d(32)
            self.bn2 = nn.BatchNorm2d(64)
            self.bn3 = nn.BatchNorm2d(64)
        else:
            self.bn1 = lambda x: x
            self.bn2 = lambda x: x
            self.bn3 = lambda x: x

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.hidden(x.view(x.size(0), -1)))
        #print(x.size())
        return self.head(x)
