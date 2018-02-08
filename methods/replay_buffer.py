import numpy as np

import torch
from torch.autograd import Variable

def wrap(arr):
    return Variable(arr)

class ReplayBuffer(object):
    def __init__(self, size, cuda=False):
        self._size = size
        self._next_idx = 0
        self._buffer_size = None
        self.FloatTensor = torch.FloatTensor
        self.LongTensor = torch.LongTensor
        if cuda:
            self.FloatTensor = torch.cuda.FloatTensor
            self.LongTensor = torch.cuda.LongTensor

    def __len__(self):
        return self._buffer_size

    def _make_tensors(self, obs_size, action_size):
        # allocate buffers once
        obses_size = [self._size] + list(obs_size)[1:]
        actions_size = [self._size] + list(action_size)
        self.obses_t = self.FloatTensor(*obses_size).zero_()
        self.obses_tp1 = self.FloatTensor(*obses_size).zero_()
        self.actions = self.LongTensor(*actions_size).zero_()
        self.rewards = self.FloatTensor(self._size, 1).zero_()
        self.dones = self.FloatTensor(self._size, 1).zero_()
        self._buffer_size = 0

    def add(self, obs_t, action, reward, obs_tp1, done):
        if (self._buffer_size is None):
            obs_size = obs_t.size()
            action_size = getattr(action, 'size', lambda: [1])()
            self._make_tensors(obs_size, action_size)

        i = self._next_idx
        self.obses_t[i] = self.FloatTensor(obs_t)
        self.obses_tp1[i] = self.FloatTensor(obs_tp1)
        self.actions[i] = self.LongTensor([action])
        self.rewards[i] = self.FloatTensor([reward])
        self.dones[i] = self.LongTensor([done])

        # circular buffer
        self._next_idx = (i + 1) % self._size
        self._buffer_size = min(self._size, i + 1)

    def _encode_sample(self, idxes):
        obses_t = torch.index_select(self.obses_t, 0, idxes)
        obses_tp1 = torch.index_select(self.obses_tp1, 0, idxes)
        actions = torch.index_select(self.actions, 0, idxes)
        rewards = torch.index_select(self.rewards, 0, idxes)
        dones = torch.index_select(self.dones, 0, idxes)
        return wrap(obses_t), wrap(actions), wrap(rewards), wrap(obses_tp1), wrap(dones)

    def sample(self, batch_size):
        if self._buffer_size > batch_size:
            idxes = np.random.randint(0, self._buffer_size - 1, batch_size)
        else:
            idxes = np.arange(self._buffer_size)
        return self._encode_sample(self.LongTensor(idxes))
