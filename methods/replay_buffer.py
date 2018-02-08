import numpy as np

import torch
from torch.autograd import Variable

def wrap(arr):
    return Variable(torch.cat(arr))

class ReplayBuffer(object):
    def __init__(self, size):
        self._storage = []
        self._maxsize = size
        self._next_idx = 0

    def __len__(self):
        return len(self._storage)

    def add(self, obs_t, action, reward, obs_tp1, done):
        data = (obs_t, action, reward, obs_tp1, done)
        # IDEA: Maybe change to giant tensors so we can just
        # slice out the elements for sampling

        if self._next_idx >= len(self._storage):
            self._storage.append(data)
        else:
            self._storage[self._next_idx]

        # circular buffer
        self._next_idx = (self._next_idx + 1) % self._maxsize

    def _encode_sample(self, idxes):
        obses_t, actions, rewards, obses_tp1, dones = [], [], [], [], []
        for i in idxes:
            data = self._storage[i]
            obs_t, action, reward, obs_tp1, done = data
            obses_t.append(torch.FloatTensor(obs_t))
            actions.append(torch.LongTensor([action]))
            rewards.append(torch.FloatTensor([reward]))
            obses_tp1.append(torch.FloatTensor(obs_tp1))
            dones.append(torch.LongTensor([done]))
        return wrap(obses_t), wrap(actions), wrap(rewards), wrap(obses_tp1), wrap(dones)

    def sample(self, batch_size):
        if len(self._storage) > batch_size:
            idxes = np.random.randint(0, len(self._storage) - 1, batch_size)
        else:
            idxes = np.arange(len(self._storage))
        return self._encode_sample(idxes)
