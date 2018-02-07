import numpy as np

import torch
from torch.autograd import Variable

def wrap(arr):
    # return np.array(arr)
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
        # IDEA: Maybe change to numpy buffers so we can just
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
            obs_t, action, reward, obs_tp1, done = self._storage[i]
            obses_t.append(np.array(obs_t, copy=False))
            actions.append(np.array(action, copy=False))
            rewards.append(reward)
            obses_tp1.append(np.array(obs_tp1, copy=False))
            dones.append(done)
        return wrap(obses_t), wrap(actions), wrap(rewards), wrap(obses_tp1), wrap(dones)

    def sample(self, batch_size):
        idxes = np.random.randint(0, len(self._storage) - 1, batch_size)
        return self._encode_sample(idxes)
