import numpy as np
import torch
from methods.replay_buffer import ReplayBuffer

class BatchOptimizer(object):
    def __init__(self, model, batch_size, buffer_size, cuda=False):
        self.model = model
        self.batch_size = batch_size
        self.buffer = ReplayBuffer(buffer_size, cuda=cuda)
        self.gamma = 0.9
        self.criterion = torch.nn.SmoothL1Loss()
        if cuda:
            self.criterion = self.criterion.cuda()
        self.optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        self.cuda = cuda

    def update(self, obs_t, action, reward, obs_tp1, done):
        self.buffer.add(obs_t, action, reward, obs_tp1, done)
        obs_ts, actions, rewards, obs_tp1s, dones = self.buffer.sample(self.batch_size)

        # compute next state values
        self.model.eval()
        obs_tp1s.volatile = True
        tp1_actions = self.model(obs_tp1s)
        tp1_values = torch.max(tp1_actions)

        # now go through each observation and get expected values
        self.model.train()
        predictions = self.model(obs_ts)
        state_action_values = predictions.gather(1, actions.view(-1,1))

        # set target and find the loss
        target_values = rewards + self.gamma * tp1_values * (1.0 - dones)
        loss = self.criterion(state_action_values, target_values)

        # optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # return loss value
        return loss.data[0]
