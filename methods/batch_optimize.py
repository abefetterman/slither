import numpy as np
import torch
from replay_buffer import ReplayBuffer

class BatchOptimizer(object):
    def __init__(self, model, batch_size, buffer_size):
        self.model = model
        self.batch_size = batch_size
        self.buffer = ReplayBuffer(buffer_size)
        self.gamma = 0.99
        self.criterion = torch.nn.functional.smooth_l1_loss
        self.optimizer = torch.optim.SGD(model, lr=0.01)

    def update(self, obs_t, action, reward, obs_tp1, done):
        self.buffer.add(obs_t, action, reward, obs_tp1, done)
        obs_ts, actions, rewards, obs_tp1s, dones = self.buffer.sample(self.batch_size)

        # compute next state values
        model.eval()
        tp1_actions = model(obs_tp1s)
        tp1_values = np.max(tp1_actions, axis=1)

        # now go through each observation and get expected values
        model.train()
        state_action_values = model(obs_ts).gather(1, actions)

        # set target and find the loss
        target_values = rewards + self.gamma * tp1_values
        loss = self.criterion(state_action_values, target_values)

        # optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # return loss value
        return loss.data[0]
