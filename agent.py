import gym
from snake import SnakeEnv
from methods.scheduler import LinearSchedule
from models.dqn import DQN
from methods.utils import HWC_to_BCHW
from methods.batch_optimize import BatchOptimizer
import torch

env = SnakeEnv()
eps = LinearSchedule(1000, 1.0, .01)
model = DQN()
optimizer = BatchOptimizer(model, 10, 1000)

state_hwc = env.reset()
for i in range(0,2):
    action = 1
    new_state_hwc, reward, done, _ = env.step(action)
    optimizer.update(HWC_to_BCHW(state_hwc), action, reward, HWC_to_BCHW(new_state_hwc), done)
    state_hwc = new_state_hwc
