import gym
from snake import SnakeEnv
from models.dqn import DQN
from methods.utils import HWC_to_BCHW
from methods.batch_optimize import BatchOptimizer
from methods.policy import EpsPolicy
import torch

env = SnakeEnv()
model = DQN()
policy = EpsPolicy(model)
optimizer = BatchOptimizer(model, 100, 10000)
print_every = 10
total_frames = 0
total_reward = 0

for i in range(0,10000):
    if (i % print_every == 0):
        print('{0}: {1} frames, reward: {2:.2f}'.format(i, total_frames, total_reward))
        total_frames = 0
        total_reward = 0
    state_hwc = env.reset()
    state = HWC_to_BCHW(state_hwc)
    while True:
        action = policy.get(state, i)
        new_state_hwc, reward, done, _ = env.step(action)
        env.render()
        total_reward += reward + 1
        new_state = HWC_to_BCHW(new_state_hwc)
        optimizer.update(state, action, reward, new_state, done)
        state = new_state
        total_frames+=1
        if done:
            break
