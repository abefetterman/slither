import gym
from snake import SnakeEnv
from methods.scheduler import LinearSchedule
from models.dqn import DQN
import torch

env = SnakeEnv()
eps = LinearSchedule(1000, 1.0, .01)
model = DQN()

state = env.reset()
