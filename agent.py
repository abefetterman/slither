import gym
from snake import SnakeEnv
from models.dqn import DQN
from methods.batch_optimize import BatchOptimizer
from methods.policy import EpsPolicy
import torch

cuda = True
env = SnakeEnv()
model = DQN()
policy = EpsPolicy(model)
optimizer = BatchOptimizer(model, 100, 10000, cuda = cuda)
print_every = 10
total_frames = 0
total_reward = 0

FloatTensor = torch.FloatTensor
if (cuda):
    FloatTensor = torch.cuda.FloatTensor
    model = model.cuda()
# converts hwc to bchw:
tensorize = lambda t: FloatTensor(t.transpose((2,0,1))).unsqueeze(0)

for i in range(0,10000):
    if (i % print_every == 0):
        print('{0}: {1} frames, reward: {2:.2f}'.format(i, total_frames, total_reward))
        total_frames = 0
        total_reward = 0
    state_hwc = env.reset()
    state = tensorize(state_hwc)
    while True:
        action = policy.get(state, i)
        new_state_hwc, reward, done, _ = env.step(action)
        # env.render()
        total_reward += reward
        new_state = tensorize(new_state_hwc)
        optimizer.update(state, action, reward, new_state, done)
        state = new_state
        total_frames+=1
        if done:
            break
