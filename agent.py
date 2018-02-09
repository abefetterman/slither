import gym
from snake import SnakeEnv
from models.dqn import DQN
from methods.batch_optimize import BatchOptimizer
from methods.policy import EpsPolicy
import torch

cuda = True
env = SnakeEnv()
model = DQN(8,8)
policy = EpsPolicy(model)
optimizer = BatchOptimizer(model, 10, 10000, cuda = cuda)
print_every = 1000
total_frames = 0
total_reward = 0
total_loss = 0

FloatTensor = torch.FloatTensor
if (cuda):
    FloatTensor = torch.cuda.FloatTensor
    model = model.cuda()
# converts hwc to bchw:
tensorize = lambda t: FloatTensor(t.transpose((2,0,1))).unsqueeze(0)

for i in range(0,1000000):
    state_hwc = env.reset()
    state = tensorize(state_hwc)
    while True:
        action = policy.get(state, i)
        new_state_hwc, reward, done, _ = env.step(action)
        # env.render()
        total_reward += reward
        new_state = tensorize(new_state_hwc)
        total_loss += optimizer.update(state, action, reward, new_state, done)
        state = new_state
        total_frames+=1
        if (total_frames % print_every == 0):
            print('{0}: {1} frames, reward: {2:.2f}, loss: {3}'.format(i, total_frames, total_reward, total_loss))
            total_frames = 0
            total_reward = 0
            total_loss = 0

        if done:
            break
