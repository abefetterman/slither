import gym
from snake import SnakeEnv
from models.dqn import DQN
from methods.batch_optimize import BatchOptimizer
from policies.eps import EpsPolicy
from policies.scheduler import ExpoSchedule
import torch

cuda = True
env = SnakeEnv(8,8)
model = DQN(8,8)
criterion = torch.nn.SmoothL1Loss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
schedule = ExpoSchedule(10000, 1.0, .1)
policy = EpsPolicy(model, schedule)

FloatTensor = torch.FloatTensor
if (cuda):
    FloatTensor = torch.cuda.FloatTensor
    model = model.cuda()
    criterion = criterion.cuda()

batch = BatchOptimizer(model, criterion, optimizer, 10, 10000, cuda = cuda)

print_every = 1000
total_frames = 0
total_reward = 0
total_loss = 0

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
        total_loss += batch.update(state, action, reward, new_state, done)
        state = new_state
        total_frames+=1
        if (total_frames % print_every == 0):
            print('{0}: {1} frames, reward: {2:.2f}, loss: {3}'.format(i, total_frames, total_reward, total_loss))
            total_frames = 0
            total_reward = 0
            total_loss = 0

        if done:
            break
