import gym
from snake import SnakeEnv
from models.dqn import DQN
from methods.batch_optimize import BatchOptimizer
from policies.eps import EpsPolicy
from policies.scheduler import ExpoSchedule
import torch

cuda = True
env = SnakeEnv(5,5)
model = DQN(5,5)
criterion = torch.nn.SmoothL1Loss()
optimizer = torch.optim.Adam(model.parameters(), lr=2e-7)
schedule = ExpoSchedule(1000, 1.0, .1)
policy = EpsPolicy(model, schedule)

FloatTensor = torch.FloatTensor
if (cuda):
    FloatTensor = torch.cuda.FloatTensor
    model = model.cuda()
    criterion = criterion.cuda()

batch = BatchOptimizer(model, criterion, optimizer, 30, 10000, cuda = cuda)

print_every = 1000
total_frames = 0
total_reward = 0
total_loss = 0
last_game = 0

# converts hwc to bchw:
tensorize = lambda t: FloatTensor(t.transpose((2,0,1))).unsqueeze(0)

for i in range(0,100000):
    state_hwc = env.reset()
    state = tensorize(state_hwc)
    while True:
        action = policy.get(state, i)
        new_state_hwc, reward, done, _ = env.step(action)
        # arr = env.render(mode='rgb_array')
        total_reward += reward
        new_state = tensorize(new_state_hwc)
        total_loss += batch.update(state, action, reward, new_state, done)
        state = new_state
        total_frames+=1
        if (total_frames % print_every == 0):
            print('{0:.0f}k/{1}: games: {2}, reward: {3:.2f}, loss: {4:.2f}'.format(total_frames/1000, i, i - last_game, total_reward, total_loss))
            total_reward = 0
            total_loss = 0
            last_game = i

        if done:
            break
