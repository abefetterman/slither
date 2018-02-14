import gym
from snake import SnakeEnv
from models.dqn import DQN
from methods.batch_optimize import BatchOptimizer
from policies.eps import EpsPolicy
from policies.scheduler import ExpoSchedule
import torch
import os

cuda = torch.cuda.is_available()
dim = 12
zoom = 8
env = SnakeEnv(dim=dim, zoom=zoom)
model = DQN((dim + 2)*zoom,(dim + 2)*zoom, batch_norm=True)
criterion = torch.nn.SmoothL1Loss()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
schedule = ExpoSchedule(10000, 1.0, .5)
policy = EpsPolicy(model, schedule)

FloatTensor = torch.FloatTensor
if (cuda):
    FloatTensor = torch.cuda.FloatTensor
    model = model.cuda()
    criterion = criterion.cuda()

batch = BatchOptimizer(model, criterion, optimizer, 100, 10000, cuda = cuda)

save_dir = 'backups/'
print_every = 1000 if cuda else 10
save_every = 10000

if not os.path.exists(save_dir): os.mkdir(save_dir)

total_frames = 0
total_reward = 0
total_loss = 0
last_game = 0

# converts hwc to bchw:
tensorize = lambda t: FloatTensor(t.transpose((2,0,1)).copy()).unsqueeze(0)

try:
    for i in range(0,100000):
        state_hwc = env.reset()
        state = tensorize(state_hwc)
        while True:
            action = policy.get(state, i)
            new_state_hwc, reward, done, _ = env.step(action)
            total_reward += reward
            new_state = tensorize(state_hwc)
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
        if (i % save_every == 0):
            filename = '{}checkpoint{}.pt'.format(save_dir, i // save_every)
            torch.save(model, filename)
    torch.save('{}final.pt'.format(save_dir))
except KeyboardInterrupt:
    torch.save(model, '{}interrupted.pt'.format(save_dir))
