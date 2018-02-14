import argparse
import pyglet
import torch
from models.dqn import DQN
from policies.pure import PurePolicy
from snake import SnakeEnv

parser = argparse.ArgumentParser(description='Play Snake with model')
parser.add_argument('filename', type=argparse.FileType('rb'), nargs='?', default=None,
                    help='filename for model')
parser.add_argument('--dim', type=int, nargs='?', default=12,
                    help='States per dimension')
parser.add_argument('--zoom', type=int, nargs='?', default=8,
                    help='Zoom per dimension')
parser.add_argument('--fps', type=int, nargs='?', default=10,
                    help='Frames per second')

tensorize = lambda t: torch.FloatTensor(t.transpose((2,0,1)).copy()).unsqueeze(0)

if __name__=="__main__":
    args = parser.parse_args()
    env = SnakeEnv(args.dim, zoom=args.zoom)
    pyglet.clock.set_fps_limit(args.fps)
    global a, policy
    if (args.filename is None):
        a = np.random.randint(4)
        from pyglet.window import key
        def key_press(k, mod):
            global restart
            global a
            if k==key.R: restart = True
            if k==key.UP:    a=0
            if k==key.DOWN:  a=1
            if k==key.LEFT:  a=2
            if k==key.RIGHT: a=3
        env.render()
        env.viewer.window.on_key_press = key_press
    else:
        size = (args.dim + 2)*args.zoom
        model = DQN(size, size, batch_norm=True)
        model.load_state_dict(torch.load(args.filename))
        policy=PurePolicy(model)
    try:
        while True:
            state = env.reset()
            total_reward = 0.0
            steps = 0
            restart = False
            while True:
                pyglet.clock.tick()
                if (policy is not None):
                    state_ten = tensorize(state)
                    a=policy.get(state_ten)
                state,r,done,info = env.step(a)
                total_reward += r
                steps += 1
                env.render()
                if (steps % 20 == 0):
                    print('step {0} score: {1:.2f}'.format(steps, total_reward))
                if (restart or done): break
    except (KeyboardInterrupt):
        env.close()
