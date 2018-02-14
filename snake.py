import math
import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
import scipy.ndimage as ndi
import pyglet

INIT_SNAKE_LENGTH = 3
BORDER_COLOR = (0,0,0)
SNAKE_COLOR = (0,200,0)
SNAKE_HEAD_COLOR = (0,255,0)
FOOD_COLOR = (255,0,0)

DIRECTIONS_DICT = {
    0: (0,1),
    1: (0,-1),
    2: (-1,0),
    3: (1,0)
}

class SnakeEnv(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 5
    }

    def __init__(self, dim=8, zoom=1, window=300, rewards=(1.0,0.0,-1.0)):
        self.viewer = None
        self.snake = None
        self.dim = dim
        self.zoom = zoom
        self.window = window
        self.rewards = rewards # tuple: (eat, step, die)

        self.action_space = spaces.Discrete(4) #u, d, l, r
        self.observation_space = spaces.Box(low=0, high=255, shape=(dim, dim, 3), dtype=np.uint8)

        self.plot_sparse = []
        self.state = np.zeros((dim+2, dim+2, 3), dtype=np.uint8)
        self.seed()
        self.reset()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self):
        length = INIT_SNAKE_LENGTH
        init_x = self.np_random.randint(self.dim - 2*length) + length
        init_y = self.np_random.randint(self.dim - 2*length) + length

        self.snake = [(init_x, init_y)]
        self.snake_length = length

        self._place_food()

        return self._update_state()

    def _place_food(self):
        self.food = self.snake[0]
        while self.food in self.snake:
            food_x = self.np_random.randint(self.dim)
            food_y = self.np_random.randint(self.dim)
            self.food = (food_x, food_y)

    def step(self, action):
        assert self.action_space.contains(action), "%r (%s) invalid" % (action, type(action))

        cx,cy = self.snake[0]
        dx, dy = DIRECTIONS_DICT[action]

        head = (cx + dx, cy + dy)

        done = False
        reward = self.rewards[1]

        # check for wall hit
        if (head[0] < 0 or head[0] >= self.dim):
            done = True
        if (head[1] < 0 or head[1] >= self.dim):
            done = True

        # check for self hit
        if head in self.snake:
            done = True

        if (not done):
            self.snake.insert(0, head)

            # eat food
            if (head == self.food):
                reward = self.rewards[0]
                self.snake_length += 2
                self._place_food()

            # maintain length
            while (len(self.snake) > self.snake_length):
                self.snake.pop()
        else:
            reward = self.rewards[2]

        return self._update_state(), reward, done, {}

    def _snake_color(self,x,y):
        if (x,y)==self.snake[0]: return SNAKE_HEAD_COLOR
        return SNAKE_COLOR

    def _update_state(self):
        self.plot_sparse = [(x,y,self._snake_color(x,y)) for x,y in self.snake]
        self.plot_sparse.append((self.food[0], self.food[1], FOOD_COLOR))
        self.state = self.state*0
        self.state[1:-1,1:-1,:]=255
        for x,y,c in self.plot_sparse:
            self.state[x+1,y+1]=c

        return ndi.zoom(self.state, (self.zoom, self.zoom, 1), order=0)

    def render(self, mode='human'):

        if self.viewer is None:
            import rendering
            block_size = self.window // (self.dim + 1)
            border_size = (self.window - self.dim * block_size) // 2

            self.viewer = rendering.Viewer(self.window, self.window)
            border = rendering.Border(self.window, self.window, border_size, BORDER_COLOR)
            self.plotter = rendering.Plotter(
                self.window, self.window, border_size, self.dim, self.dim
            )
            self.viewer.add_geom(border)
            self.viewer.add_geom(self.plotter)
            self.transform = rendering.Transform()

        return_rgb_array = (mode == 'rgb_array')
        self.plotter.update_points(self.plot_sparse)
        return self.viewer.render(return_rgb_array)

    def close(self):
        if self.viewer: self.viewer.close()


if __name__=="__main__":
    env = SnakeEnv(dim=12)
    pyglet.clock.set_fps_limit(5)
    from pyglet.window import key
    global a
    a = np.random.randint(4)
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
    try:
        while True:
            env.reset()
            total_reward = 0.0
            steps = 0
            restart = False
            while True:
                pyglet.clock.tick()
                s,r,done,info = env.step(a)
                total_reward += r
                steps += 1
                env.render()
                if (steps % 20 == 0):
                    print('step {0} score: {1:.2f}'.format(steps, total_reward))
                if (restart or done): break
    except (KeyboardInterrupt):
        env.close()
