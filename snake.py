import math
import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
import pyglet

STATE_W = 30
STATE_H = 30
VIDEO_W = 300
VIDEO_H = 300
WINDOW_W = 600
WINDOW_H = 600

SCALE = WINDOW_W/STATE_W

INIT_SNAKE_LENGTH = 3

DIRECTIONS_DICT = {
    0: (0,0),
    1: (0,1),
    2: (0,-1),
    3: (-1,0),
    4: (1,0)
}

class SnakeEnv(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array', 'state_pixels'],
        'video.frames_per_second': 30
    }

    def __init__(self):
        self.viewer = None
        self.snake = None

        self.action_space = spaces.Discrete(5) #none, u, d, l, r
        self.observation_space = spaces.Box(low=0, high=255, shape=(STATE_H, STATE_W, 3), dtype=np.uint8)

        self.seed()
        self.reset()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self):
        length = INIT_SNAKE_LENGTH
        init_x = self.np_random.randint(STATE_W - 2*length) + length
        init_y = self.np_random.randint(STATE_H - 2*length) + length
        self.snake_dir = 1+self.np_random.randint(4)
        dx, dy = DIRECTIONS_DICT[self.snake_dir]

        self.snake = [(init_x, init_y)]
        self.snake_length = length

        self._place_food()

        #return self._render()[0]

    def _place_food(self):
        self.food = self.snake[0]
        while self.food in self.snake:
            food_x = self.np_random.randint(STATE_W)
            food_y = self.np_random.randint(STATE_H)
            self.food = (food_x, food_y)

    def step(self, action):
        assert self.action_space.contains(action), "%r (%s) invalid" % (action, type(action))

        if (action > 0):
            self.snake_dir = action

        cx,cy = self.snake[0]
        dx, dy = DIRECTIONS_DICT[self.snake_dir]

        head = (cx + dx, cy + dy)

        done = False
        reward = 0.0

        # check for wall hit
        if (head[0] < 0 or head[0] >= STATE_W):
            done = True
        if (head[1] < 0 or head[1] >= STATE_H):
            done = True

        # check for self hit
        if head in self.snake:
            done = True

        if (not done):
            self.snake.insert(0, head)

            # eat food
            if (head == self.food):
                reward = 1.0
                self.snake_length += 2
                self._place_food()

            # maintain length
            while (len(self.snake) > self.snake_length):
                self.snake.pop()


        self.state = self._render()[0]
        return np.array(self.state), reward, done, {}


    def render(self, mode='human'):

        if self.viewer is None:
            import rendering
            self.viewer = rendering.Viewer(WINDOW_W, WINDOW_H)
            label = pyglet.text.Label('Hello, world',
                          font_name='Times New Roman',
                          font_size=36,
                          x=WINDOW_W//2, y=WINDOW_H//2,
                          color=(0,0,0,100),
                          anchor_x='center', anchor_y='center')
            self.viewer.add_geom(label)
            self.transform = rendering.Transform()
        self.viewer.render()


    def close(self):
        if self.viewer: self.viewer.close()


if __name__=="__main__":
    env = SnakeEnv()
    try:
        while True:
            env.render()
    except (KeyboardInterrupt):
        env.close()
