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
BORDER_SIZE = 20

INIT_SNAKE_LENGTH = 3
BORDER_COLOR = (0,0,0)
SNAKE_COLOR = (0,255,0)
FOOD_COLOR = (255,0,0)

DIRECTIONS_DICT = {
    0: (0,0),
    1: (0,1),
    2: (0,-1),
    3: (-1,0),
    4: (1,0)
}

class SnakeEnv(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 5
    }

    def __init__(self):
        self.viewer = None
        self.snake = None

        self.action_space = spaces.Discrete(5) #none, u, d, l, r
        self.observation_space = spaces.Box(low=0, high=255, shape=(STATE_H, STATE_W, 3), dtype=np.uint8)

        self.plot_sparse = []
        self.state = np.zeros((STATE_H, STATE_W, 3), dtype=np.uint8)
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

        return self._update_state()

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

        return self._update_state(), reward, done, {}

    def _update_state(self):
        self.plot_sparse = [(x,y,SNAKE_COLOR) for x,y in self.snake]
        self.plot_sparse.append((self.food[0], self.food[1], FOOD_COLOR))
        self.state = self.state*0
        for x,y,c in self.plot_sparse:
            self.state[x,y]=c
        return self.state

    def render(self, mode='human'):

        if self.viewer is None:
            import rendering
            self.viewer = rendering.Viewer(WINDOW_W, WINDOW_H)
            border = rendering.Border(WINDOW_W, WINDOW_H, BORDER_SIZE, BORDER_COLOR)
            self.plotter = rendering.Plotter(
                WINDOW_W, WINDOW_H, BORDER_SIZE, STATE_W, STATE_H
            )
            self.viewer.add_geom(border)
            self.viewer.add_geom(self.plotter)
            self.transform = rendering.Transform()


        self.plotter.update_points(self.plot_sparse)
        self.viewer.render()


    def close(self):
        if self.viewer: self.viewer.close()


if __name__=="__main__":
    env = SnakeEnv()
    pyglet.clock.set_fps_limit(5)
    from pyglet.window import key
    global a
    a = 0
    def key_press(k, mod):
        global restart
        global a
        if k==key.R: restart = True
        if k==key.UP:    a=1
        if k==key.DOWN:  a=2
        if k==key.LEFT:  a=3
        if k==key.RIGHT: a=4
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
                if restart: break
    except (KeyboardInterrupt):
        env.close()
