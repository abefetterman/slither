import math
import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
import pyglet

WINDOW_DIM = 84
BLOCK_SIZE = 30
BORDER_SIZE = 30

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

    def __init__(self, h=8, w=8):
        self.viewer = None
        self.snake = None
        self.state_h = h
        self.state_w = w

        self.action_space = spaces.Discrete(4) #u, d, l, r
        self.observation_space = spaces.Box(low=0, high=255, shape=(h, w, 3), dtype=np.uint8)

        self.plot_sparse = []
        self.state = np.zeros((h, w, 3), dtype=np.uint8)
        self.seed()
        self.reset()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self):
        length = INIT_SNAKE_LENGTH
        init_x = self.np_random.randint(self.state_w - 2*length) + length
        init_y = self.np_random.randint(self.state_h - 2*length) + length

        self.snake = [(init_x, init_y)]
        self.snake_length = length

        self._place_food()

        return self._update_state()

    def _place_food(self):
        self.food = self.snake[0]
        while self.food in self.snake:
            food_x = self.np_random.randint(self.state_w)
            food_y = self.np_random.randint(self.state_h)
            self.food = (food_x, food_y)

    def step(self, action):
        assert self.action_space.contains(action), "%r (%s) invalid" % (action, type(action))

        cx,cy = self.snake[0]
        dx, dy = DIRECTIONS_DICT[action]

        head = (cx + dx, cy + dy)

        done = False
        reward = 0.0

        # check for wall hit
        if (head[0] < 0 or head[0] >= self.state_w):
            done = True
        if (head[1] < 0 or head[1] >= self.state_h):
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
        else:
            reward = -1.0

        return self._update_state(), reward, done, {}

    def _snake_color(self,x,y):
        if (x,y)==self.snake[0]: return SNAKE_HEAD_COLOR
        return SNAKE_COLOR
    def _update_state(self):
        self.plot_sparse = [(x,y,self._snake_color(x,y)) for x,y in self.snake]
        self.plot_sparse.append((self.food[0], self.food[1], FOOD_COLOR))
        self.state = self.state*0
        for x,y,c in self.plot_sparse:
            self.state[x,y]=c
        return self.state

    def render(self, mode='human'):

        if self.viewer is None:
            import rendering
            window_w = self.state_w*BLOCK_SIZE + 2*BORDER_SIZE
            window_h = self.state_h*BLOCK_SIZE + 2*BORDER_SIZE
            state_dim = max(self.state_h, self.state_w)
            block_size = WINDOW_DIM // (state_dim + 1)
            border_size = (WINDOW_DIM - state_dim * block_size) // 2

            self.viewer = rendering.Viewer(WINDOW_DIM, WINDOW_DIM)
            border = rendering.Border(WINDOW_DIM, WINDOW_DIM, border_size, BORDER_COLOR)
            self.plotter = rendering.Plotter(
                WINDOW_DIM, WINDOW_DIM, border_size, self.state_w, self.state_h
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
    env = SnakeEnv()
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
