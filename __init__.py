from gym.envs.registration import registry, register, make, spec
from snake import SnakeEnv

register(
    id='Snake-v0',
    entry_point='snake:SnakeEnv',
    max_episode_steps=2000,
    reward_threshold=100.0,
)
