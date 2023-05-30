

import gym


class Env:

    def __init__(self, env_name):
        self.env = gym.make(env_name, render_mode="rgb_array")

    @property
    def state_size(self):
        return self.env.observation_space.shape[0]

    @property
    def action_size(self):
        return self.env.action_space.shape[0]

    def reset(self):
        return self.env.reset()

    def step(self, action, bound=2):
        return self.env.step(action * bound)

    def render(self):
        self.env.render()
