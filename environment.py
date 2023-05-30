

import gym


class Env:

    def __init__(self, env_name):
        self.env = gym.make(env_name)

    @property
    def state_size(self):
        return self.env.observation_space.shape[0]

    @property
    def action_size(self):
        try:
            return self.env.action_space.n
        except:
            return 1

    def reset(self):
        return self.env.reset()

    def step(self, action):
        return self.env.step(action)

    def render(self):
        self.env.render()
