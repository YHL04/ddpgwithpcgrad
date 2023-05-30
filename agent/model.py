

import torch
import torch.nn as nn


class Body(nn.Module):

    def __init__(self, state_size, d_model):
        super(Body, self).__init__()

        self.body = nn.Sequential(
            nn.Linear(state_size, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.body(x)
        return x


class Actor(nn.Module):

    def __init__(self, state_size, action_size, d_model):
        super(Actor, self).__init__()

        self.actor = nn.Sequential(
            nn.Linear(state_size, d_model),
            nn.ReLU(),
            nn.Linear(d_model, action_size),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.actor(x)

        return x


class Critic(nn.Module):

    def __init__(self, state_size, action_size, d_model):
        super(Critic, self).__init__()

        self.linear1 = nn.Sequential(
            nn.Linear(state_size, d_model),
            nn.ReLU()
        )
        self.critic = nn.Sequential(
            nn.Linear(d_model + action_size, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, 1)
        )

    def forward(self, x, action):
        x = torch.concat([self.linear1(x), action], dim=-1)
        x = self.critic(x)

        return x


class Model(nn.Module):

    def __init__(self, state_size, action_size, d_model):
        super(Model, self).__init__()

        self.body = Body(state_size, d_model)
        self.actor = Actor(d_model, action_size, d_model)
        self.critic = Critic(d_model, action_size, d_model)

    def scale_gradients(self, scale):
        for x in self.parameters():
            if x.grad is not None:
                x.grad = x.grad * scale

    def body_forward(self, x):
        x = self.body(x)

        return x

    def actor_forward(self, x, body=True):
        if body:
            x = self.body(x)

        x = self.actor(x)

        return x

    def critic_forward(self, x, action=None, body=True):
        if body:
            x = self.body(x)

        if action is None:
            action = self.actor(x)

        x = self.critic(x, action)

        return x
