

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

    def actor_forward(self, x):
        x = self.actor(self.body(x))

        return x

    def critic_forward(self, x, freeze_critic=False):
        rep = self.body(x)
        action = self.actor(rep)

        if freeze_critic:
            with torch.no_grad():
                x = self.critic(rep, action)
        else:
            x = self.critic(rep, self.actor(rep))

        return x
