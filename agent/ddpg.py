

import torch
import torch.nn.functional as F
from torch.optim import Adam

import numpy as np
from copy import deepcopy

from .model import Model, Actor, Critic
from .memory import Memory
from .noise import OrnsteinUhlenbeckProcess


class DDPG:

    def __init__(self, state_size, action_size, d_model, buffer_size=1000000, device="cuda"):
        self.state_size = state_size
        self.action_size = action_size
        self.d_model = d_model
        self.buffer_size = buffer_size
        self.device = device

        # self.model = Model(state_size, action_size, d_model)
        # self.target_model = deepcopy(self.model)
        #
        # self.opt = Adam(self.model.parameters(), lr=1e-4)

        self.actor = Actor(state_size, action_size, d_model).to(device)
        self.critic = Critic(state_size, action_size, d_model).to(device)
        self.target_actor = Actor(state_size, action_size, d_model).to(device)
        self.target_critic = Critic(state_size, action_size, d_model).to(device)
        self.hard_update(self.target_actor, self.actor)
        self.hard_update(self.target_critic, self.critic)

        self.actor_opt = Adam(self.actor.parameters(), lr=1e-4)
        self.critic_opt = Adam(self.critic.parameters(), lr=1e-3)

        self.memory = Memory(state_size, action_size, buffer_size)
        self.ou_noise = OrnsteinUhlenbeckProcess(size=action_size,
                                                 theta=0.15,
                                                 mu=0.0,
                                                 sigma=0.2)

    def reset(self):
        self.ou_noise.reset_states()

    @torch.no_grad()
    def get_action(self, state, add_noise=False):
        state = torch.tensor(state, dtype=torch.float32).view(1, self.state_size).to(self.device)
        action = self.actor(state)
        action = action.view(self.action_size,).cpu().numpy()

        if add_noise:
            action = action + self.ou_noise.sample()
            action = np.clip(action, -1, 1)

        return action

    def remember(self, state, action, reward, done):
        self.memory.add_experience(state, action, reward, done)

    def update(self, batch_size, gamma=0.99):
        state, action, reward, next_state, done = self.memory.get_minibatch(batch_size)

        state = torch.tensor(np.array(state), dtype=torch.float32, device=self.device).view(batch_size, self.state_size)
        action = torch.tensor(action, dtype=torch.float32, device=self.device).view(batch_size, self.action_size)
        reward = torch.tensor(reward, dtype=torch.float32, device=self.device).view(batch_size, 1)
        next_state = torch.tensor(np.array(next_state), dtype=torch.float32, device=self.device).view(batch_size, self.state_size)
        done = torch.tensor(done, dtype=torch.float32, device=self.device).view(batch_size, 1)

        with torch.no_grad():
            next_q_values = self.critic(next_state, self.actor(next_state))
            target = reward + gamma * next_q_values  # * (1 - done)

        # critic training
        self.critic.zero_grad()

        expected = self.critic(state, action)
        critic_loss = F.huber_loss(expected, target)
        critic_loss = critic_loss.mean()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 0.01)
        self.critic_opt.step()

        # actor training
        self.actor.zero_grad()

        actor_loss = -self.critic(state, self.actor(state))
        actor_loss = actor_loss.mean()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 0.01)
        self.actor_opt.step()

        self.soft_update(self.target_critic, self.critic, tau=0.001)
        self.soft_update(self.target_actor, self.actor, tau=0.001)

        return critic_loss.item(), actor_loss.item()

    @staticmethod
    def hard_update(target, source):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(param.data)

    @staticmethod
    def soft_update(target, source, tau):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(
                target_param.data * (1.0 - tau) + param.data * tau
            )

