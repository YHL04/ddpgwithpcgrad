

import numpy as np
import random


class Memory:

    def __init__(self, state_size, action_size, buffer_size):
        self.state_size = state_size
        self.action_size = action_size
        self.buffer_size = buffer_size
        self.count = 0
        self.current = 0

        self.frames = np.empty((self.buffer_size, self.state_size), dtype=np.float32)
        self.actions = np.empty((self.buffer_size, self.action_size), dtype=np.float32)
        self.rewards = np.empty((self.buffer_size, 1), dtype=np.float32)
        self.terminal = np.empty((self.buffer_size, 1), dtype=bool)

    def add_experience(self, frame, action, reward, terminal):
        self.frames[self.current] = frame
        self.actions[self.current] = action
        self.rewards[self.current] = reward
        self.terminal[self.current] = terminal

        self.count = max(self.count, self.current+1)
        self.current = (self.current + 1) % self.buffer_size

    def get_minibatch(self, batch_size):

        # Get a list of valid indices
        indices = []
        for i in range(batch_size):
            while True:
                index = random.randint(0, self.count - 1)

                if index == self.current - 1:
                    continue
                if self.terminal[index].any():
                    continue
                break
            indices.append(index)

        # Retrieve states from memory
        states = []
        new_states = []
        for idx in indices:
            states.append(self.frames[idx, ...])
            new_states.append(self.frames[idx+1, ...])

        return states, self.actions[indices], self.rewards[indices], new_states, self.terminal[indices]

