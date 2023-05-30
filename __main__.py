

import numpy as np

from agent import DDPG
from environment import Env


def main(epochs=100000, d_model=256, batch_size=32):
    env = Env("Pendulum-v1")
    agent = DDPG(env.state_size, env.action_size, d_model)

    for i in range(100):
        state = env.reset()
        done = False
        while not done:
            action = np.random.uniform(-2, 2, (1,))
            next_state, reward, done, _ = env.step(action)
            agent.remember(state, action, reward, done)

            state = next_state

    for i in range(epochs):
        state = env.reset()
        total_reward, total_loss = 0., 0.
        done = False
        while not done:
            action = agent.get_action(state, add_noise=False)
            next_state, reward, done, _ = env.step(action)
            agent.remember(state, action, reward, done)
            critic_loss, actor_loss = agent.update(batch_size)

            total_reward += reward
            total_loss += critic_loss
            state = next_state
            env.render()

        print(f"Episode {i} Total_reward {total_reward} Total_loss {total_loss}")


if __name__ == "__main__":
    main()
