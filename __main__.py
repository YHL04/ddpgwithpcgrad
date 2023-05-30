

import datetime
import numpy as np

from agent import DDPG
from environment import Env


def main(epochs=100000, d_model=512, batch_size=64, device="cuda"):
    dt = f"{datetime.datetime.now().strftime('%Y_%m_%d_%H_%M')}.txt"
    file = open(f"logs/{dt}", "w")

    env = Env("Pendulum-v1")
    agent = DDPG(env.state_size, env.action_size, d_model, buffer_size=1000000, device=device)

    print("State size ", env.state_size)
    print("Action size ", env.action_size)

    for i in range(100):
        state, _ = env.reset()
        agent.reset()
        done = False
        while not done:
            action = np.random.uniform(-1, 1, (env.action_size,))
            next_state, reward, _, done, _ = env.step(action)
            agent.remember(state, action, reward, done)

            state = next_state

    for i in range(epochs):
        state, _ = env.reset()
        agent.reset()
        total_reward, total_loss = 0., 0.
        done = False
        while not done:
            action = agent.get_action(state, add_noise=True)
            next_state, reward, _, done, _ = env.step(action)
            agent.remember(state, action, reward, done)
            critic_loss, actor_loss = agent.update(batch_size)

            total_reward += reward
            total_loss += critic_loss
            state = next_state
            env.render()

        print(f"Episode {i} Total_reward {total_reward} Total_loss {total_loss}")
        file.write(f"{total_reward}\n")
        file.flush()


if __name__ == "__main__":
    main()

