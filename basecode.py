#https://github.com/EricSteinberger/PokerRL

#https://pypi.org/project/pokerenv/1.0.1/

#gymnasium using top
from collections import defaultdict
from pettingzoo.classic import texas_holdem_v4
import numpy as np
import gym
from tqdm import tqdm

env = texas_holdem_v4.env(render_mode="human")
env.reset(seed=42)

class Pokerplayer:
    def __init__(
            self,
            learning_rate: float,
            initial_epsilon: float,
            epsilon_decay: float,
            final_epsilon: float,
            discount_factor: float = 0.95
    ):
        
        self.q_values = defaultdict(lambda: np.zeros(env.action_space.n))
        self.lr = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = initial_epsilon
        self.epsilon_decay = epsilon_decay
        self.final_epsilon = final_epsilon
        self.training_error = []

    def get_action(self, obs: tuple[int, int, bool]) -> int:
        """
        Returns the best action with probability (1 - epsilon)
        otherwise a random action with probability epsilon to ensure exploration.
        """
        # with probability epsilon return a random action to explore the environment
        if np.random.random() < self.epsilon:
            return env.action_space(agent).sample(mask)

        # with probability (1 - epsilon) act greedily (exploit)
        else:
            return int(np.argmax(self.q_values[obs]))
        
    def update(
        self,
        obs: tuple[int, int, bool],
        action: int,
        reward: float,
        terminated: bool,
        next_obs: tuple[int, int, bool],
    ):
        future_q_value = (not terminated) * np.max(self.q_values[next_obs])
        temporal_difference = (
            reward + self.discount_factor * future_q_value - self.q_values[obs][action]
        )

        self.q_values[obs][action] = (
            self.q_values[obs][action] + self.lr * temporal_difference
        )
        self.training_error.append(temporal_difference)

    
    def decay_epsilon(self):
        self.epsilon = max(self.final_epsilon, self.epsilon - epsilon_decay)


# hyperparameters
learning_rate = 0.01
n_episodes = 100_000
start_epsilon = 1.0
epsilon_decay = start_epsilon / (n_episodes / 2)  # reduce the exploration over time
final_epsilon = 0.1

agentparam = Pokerplayer(
    learning_rate=learning_rate,
    initial_epsilon=start_epsilon,
    epsilon_decay=epsilon_decay,
    final_epsilon=final_epsilon,
)

env = gym.wrappers.RecordEpisodeStatistics(env, deque_size=1000)
"""
for episode in tqdm(range(n_episodes)):
    env.reset()
    done = False
    action = 1
    # play one episode
    while not done:
        obs, reward, terminated, truncated, info = env.step(action)
        action = agent.get_action(obs)
        # update the agent
        agent.update(obs, action, reward, terminated, obs)

        # update if the environment is done and the current obs
        done = terminated or truncated

    agent.decay_epsilon()
"""
for i in range(10):
    done = False
    while not done:
        env.reset()
        preobs, reward, termination, truncation, info = env.last()
        for agent in env.agent_iter():
            observation, reward, termination, truncation, info = env.last()
            if termination or truncation:
                action = None
            else:
                mask = observation["action_mask"]
                # this is where you would insert your policy
                action = agentparam.get_action(observation)
                agentparam.update(observation, action, reward, termination, preobs)
                action = env.action_space(agent).sample(mask)
            env.step(action)
        done = True
        agent.decay_epsilon()
    env.close()
