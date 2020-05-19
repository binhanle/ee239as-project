import torch.nn as nn
from nn_models import DQN
from memory import ReplayMemory
import gym

env = gym.make("Breakout-v0")
n_actions = env.action_space.n
print(n_actions)
print(env.action_space)
print(env.observation_space)

memory = ReplayMemory(10000, (4, 84, 84))
dqn = DQN(n_actions)