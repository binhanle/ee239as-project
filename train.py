from nn_models import DQN
from memory import ReplayMemory
from agents import Agent
import torch
import numpy as np
import gym

env = gym.make("Breakout-v0")

n_actions = env.action_space.n
print(n_actions)
print(env.action_space)
print(env.observation_space)
original_shape = env.observation_space.shape

MEMORY_SIZE = 1000
STATE_SHAPE = (original_shape[2], original_shape[0], original_shape[1]) 
LR = 0.01
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(STATE_SHAPE)

mem_buffer = ReplayMemory(MEMORY_SIZE, STATE_SHAPE)
dqn_online = DQN(n_actions, STATE_SHAPE)
dqn_target = DQN(n_actions, STATE_SHAPE)
optimizer = torch.optim.RMSprop(dqn_online.parameters(), lr=LR)
loss_fn = torch.nn.MSELoss()
agent = Agent(device, mem_buffer, dqn_online, dqn_target, optimizer, loss_fn)

def preprocess_state(state):
    return np.transpose(state, (2, 0, 1))

num_episodes = 3
for i_episode in range(num_episodes):
  print("Running episode:", i_episode)
  score = 0.0
  done = False
  epsilon = 0.01
  cur_state = preprocess_state(env.reset())
  time_step = 0
  while not done:
    if np.random.random() > epsilon:
      action = agent.select_action(cur_state)
    else:
      action = env.action_space.sample()

    next_state, reward, done, info = env.step(action)
    next_state = preprocess_state(next_state)
    score += reward

    agent.add_memory(cur_state, action, reward, next_state, done)
    agent.optimize_model()

    cur_state = next_state
    time_step += 1
    if time_step % 100 == 0:
      print("Completed time step:", time_step)

  print("Episode {} score {}".format(i_episode, score))