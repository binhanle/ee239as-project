from nn_models import DQN
from memory import ReplayMemory
from agents import Agent
from checkpoint import save_checkpoint, load_checkpoint
from atari_wrappers import make_atari, wrap_deepmind
import torch
import numpy as np
import gym
import os

# set environment here
ATARI_GAME = "BreakoutNoFrameskip-v4"
print("Using atari game:", ATARI_GAME)

env = make_atari("BreakoutNoFrameskip-v4")
env = wrap_deepmind(env)

n_actions = env.action_space.n
print("Action space is:", env.action_space)

STATE_SHAPE = env.observation_space.shape
print("Observation space is:", STATE_SHAPE)

# set training parameters here
MEMORY_SIZE = 1000
LR = 0.001
CKPT_FILENAME = "breakout.ckpt"
CKPT_ENABLED = False
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dqn_online = DQN(n_actions, STATE_SHAPE)
dqn_target = DQN(n_actions, STATE_SHAPE)
dqn_online.to(device)
dqn_target.to(device)
optimizer = torch.optim.RMSprop(dqn_online.parameters(), lr=LR)
if CKPT_ENABLED and os.path.exists(CKPT_FILENAME):
    mem_buffer, progress = load_checkpoint(dqn_online, dqn_target, optimizer, CKPT_FILENAME)
else:
    mem_buffer = ReplayMemory(MEMORY_SIZE, STATE_SHAPE)
    progress = []

loss_fn = torch.nn.SmoothL1Loss()
agent = Agent(device, mem_buffer, dqn_online, dqn_target, optimizer, loss_fn)

num_episodes = 10
for i_episode in range(num_episodes):
  print("Running episode:", i_episode)
  score = 0.0
  done = False
  epsilon = 0.01
  cur_state = env.reset()
  time_step = 0
  while not done:
    if np.random.random() > epsilon:
      action = agent.select_action(cur_state)
    else:
      action = env.action_space.sample()

    next_state, reward, done, info = env.step(action)
    score += reward

    agent.add_memory(cur_state, action, reward, next_state, done)
    agent.optimize_model()

    cur_state = next_state
    time_step += 1
    if time_step % 100 == 0:
      print("Completed time step:", time_step)

  print("Episode {} score {}".format(i_episode, score))
  progress.append((time_step, score))
  print("Progress is", progress)
  if CKPT_ENABLED:
    save_checkpoint(mem_buffer, progress, dqn_online, dqn_target, optimizer, CKPT_FILENAME)
