from nn_models import DQN
from memory import ReplayMemory
from dqnagent import DQNAgent
from ddqnagent import DDQNAgent
from checkpoint import save_checkpoint, load_checkpoint
from atari_wrappers import make_atari, wrap_deepmind, clip_reward
import torch
import numpy as np
import gym
import os

# set environment here
ATARI_GAME = "BreakoutNoFrameskip-v4"
print("Using atari game:", ATARI_GAME)

env = make_atari("BreakoutNoFrameskip-v4")
env = wrap_deepmind(env, clip_rewards=False)

N_ACTIONS = env.action_space.n
print("Action space is:", env.action_space)

STATE_SHAPE = env.observation_space.shape
print("Observation space is:", STATE_SHAPE)

# set training parameters here
MEMORY_SIZE = 10000 # maximum size of memory buffer, increase to as large as possible
LR = 0.00025 # learning rate
GAMMA = 0.99
BATCH_SIZE = 32
UPDATE_ONLINE_INTERVAL = 4 # number of steps in bewteen paramter updates to online net
UPDATE_TARGET_INTERVAL = 10000 # how frequently parameters are copied from online net to target net

CKPT_FILENAME = "breakout.ckpt"
CKPT_ENABLED = False
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dqn_online = DQN(N_ACTIONS, STATE_SHAPE)
dqn_target = DQN(N_ACTIONS, STATE_SHAPE)
dqn_online.to(device)
dqn_target.to(device)
# optimizer = torch.optim.RMSprop(dqn_online.parameters(), lr=LR, momentum=0.95, eps=0.01)
optimizer = torch.optim.Adam(dqn_online.parameters(), lr=LR)
if CKPT_ENABLED and os.path.exists(CKPT_FILENAME):
    progress = load_checkpoint(dqn_online, dqn_target, optimizer, CKPT_FILENAME)
else:
    progress = []

mem_buffer = ReplayMemory(MEMORY_SIZE, STATE_SHAPE)

loss_fn = torch.nn.SmoothL1Loss() # huber loss function
agent = DQNAgent(device, mem_buffer, dqn_online, dqn_target, optimizer, loss_fn, GAMMA, BATCH_SIZE, UPDATE_ONLINE_INTERVAL, UPDATE_TARGET_INTERVAL)

# training phase

# adjust these hyperparameters as necessary
num_episodes = 10 # number of episodes to train for
explore_phase_length = 50000 # number of steps without any exploitation
epsilon = 1.0 # initial epsilon value
epsilon_decrement_steps = 200000 # how many steps to decrement epsilon to min value (paper used 1 million)
min_epsilon = 0.01 # smallest possible value of epsilon
epsilon_dec = (epsilon - min_epsilon) / epsilon_decrement_steps

total_steps = 0
max_score = 0.0
for i_episode in range(num_episodes):
  # print("Running episode:", i_episode)
  score = 0.0
  agent_score = 0.0
  done = False
  time_step = 0

  cur_state = env.reset()

  while not done:
    
    # linearly anneal epsilon
    if total_steps > explore_phase_length:
      epsilon = max(epsilon - epsilon_dec, min_epsilon)
    
    if total_steps > explore_phase_length and np.random.random() > epsilon:
        action = agent.select_action(cur_state) # exploit
    else:
        action = env.action_space.sample() # explore

    next_state, reward, done, info = env.step(action)
    agent_reward = clip_reward(reward)
    agent.add_memory(cur_state, action, agent_reward, next_state, done)

    score += reward
    agent_score += agent_reward
    
    agent.optimize_model()

    cur_state = next_state
    
    time_step += 1
    total_steps += 1
    if time_step % 100 == 0:
      print("Completed iteration", time_step)

  print("Episode {} score: {}, agent score: {}, total steps taken: {}, epsilon: {}".format(i_episode, score, agent_score, total_steps, epsilon))
  progress.append((time_step, total_steps, score, agent_score))
  # print("Progress is", progress)
  if CKPT_ENABLED and score > max_score:
    max_score = score
    save_checkpoint(progress, dqn_online, dqn_target, optimizer, CKPT_FILENAME)