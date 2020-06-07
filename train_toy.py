from toy_example import ToyExample
from qlagent import QLAgent
from dqlagent import DQLAgent
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

# set environment here
print("Using toy example from Sutton and Barto, p. 135")

NUM_B_ACTIONS = 8 # number of actions from state B
env = ToyExample(NUM_B_ACTIONS)

# set training parameters here
LR = 0.1 # learning rate (textbook used 0.1)
GAMMA = 1 # textbook used 1

q_init = [[0, 0], [0] * NUM_B_ACTIONS]

agent = QLAgent(q_init, LR, GAMMA)

# training phase

# adjust these hyperparameters as necessary
num_episodes = 300 # number of episodes to train for
num_trials = 10000 # number of runs (textbook used 10k)
epsilon = 0.1 # initial epsilon value (textbook used 0.1)

def do_experiment(agent):
  score_totals = np.zeros(num_episodes)
  value_totals = np.zeros(num_episodes)
  left_totals = np.zeros(num_episodes)
  
  for i_trial in range(num_trials):
    agent.reset()
  
    for i_episode in range(num_episodes):
      score = 0.0
      done = False
    
      cur_state = env.reset()
      value = agent.get_value(cur_state)
    
      while not done:
        if np.random.random() > epsilon:
          action = agent.select_action(cur_state) # exploit
        else:
          action = env.random_action() # explore
  
        if cur_state == 0 and action == 0:
          left_totals[i_episode] += 1
    
        next_state, reward, done = env.step(action)
        score += reward
        
        agent.update(cur_state, action, next_state, reward)
        cur_state = next_state
    
      # print("Episode {} score: {}, value: {}".format(i_episode, score, value))
      score_totals[i_episode] += score
      value_totals[i_episode] += value

    if (i_trial + 1) % 1000 == 0:
      print("Completed trial", i_trial + 1, "of", num_trials)
  
  return score_totals/num_trials, value_totals/num_trials, left_totals/num_trials

# Perform experiment with Q learning
print("Performing Q-learning experiment...")
ql_scores, ql_values, ql_left = do_experiment(agent)

# Repeat experiment with double Q learning
print("Performing double Q-learning experiment...")
agent = DQLAgent(q_init, LR, GAMMA)
dql_scores, dql_values, dql_left = do_experiment(agent)

# Plot results
plt.figure()
plt.plot(ql_scores)
plt.plot(dql_scores)
plt.title("Q-learning vs. Double Q-learning, average score by episode")
plt.xlabel("Episode")
plt.ylabel("Score")

plt.figure()
plt.plot(ql_values)
plt.plot(dql_values)
plt.title("Q-learning vs. Double Q-learning, average value by episode")
plt.xlabel("Episode")
plt.ylabel("Value")

plt.figure()
plt.plot(ql_left)
plt.plot(dql_left)
ax = plt.gca()
ax.yaxis.set_major_formatter(mtick.PercentFormatter(1))
plt.title("Q-learning vs. Double Q-learning, average % left action from A")
plt.xlabel("Episode")
plt.ylabel("% left action from A")

plt.show()
