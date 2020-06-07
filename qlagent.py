import numpy as np

class QLAgent():
    def __init__(self, q_init, lr=0.1, gamma=1):
        self.q_init = q_init
        self.q_table = [q_init[0][:], q_init[1][:]]
        self.lr = lr
        self.gamma = gamma
        self.step_counter = 0

    def reset(self):
        self.q_table = [self.q_init[0][:], self.q_init[1][:]]

    def select_action(self, state):
        max_q = max(self.q_table[state])
        is_max = [q == max_q for q in self.q_table[state]]
        max_actions = np.flatnonzero(is_max)
        return np.random.choice(max_actions)

    def get_value(self, state):
        return np.max(self.q_table[state])

    def update(self, cur_state, action, next_state, reward):
        cur_q = self.q_table[cur_state][action]
        if next_state != -1:
            next_q = self.get_value(next_state)
        else:
            next_q = 0

        q_target = reward + self.gamma*next_q

        error = q_target - cur_q
        self.q_table[cur_state][action] += self.lr*error

        self.step_counter += 1
