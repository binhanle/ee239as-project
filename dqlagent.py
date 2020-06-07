import numpy as np

class DQLAgent():
    def __init__(self, q_init, lr=0.1, gamma=1):
        self.q_init = q_init
        self.q1_table = [q_init[0][:], q_init[1][:]]
        self.q2_table = [q_init[0][:], q_init[1][:]]
        self.lr = lr
        self.gamma = gamma
        self.step_counter = 0

    def reset(self):
        self.q1_table = [self.q_init[0][:], self.q_init[1][:]]
        self.q2_table = [self.q_init[0][:], self.q_init[1][:]]

    def select_action(self, state):
        max_q = max(self.q1_table[state])
        is_max = [q == max_q for q in self.q1_table[state]]
        max_actions = np.flatnonzero(is_max)
        return np.random.choice(max_actions)

    def get_value(self, state):
        return np.max(self.q1_table[state])

    def update(self, cur_state, action, next_state, reward):
        cur_q = self.q1_table[cur_state][action]
        if next_state != -1:
            max_actions = self.select_action(next_state)
            next_q = self.q2_table[next_state][max_actions]
        else:
            next_q = 0

        q_target = reward + self.gamma*next_q

        error = q_target - cur_q
        self.q1_table[cur_state][action] += self.lr*error
        self.q1_table, self.q2_table = self.q2_table, self.q1_table

        self.step_counter += 1
