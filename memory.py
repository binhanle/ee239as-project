import numpy as np

class ReplayMemory:
    def __init__(self, capacity, state_shape):
        self.capacity = capacity
        self.position = 0
        self.buffer_size = 0
        self.state_mem = np.zeros((capacity, *state_shape), dtype=np.float32)
        self.next_state_mem = np.zeros((capacity, *state_shape), dtype=np.float32)
        self.action_mem = np.zeros(capacity, dtype=np.int32)
        self.reward_mem = np.zeros(capacity, dtype=np.float32)
        self.terminal_mem = np.zeros(capacity, dtype=np.bool)

    def push(self, state, action, reward, next_state, done):
        self.state_mem[self.position] = state
        self.next_state_mem[self.position] = next_state
        self.action_mem[self.position] = action
        self.reward_mem[self.position] = reward
        self.terminal_mem[self.position] = done

        self.position = (self.position + 1) % self.capacity
        self.buffer_size = min(self.buffer_size + 1, self.capacity)

    def sample(self, batch_size):
        samples = np.random.choice(self.buffer_size, batch_size, replace=False)

        states = self.state_mem[samples]
        actions = self.action_mem[samples]
        rewards = self.reward_mem[samples]
        next_states = self.next_state_mem[samples]
        terminal = self.terminal_mem[samples]

        return states, actions, rewards, next_states, terminal

    def __len__(self):
        return self.buffer_size