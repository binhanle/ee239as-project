import numpy as np

class ReplayMemory:
    def __init__(self, capacity, state_shape):
        self.capacity = capacity
        self.state_shape = state_shape
        self.position = 0
        self.buffer = []

    def push(self, state, action, reward, next_state, done):
        mem = {"state":state,
                "action":action,
                "reward":reward,
                "next_state":next_state,
                "done":done}

        if len(self.buffer) < self.capacity:
            self.buffer.append(mem)
        else:
            self.buffer[self.position] = mem

        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        states = np.zeros((batch_size,) + self.state_shape, dtype=np.float32)
        next_states = np.zeros((batch_size,) + self.state_shape, dtype=np.float32)
        actions = np.zeros(batch_size, dtype=np.float32)
        rewards = np.zeros(batch_size, dtype=np.float32)
        terminal = np.zeros(batch_size, dtype=np.bool)

        samples = np.random.choice(len(self.buffer), batch_size, replace=False)
        for index, sample in enumerate(samples):
            states[index] = np.asarray(self.buffer[sample]["state"])
            next_states[index] = np.asarray(self.buffer[sample]["next_state"])
            actions[index] = self.buffer[sample]["action"]
            rewards[index] = self.buffer[sample]["reward"]
            terminal[index] = self.buffer[sample]["done"]

        return states, actions, rewards, next_states, terminal

    def __len__(self):
        return len(self.buffer)