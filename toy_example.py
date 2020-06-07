import numpy as np

class ToyExample():
    def __init__(self, num_b_actions):
        self.state = 0
        self.num_b_actions = num_b_actions

    def reset(self):
        self.state = 0
        return self.state

    def random_action(self):
        if self.state == 0:
            return np.random.randint(2)

        return np.random.randint(self.num_b_actions)

    def step(self, action):
        if self.state == 0:
            if action == 0:
                # Move left from A to B
                self.state = 1
                return self.state, 0, False

            # Move right from A to end
            self.state = -1
            return self.state, 0, True

        # Move from B to end
        self.state = -1
        return self.state, np.random.normal(-0.1, 1), True
