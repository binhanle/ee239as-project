import torch

class Agent():
    def __init__(self, device, mem_buffer, q_online, q_target, optimizer, loss_fn, gamma=0.99, batch_size=200, update_target_interval=1000):
        self.device = device
        self.mem_buffer = mem_buffer
        self.q_online = q_online
        self.q_target = q_target
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.gamma = gamma
        self.batch_size = batch_size
        self.update_target_interval = update_target_interval
        self.step_counter = 0

    def select_action(self, state):
        state_tensor = torch.tensor([state], dtype=torch.float).to(self.device)
        qvalues = self.q_online(state_tensor)
        return torch.argmax(qvalues).item()

    def add_memory(self, state, action, reward, next_state, done):
        self.mem_buffer.push(state, action, reward, next_state, done)

    def sample_memory(self):
        state, action, reward, next_state, done = self.mem_buffer.sample(self.batch_size)

        states = torch.tensor(state).to(self.device)
        rewards = torch.tensor(reward).to(self.device)
        dones = torch.tensor(done).to(self.device)
        actions = torch.tensor(action, dtype=torch.long).to(self.device)
        next_states = torch.tensor(next_state).to(self.device)

        return states, actions, rewards, next_states, dones

    def update_target_network(self):
        if self.step_counter % self.update_target_interval == 0:
            self.q_target.load_state_dict(self.q_online.state_dict())

    def optimize_model(self):
        if len(self.mem_buffer) < self.batch_size:
            return

        states, actions, rewards, next_states, dones = self.sample_memory()

        self.optimizer.zero_grad()

        indices = list(range(self.batch_size))
        cur_Q = self.q_online(states)[indices, actions]
        next_Q = self.q_target(next_states).max(dim=1).values

        cur_Q[dones] = 0.0
        q_target = rewards + self.gamma*next_Q

        loss = self.loss_fn(q_target, cur_Q).to(self.device)
        loss.backward()
        self.optimizer.step()

        self.update_target_network()

        self.step_counter += 1