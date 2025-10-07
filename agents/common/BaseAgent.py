
import numpy as np
import torch
from datetime import datetime

class BaseAgent():
    def __init__(self, *, input_dims=None, n_actions=None, gamma=0.99, epsilon=1.0, epsilon_dec=5e-7, min_epsilon=0.01,
                  batch_size=32, learning_rate=0.001, replace_limit=1000, env_name=None, **kwargs):
        if not input_dims or not n_actions:
            raise ValueError("Agent requires inputs_dims and n_actions to be set")

        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_dec = epsilon_dec
        self.min_epsilon = min_epsilon
        self.learning_rate = learning_rate
        self.n_actions = n_actions
        self.input_dims = input_dims
        self.learn_step_cnt = 0
        self.replace_limit = replace_limit
        self.batch_size = batch_size
        self.env_name=env_name

        self.networks = []

    def choose_action(self, state):
        device = self.networks[0].device
        if np.random.random() > self.epsilon:
            state = torch.tensor(state, dtype=torch.float).unsqueeze(0).to(device=device) # unsqueeze to add required batch dimension
            action = self.networks[0](state).argmax().item()
        else:
            action = np.random.choice(self.n_actions)
        print(f"Agent chose action: {action}, number of actions available: {self.n_actions}")
        return action

    def save_models(self):
        for network in self.networks:
            network.save_checkpoint()

    def load_models(self):
        for network in self.networks:
            network.load_checkpoint()