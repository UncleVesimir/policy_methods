
import numpy as np
import torch
from agents.common.BaseAgent import BaseAgent
from agents.features.ReplayBuffer import ReplayBuffer

class ReplayBufferAgent(BaseAgent):
    def __init__(self, mem_size=100_000,*args, **kwargs):
        super().__init__(*args, **kwargs)
        self.mem_size = mem_size
        self.memory = ReplayBuffer(mem_size=self.mem_size, input_dims=self.input_dims)

    def store_transition(self, state, action, reward, next_state, terminal):
        self.memory.store_transition(state, action, reward, next_state, terminal)

    
    def sample_from_buffer(self):
        states, actions, rewards, next_states, terminals = self.memory.sample_from_buffer(self.batch_size)
        states = torch.tensor(states).to(self.q_eval.device)
        actions = torch.tensor(actions).to(self.q_eval.device)
        rewards = torch.tensor(rewards).to(self.q_eval.device)
        next_states = torch.tensor(next_states).to(self.q_eval.device)
        terminals = torch.tensor(terminals).to(self.q_eval.device)

        return states, actions, rewards, next_states, terminals