import numpy as np
import torch
from agents.common.BaseAgent import BaseAgent
from agents.features.PrioritizedReplayBuffer import PERBuffer

class PERBufferAgent(BaseAgent):
    def __init__(self, *, mem_size=100_000, alpha=0.5, beta0=0.4, beta_steps=1_000_000, eps=1e-6, **kwargs):
        super().__init__(**kwargs)
        self.memory = PERBuffer(
            mem_size=mem_size,
            input_dims=self.input_dims,
            alpha=alpha,
            beta0=beta0,
            beta_steps=beta_steps,
            eps=eps
        )

    def store_transition(self, state, action, reward, next_state, terminal):
        self.memory.store_transition(state, action, reward, next_state, terminal)

    
    def sample_from_buffer(self):
        device = self.networks[0].device 
        states, actions, rewards, next_states, terminals, idxs, w = self.memory.sample_from_buffer(self.batch_size, step=self.learn_step_cnt)
        states = torch.tensor(states).to(device=device)
        actions = torch.tensor(actions).to(device=device)
        rewards = torch.tensor(rewards).to(device=device)
        next_states = torch.tensor(next_states).to(device=device)
        terminals = torch.tensor(terminals).to(device=device)
        is_weights = torch.tensor(w).to(device=device)

        return states, actions, rewards, next_states, terminals, idxs, is_weights
    

    def update_priorities(self, idxs, td_errors):
        self.memory.update_priorities(idxs, td_errors)