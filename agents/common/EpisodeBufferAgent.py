
import numpy as np
import torch
from agents.common.BaseAgent import BaseAgent
from agents.features.EpisodeBuffer import EpisodeBuffer

class EpisodeBufferAgent(BaseAgent):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.memory = EpisodeBuffer()

    def store_transition(self, state, action, reward, next_state, truncated, terminal):
        self.memory.store_transition(state, action, reward, next_state, truncated, terminal)

    
    def sample_episode(self) -> tuple[torch.Tensor, ...]:
        device = self.networks[0].device
        states, actions, rewards, next_states, truncateds, terminals = self.memory.sample_episode()

        states = torch.tensor(states).to(device=device)
        actions = torch.tensor(actions).to(device=device)
        # action_probs = torch.stack(action_probs).to(device=device)  # Stack list of tensors, preserving gradients
        rewards = torch.tensor(rewards).to(device=device)
        next_states = torch.tensor(next_states).to(device=device)
        truncateds = torch.tensor(truncateds).to(device=device)
        terminals = torch.tensor(terminals).to(device=device)


        return states, actions, rewards, next_states, truncateds, terminals
    
    def episode_is_terminal(self) -> bool:
        return self.memory.terminal_memory[-1]    
    
    def clear_memory(self):
        self.memory.clear_memory()