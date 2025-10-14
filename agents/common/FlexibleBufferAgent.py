import numpy as np
from collections import deque
import torch
from agents.common.BaseAgent import BaseAgent


class FlexibleBufferAgent(BaseAgent):
    """
    Unified agent supporting both Monte Carlo and TD(n) learning.
    
    Modes:
    - 'episode': Accumulate full episodes (Monte Carlo)
    - 'n_step': Fixed n-step buffer (TD(n))
    - 'hybrid': Episode buffer but can learn before episode ends
    """
    
    def __init__(self, *, warmup_episodes=50, n_steps: int = 128, mode: str = 'episode', **kwargs):
        super().__init__(**kwargs)
        
        self.episode_count = 0
        self.n_steps = n_steps
        self.mode = mode
        self.warmup_episodes = warmup_episodes
        
        # Use unlimited deques for episode mode, limited for n_step mode
        maxlen = None if mode in ['episode', 'hybrid'] else n_steps
        
        self.state_memory = deque(maxlen=maxlen)
        self.next_state_memory = deque(maxlen=maxlen)
        self.action_memory = deque(maxlen=maxlen)
        self.reward_memory = deque(maxlen=maxlen)
        self.truncated_memory = deque(maxlen=maxlen)
        self.terminal_memory = deque(maxlen=maxlen)

    def store_transition(self, state, action, reward, next_state, truncated, terminal):
        self.state_memory.append(np.array(state, dtype=np.float32))
        self.next_state_memory.append(np.array(next_state, dtype=np.float32))
        self.action_memory.append(int(action))
        self.reward_memory.append(float(reward))
        self.truncated_memory.append(bool(truncated))
        self.terminal_memory.append(bool(terminal))

    def is_ready(self):
        """Override to handle warmup logic."""
        if self.mode == 'episode':
            return self.last_done_flag()
        
        elif self.mode == 'hybrid':
            # During warmup: only learn at episode end (MC behavior)
            if self.episode_count is not None and self.episode_count < self.warmup_episodes:
                return self.last_done_flag()
            # After warmup: learn every n_steps or at episode end
            else:
                return len(self.state_memory) >= self.n_steps or self.last_done_flag()
        
        elif self.mode == 'n_step':
            return len(self.state_memory) >= self.n_steps or self.last_done_flag()
        
        return False

    def sample_rollout(self) -> tuple[torch.Tensor, ...]:
        """Return all transitions currently in buffer as torch tensors."""
        device = self.networks[0].device

        states = torch.tensor(np.stack(self.state_memory), dtype=torch.float32).to(device)
        actions = torch.tensor(np.array(self.action_memory), dtype=torch.long).to(device)
        rewards = torch.tensor(np.array(self.reward_memory), dtype=torch.float32).to(device)
        next_states = torch.tensor(np.stack(self.next_state_memory), dtype=torch.float32).to(device)
        truncateds = torch.tensor(np.array(self.truncated_memory), dtype=torch.bool).to(device)
        terminals = torch.tensor(np.array(self.terminal_memory), dtype=torch.bool).to(device)

        return states, actions, rewards, next_states, truncateds, terminals

    def clear_rollout(self):
        """Empty the buffer."""
        self.state_memory.clear()
        self.next_state_memory.clear()
        self.action_memory.clear()
        self.reward_memory.clear()
        self.truncated_memory.clear()
        self.terminal_memory.clear()

    def pop_left(self):
        """Remove oldest transition (for sliding window)."""
        if len(self.state_memory) > 0:
            self.state_memory.popleft()
            self.next_state_memory.popleft()
            self.action_memory.popleft()
            self.reward_memory.popleft()
            self.truncated_memory.popleft()
            self.terminal_memory.popleft()

    def size(self):
        return len(self.state_memory)

    def last_done_flag(self):
        return bool(self.terminal_memory and self.terminal_memory[-1] or
                    self.truncated_memory and self.truncated_memory[-1])