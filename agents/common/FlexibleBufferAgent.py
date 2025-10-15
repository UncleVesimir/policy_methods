import numpy as np
import torch
from collections import deque
from gymnasium import spaces

from agents.common.BaseAgent import BaseAgent


class FlexibleBufferAgent(BaseAgent):
    """
    Unified agent supporting both Monte Carlo and TD(n) learning.
    
    Modes:
    - 'episode': Accumulate full episodes (Monte Carlo)
    - 'n_step': Fixed n-step buffer (TD(n))
    - 'hybrid': Episode buffer but can learn before episode ends
    """
    
    def __init__(self, *, n_steps: int = 128, mode: str = 'episode',
                 action_space: spaces.Space, observation_space: spaces.Space,
**kwargs):
        super().__init__(**kwargs)
        
        self.episode_count = 0
        self.n_steps = n_steps
        self.mode = mode
        
        # Use unlimited deques for episode mode, limited for n_step mode
        maxlen = None if mode in ['episode'] else n_steps
        
        self.action_dtype = self.infer_action_dtype(action_space)
        self.obs_dtype = self.infer_obs_storage_dtype(observation_space)
        ##vectorized stores 
                                                        # vec                        | non-vec
        self.state_memory = deque(maxlen=maxlen)        # N_STEPS * [N_ENVS, 4, H, W] | [N_STEPS, 4, H, W]
        self.next_state_memory = deque(maxlen=maxlen)   # [N_STEPS, N_ENVS, 4, H, W] | [N_STEPS, 4, H, W]
        self.action_memory = deque(maxlen=maxlen)       # [N_STEPS, N_ENVS]          | [N_STEPS]
        self.reward_memory = deque(maxlen=maxlen)       # [N_STEPS, N_ENVS]          | [N_STEPS]
        self.truncated_memory = deque(maxlen=maxlen)    # [N_STEPS, N_ENVS]          | [N_STEPS]
        self.terminal_memory = deque(maxlen=maxlen)     # [N_STEPS, N_ENVS]          | [N_STEPS]

    def store_transition(self, state, action, reward, next_state, truncated, terminal):
        self.state_memory.append(np.array(state, dtype=self.obs_dtype))
        self.next_state_memory.append(np.array(next_state, dtype=self.obs_dtype))
        self.action_memory.append(np.array(action, dtype=self.action_dtype))
        self.reward_memory.append(np.array(reward, dtype=np.float32))
        self.truncated_memory.append(np.array(truncated, dtype=np.bool_))
        self.terminal_memory.append(np.array(terminal, dtype=np.bool_))


        

    def is_ready(self):
        """Override to handle warmup logic."""
        if self.mode == 'episode':
            return self.last_done_flag()
        
        elif self.mode == 'n_step':
            print("mem length: ", len(self.state_memory))
            return len(self.state_memory) >= self.n_steps or self.last_done_flag()
        
        return False

    def sample_rollout(self) -> tuple[torch.Tensor, ...]:
        """Return all transitions currently in buffer as torch tensors."""
        device = self.networks[0].device

        states = torch.from_numpy(np.stack(self.state_memory)).to(device)
        actions = torch.from_numpy(np.stack(self.action_memory)).to(device)
        rewards = torch.from_numpy(np.stack(self.reward_memory)).to(device)
        next_states = torch.from_numpy(np.stack(self.next_state_memory)).to(device)
        truncateds = torch.from_numpy(np.stack(self.truncated_memory)).to(device)
        terminals = torch.from_numpy(np.stack(self.terminal_memory)).to(device)

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

        return (self.terminal_memory and np.any(self.terminal_memory[-1])) or \
               (self.truncated_memory and np.any(self.truncated_memory[-1]))

    def infer_action_dtype(self, space: spaces.Space) -> np.dtype:
        if isinstance(space, (spaces.Discrete, spaces.MultiDiscrete, spaces.MultiBinary)):
            return np.int64  #                  # torch long for discrete actions
        if isinstance(space, spaces.Box):
            return np.float32                       # torch float for continuous actions
        raise NotImplementedError(f"Unsupported action space: {type(space)}")

    def infer_obs_storage_dtype(self, space: spaces.Space) -> np.dtype:
        # store as uint8 if observations are images; otherwise float32
        if isinstance(space, spaces.Box) and space.dtype == np.uint8:
            return np.uint8
        return np.float32
