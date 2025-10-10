import numpy as np
from collections import deque
import torch
from agents.common.BaseAgent import BaseAgent


class SlidingWindowBufferAgent(BaseAgent):
    def __init__(self, *,  n_steps: int = 5, **kwargs):
        super().__init__(**kwargs)

        self.n_steps = n_steps
        self.state_memory      = deque(maxlen=n_steps)
        self.next_state_memory = deque(maxlen=n_steps)
        self.action_memory     = deque(maxlen=n_steps)
        self.reward_memory     = deque(maxlen=n_steps)
        self.truncated_memory  = deque(maxlen=n_steps)
        self.terminal_memory   = deque(maxlen=n_steps)

    def store_transition(self, state, action, reward, next_state, truncated, terminal):
        self.state_memory.append(np.array(state, dtype=np.float32))
        self.next_state_memory.append(np.array(next_state, dtype=np.float32))
        self.action_memory.append(int(action))
        self.reward_memory.append(float(reward))
        self.truncated_memory.append(bool(truncated))
        self.terminal_memory.append(bool(terminal))

    def is_ready(self):
        """Return True when at least n_steps transitions are stored or a boundary is reached."""
        return len(self.state_memory) >= self.n_steps or (self.terminal_memory and self.terminal_memory[-1])

    def sample_rollout(self) -> tuple[torch.Tensor, ...]:
        """Return all transitions currently in buffer as torch tensors (in insertion order)."""
        device = self.networks[0].device

        states = torch.tensor(np.stack(self.state_memory), dtype=torch.float32).to(device)
        actions = torch.tensor(np.array(self.action_memory), dtype=torch.long).to(device)
        rewards = torch.tensor(np.array(self.reward_memory), dtype=torch.float32).to(device)
        next_states = torch.tensor(np.stack(self.next_state_memory), dtype=torch.float32).to(device)
        truncateds = torch.tensor(np.array(self.truncated_memory), dtype=torch.bool).to(device)
        terminals = torch.tensor(np.array(self.terminal_memory), dtype=torch.bool).to(device)

        return states, actions, rewards, next_states, truncateds, terminals


    def clear_rollout(self):
        """Empty the buffer after an update (A2C-style)."""
        self.state_memory.clear()
        self.next_state_memory.clear()
        self.action_memory.clear()
        self.reward_memory.clear()
        self.truncated_memory.clear()
        self.terminal_memory.clear()

    def pop_left(self):
        """Optional: pop one transition from the left (sliding-window TD(n) style)."""
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
