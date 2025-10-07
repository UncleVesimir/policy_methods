import numpy as np
from agents.features.ReplayBuffer import ReplayBuffer


class PERBuffer(ReplayBuffer):
    """
    Prioritized Experience Replay (proportional variant): https://arxiv.org/pdf/1511.05952
    Equations follow Schaul et al. 2016:
       p_i = (|delta_i| + eps)^alpha
       P(i) = p_i / sum_k p_k
       w_i = (1 / (N * P(i)))^beta  / max_j w_j
    """
      
    def __init__(self, *, mem_size=None, input_dims=None, alpha=0.5, beta0=0.4, beta_steps=1_000_000, eps=1e-6, **kwargs):
        if not mem_size or not input_dims:
            raise ValueError("PER Buffer requires mem_size and input_dim to be set, got mem_size:", mem_size, "input_dims:", input_dims)
        
        super().__init__(mem_size=mem_size, input_dims=input_dims)

        self.alpha = alpha
        self.beta0 = beta0
        self.beta_steps = beta_steps
        self.eps = eps

        # priorities (initialize to 1.0 so new items are sampled)
        self.priorities = np.ones(self.mem_size, dtype=np.float32)

    def __len__(self):
        return min(self.mem_cntr, self.mem_size)

    def store_transition(self, state, action, reward, next_state, done, init_priority=None):
        idx = self.mem_cntr % self.mem_size # idx ensures new entries are pushed to "front" of buffer when it is full

        self.state_memory[idx]      = state
        self.next_state_memory[idx] = next_state
        self.action_memory[idx]     = int(action)
        self.reward_memory[idx]     = reward
        self.terminal_memory[idx]   = bool(done)

        N = len(self)
        # if no priority provided, use current max to ensure the new sample is seen at
        # least once
        if init_priority is None:
            max_p = self.priorities[:N].max() if N > 0 else 1.0
            self.priorities[idx] = max_p
        else:
            # p_i = (|delta| + eps), but we store p_i = (|delta| + eps)**alpha, given P(i) = p_i**alpha / sum_k(p_k**alpha)
            # convert to p_i = (|delta| + eps)^alpha if caller passes TD error
            self.priorities[idx] = (abs(float(init_priority)) + self.eps) ** self.alpha

        self.mem_cntr += 1

    def _beta_at(self, step):
        """Linear anneal beta from beta0 → 1.0 over beta_steps."""
        if self.beta_steps <= 0:
            return 1.0
        frac = min(1.0, max(0.0, step / float(self.beta_steps)))
        #gradually increase proportional to # learning steps, reducing bias of prioritizing high error experiences
        return self.beta0 + (1.0 - self.beta0) * frac 

    def sample_from_buffer(self, batch_size, step=0):
        """
        Returns:
           states, actions, rewards, next_states, terminals, indices, is_weights
        """
        N = len(self) # current size of buffer, N
        if N == 0:
            raise ValueError("PER buffer is empty.")
        
        batch_size = min(batch_size, N) ## avoid index-out-of-range

        # probabilities P(i) ∝ p_i
        p = self.priorities[:N]
        if self.alpha == 0.0:
            P = np.full_like(p, 1.0 / N)  # uniform sampling, returns np.array of size p, filled with probabilities 1/N
        else:
            P = p / p.sum()

        idxs = np.random.choice(N, batch_size, replace=False, p=P)

        # importance-sampling weights
        beta = self._beta_at(step)
        is_weights = (N * P[idxs]) ** (-beta)
        is_weights = is_weights / is_weights.max()  # normalize for stability
        is_weights = is_weights.astype(np.float32)

        states      = self.state_memory[idxs]
        actions     = self.action_memory[idxs]
        rewards     = self.reward_memory[idxs]
        next_states = self.next_state_memory[idxs]
        terminals   = self.terminal_memory[idxs]

        return states, actions, rewards, next_states, terminals, idxs, is_weights

    def update_priorities(self, idxs, td_errors):
        """
        Update priorities after a learning step.
        td_errors: numpy array or list, shape [batch]
        """
        td_errors = np.asarray(td_errors, dtype=np.float32)
        # p_i = (|delta| + eps)^alpha
        new_p = (np.abs(td_errors) + self.eps) ** self.alpha
        self.priorities[idxs] = new_p