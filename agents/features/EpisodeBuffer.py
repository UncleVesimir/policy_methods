import numpy as np

class EpisodeBuffer():
    def __init__(self, *, input_dims=None):
        self.state_memory = []
        self.next_state_memory = []
        self.action_memory = []
        self.reward_memory = []
        self.truncated_memory = []
        self.terminal_memory = []

    def store_transition(self, state, action, reward, next_state, truncated, terminal):
        self.state_memory.append(state)
        self.next_state_memory.append(next_state)
        self.action_memory.append(action)
        self.reward_memory.append(reward)
        self.truncated_memory.append(truncated)
        self.terminal_memory.append(terminal)
    
    # def store_action_prob(self, action_prob):
    #     self.action_probs.append(action_prob)

    def sample_episode(self) -> tuple[np.ndarray,...]:

        states = np.array(self.state_memory, dtype=np.float32)
        actions = np.array(self.action_memory, dtype=np.int64)
        # action_probs = self.action_probs  # Keep as list of tensors to preserve gradients
        rewards = np.array(self.reward_memory, dtype=np.float32)
        next_states = np.array(self.next_state_memory, dtype=np.float32)
        truncateds = np.array(self.truncated_memory, dtype=np.bool_)
        terminals = np.array(self.terminal_memory, dtype=np.bool_)

        return states, actions, rewards, next_states, truncateds, terminals
    

    def clear_memory(self):
        self.state_memory = []
        self.next_state_memory = []
        self.action_memory = []
        # self.action_probs = []
        self.reward_memory = []
        self.terminal_memory = []
        self.truncated_memory = []