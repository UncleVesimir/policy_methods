import numpy as np

class ReplayBuffer():
    def __init__(self, *, mem_size=None, input_dims=None):
        if not mem_size or not input_dims:
            raise ValueError("Replay Buffer requires mem_size and input_dim to be set")
        
        self.mem_size = mem_size
        self.mem_cntr = 0
        self.state_memory = np.zeros((self.mem_size, *input_dims), dtype=np.float32)
        self.next_state_memory = np.zeros((self.mem_size, *input_dims), dtype=np.float32)
        self.action_memory = np.zeros(self.mem_size, dtype=np.int32)
        self.reward_memory = np.zeros(self.mem_size, dtype=np.float32)
        self.truncated_memory = np.zeros(self.mem_size, dtype=np.float32)
        self.terminal_memory = np.zeros(self.mem_size, dtype=np.bool)

    def store_transition(self, state, action, reward, next_state, truncated, terminal):
        # print("[RB/WRITE] in.state dtype/shape:", np.asarray(state).dtype, np.asarray(state).shape)
        # print("[RB/WRITE] in.next_state dtype/shape:", np.asarray(next_state).dtype, np.asarray(next_state).shape)
        # print("[RB/WRITE] in.action dtype:", np.asarray(action).dtype, "reward dtype:", np.asarray(reward).dtype, "done dtype:", np.asarray(done).dtype)
        index = self.mem_cntr % self.mem_size
        self.state_memory[index] = state
        self.next_state_memory[index] = next_state
        self.action_memory[index] = action
        self.reward_memory[index] = reward
        self.truncated_memory[index] = truncated
        self.terminal_memory[index] = terminal

        self.mem_cntr += 1

    def sample_from_buffer(self, batch_size):
        size = min(self.mem_cntr, self.mem_size) ## avoid sampling from uninitialized memory
        batch_size = min(batch_size, size) ## ensure we don't sample more than we have

        batch = np.random.choice(size, batch_size, replace=False)

        states = self.state_memory[batch]
        actions = self.action_memory[batch]
        rewards = self.reward_memory[batch]
        next_states = self.next_state_memory[batch]
        truncateds = self.truncated_memory[batch]
        terminals = self.terminal_memory[batch]
        # print("[RB/SAMPLE] states np dtype:", states.dtype, "next_states np dtype:", next_states.dtype)
        # print("[RB/SAMPLE]actions dtype:", actions.dtype, "rewards dtype:", rewards.dtype, "terminals dtype:", terminals.dtype)

        return states, actions, rewards, next_states, truncateds, terminals