import numpy as np
from datetime import datetime
from utils import sanitize_file_string
import torch

from networks.policyNetwork import PolicyNetwork
from agents.common.EpisodeBufferAgent import EpisodeBufferAgent



class REINFORCEAgent(EpisodeBufferAgent):
    """
        REINFORCE Policy Gradient Agent
        Based on Williams, R. J. (1992). Simple statistical gradient-following algorithms for connectionist reinforcement learning.
    """

    def __init__(self, baseline_alpha=0.01, model_name="DQN", *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.baseline_avg = 0.0
        self.baseline_alpha = baseline_alpha
        self.checkpoint_dir = f"models/{model_name}/{sanitize_file_string(self.env_name)}"
        self.filename_root = f"{model_name}_{sanitize_file_string(self.env_name)}_lr{self.learning_rate}_gamma{self.gamma}_eps{self.epsilon}_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"

        self.policy = PolicyNetwork(
            n_actions=self.n_actions,
            input_dims=self.input_dims,
            file_name=self.filename_root + "_eval"
        )

        self.networks = [self.policy] # for savings model dicts
        self.action_probs_store = []

    def choose_action(self, state):
        state = torch.tensor(state, dtype=torch.float).unsqueeze(0).to(device=self.policy.device)
        logits = self.policy(state)
        dist = torch.distributions.Categorical(logits=logits)
        action = dist.sample()
        log_prob = dist.log_prob(action)

        self.action_probs_store.append(log_prob) ## bad code, but hacky way of storing log prob without changing main training loop in train.py

        return action.item()
    
    def _compute_cumsum_returns(self, rewards: torch.Tensor, gamma=0.99):
        returns = torch.zeros_like(rewards)
        running = 0
        for t in reversed(range(len(rewards))):
            running = rewards[t] + gamma * running
            returns[t] = running
        return returns
    
    def learn(self):
        if self.episode_is_terminal():
            self.policy.optimizer.zero_grad()

            _, _, rewards, _, _ = self.sample_episode()
            action_probs = torch.stack(self.action_probs_store).to(self.policy.device)

            returns = self._compute_cumsum_returns(rewards, self.gamma)
            # Update baseline
            episode_return = returns[0].item()
            self.baseline_avg = (1 - self.baseline_alpha) * self.baseline_avg + self.baseline_alpha * episode_return

            # Compute advantages
            advantages = returns - self.baseline_avg

            # Normalize ADVANTAGES (not raw returns)
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

            loss = (-action_probs * advantages).mean()
            loss.backward()

             # Clip gradients to prevent large updates
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=0.5)
        
            self.policy.optimizer.step()

            self.learn_step_cnt += 1
            self.memory.clear_memory()