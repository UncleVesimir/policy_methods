import numpy as np
from datetime import datetime
from utils import sanitize_file_string
import torch

from networks.actorCriticNetwork import ActorCriticNetwork
from agents.common.EpisodeBufferAgent import EpisodeBufferAgent



class ActorCriticAgent(EpisodeBufferAgent):
    """
        Actor Critic Agent
        Based on Williams, R. J. (1992). Simple statistical gradient-following algorithms for connectionist reinforcement learning.
    """

    def __init__(self, baseline_alpha=0.01, model_name="DQN", *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.baseline_avg = 0.0
        self.baseline_alpha = baseline_alpha
        self.checkpoint_dir = f"models/{model_name}/{sanitize_file_string(self.env_name)}"
        self.filename_root = f"{model_name}_{sanitize_file_string(self.env_name)}_lr{self.learning_rate}_gamma{self.gamma}_eps{self.epsilon}_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"

        self.policy = ActorCriticNetwork(
            n_actions=self.n_actions,
            input_dims=self.input_dims,
            file_name=self.filename_root + "_eval"
        )
        self.value_store = []
        self.entropy_store = []
        self.action_prob_store = []

        self.networks = [self.policy] # for savings model dicts

    def choose_action(self, state):
        state = torch.tensor(state, dtype=torch.float).unsqueeze(0).to(device=self.policy.device)
        actor_logits, critic_value = self.policy(state)
        dist = torch.distributions.Categorical(logits=logits)

        action = dist.sample()
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()

        self.action_prob_store.append(log_prob)
        self.value_store.append(critic_value.squeeze(-1))
        self.entropy_store.append(entropy)

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

            states, actions, rewards, next_states, terminals = self.sample_episode()
            
            with torch.no_grad():
                _, next_values = self.policy(next_states)

            values = torch.stack(self.value_store).to(self.policy.device)          # [T]
            action_probs = torch.stack(self.action_prob_store).to(self.policy.device)  # [     
            
            targets = rewards + gamma * next_values * (~terminals)                # [T]
            advantages = (targets - values).detach()


            polcicy_loss = (-action_probs * advantages).mean()
            critic_loss = 0.5 * (targets - values).pow(2).mean()

            entropy_loss = -0.01 * torch.stack(self.entropy_store).mean()
            loss = polcicy_loss + critic_loss + entropy_loss

            loss.backward()

                # Clip gradients to prevent large updates
            # torch.nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=0.5)
        
            self.policy.optimizer.step()

            self.learn_step_cnt += 1