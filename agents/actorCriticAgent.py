import numpy as np
from datetime import datetime
from utils import sanitize_file_string
import torch

from networks.actorCriticNetwork import ActorCriticNetwork
from agents.common.FlexibleBufferAgent import FlexibleBufferAgent



class ActorCriticAgent(FlexibleBufferAgent):
    """
    Actor Critic Agent with flexible learning modes.
    """

    def __init__(self, model_name="A2C", n_steps=5, value_coef=0.5,
                 max_grad_norm=0.5, entropy_coef=0.0, lr=1e-3, mode='episode', 
                 warmup_episodes=50, *args, **kwargs):
        super().__init__(n_steps=n_steps, mode=mode, warmup_episodes=warmup_episodes, *args, **kwargs)
        
        self.n_steps = n_steps
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.mode = mode
        self.warmup_episodes = warmup_episodes
        self.max_grad_norm = max_grad_norm

        self.checkpoint_dir = f"models/{model_name}/{sanitize_file_string(self.env_name)}"
        self.filename_root = f"{model_name}_{sanitize_file_string(self.env_name)}_lr{lr}_gamma{self.gamma}_eps{self.epsilon}_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"

        self.policy = ActorCriticNetwork(
            n_actions=self.n_actions,
            input_dims=self.input_dims,
            file_name=self.filename_root + "_eval",
            lr=lr
        )
        
        # # Initialize critic bias to expected return for Pong
        # nn.init.constant_(self.policy.critic.bias, -20.0)

        self.networks = [self.policy]

    def choose_action(self, state):
        state = torch.tensor(state, dtype=torch.float).unsqueeze(0).to(device=self.policy.device)
        with torch.no_grad():
            actor_logits, _ = self.policy(state)

        dist = torch.distributions.Categorical(logits=actor_logits)
        action = dist.sample()

        return action.item()
    
    
    def _compute_gae(
        self,
        rewards,        # [T] float
        truncateds,     # [T] bool
        terminals,      # [T] bool
        values,         # [T] float (from V(s_t))
        next_values,    # [T] float (from V(s_{t+1}))
        gamma=0.99,
        gae_lambda=0.95,
        treat_truncation_as_terminal: bool = False,
    ):
        """
        Generalized Advantage Estimation (recursive, backward pass).

        - By default, truncations are *not* treated as terminal (common on Atari time-limit truncation).
        Set treat_truncation_as_terminal=True to zero-out at truncations as well.
        """

        # Ensure float tensors on the model device
        device = values.device

        # Build the "not-done" mask:
        #   - if treat_truncation_as_terminal=False (recommended for Atari time limits),
        #     only true environment terminations zero the bootstrap.
        if treat_truncation_as_terminal:
            done = terminals | truncateds
        else:
            done = terminals

        not_done = (~done).float()

        T = rewards.shape[0]
        advantages = torch.zeros(T, dtype=torch.float32, device=device)

        gae = torch.zeros((), dtype=torch.float32, device=device)
        for t in reversed(range(T)):
            # δ_t = r_t + γ * V(s_{t+1}) * (1 - done_t) - V(s_t)
            delta = rewards[t] + gamma * next_values[t] * not_done[t] - values[t]
            # A_t = δ_t + γλ * (1 - done_t) * A_{t+1}
            gae = delta + gamma * gae_lambda * not_done[t] * gae
            advantages[t] = gae

        # Critic target: returns = advantages + V(s_t)
        returns = (advantages + values)

        # Detach advantages for the policy loss (standard)
        return returns, advantages

    def learn(self):
        if not self.is_ready():
            return

        self.policy.optimizer.zero_grad()

        states, actions, rewards, next_states, truncateds, terminals = self.sample_rollout()

        logits, values = self.policy(states)
        with torch.no_grad():
            _, next_values = self.policy(next_states)

        values = values.squeeze(-1)
        next_values = next_values.squeeze(-1)

        dist = torch.distributions.Categorical(logits=logits)
        log_probs = dist.log_prob(actions)
        entropies = dist.entropy()

        print(f"DEBUG:")
        print(f"  rewards: sum={rewards.sum():.1f}, mean={rewards.mean():.3f}")
        print(f"  values: sum={values.sum():.1f}, mean={values.mean():.1f}")
        print(f"  next_values: mean={next_values.mean():.1f}")

        # Use GAE for stable advantage estimation
        returns, advantages = self._compute_gae(
            rewards, truncateds, terminals,
            values.detach(), next_values,
            self.gamma, gae_lambda=1.0
        )

        print(f"  advantages: sum={advantages.sum():.1f}, mean={advantages.mean():.3f}")
        print(f"  returns: sum={returns.sum():.1f}, mean={returns.mean():.1f}")

        # Normalize advantages
        if len(advantages) > 1:
        # Check for pathological cases
            if advantages.std() > 1e-6:
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
            else:
                # Rare: all advantages identical (shouldn't happen with GAE)
                advantages = advantages - advantages.mean()

        policy_loss = -(log_probs * advantages).mean()
        critic_loss = self.value_coef * (returns.detach() - values).pow(2).mean()
        entropy_loss = -self.entropy_coef * entropies.mean()

        loss = policy_loss + critic_loss + entropy_loss
        loss.backward()

        grad_norm = torch.nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=self.max_grad_norm)
        self.policy.optimizer.step()

        self.learn_step_cnt += 1

        if self.learn_step_cnt % 20 == 0:
            print({
                "ep": self.episode_count,
                "buffer": len(rewards),
                "entropy": f"{entropies.mean().item():.3f}",
                "adv": f"{advantages.mean().item():.2f}±{advantages.std().item():.2f}",
                "V": f"{values.mean().item():.1f}±{values.std().item():.1f}",
                "ret": f"{returns.mean().item():.1f}±{returns.std().item():.1f}",
                "R_sum": f"{rewards.sum().item():.0f}",
                "pi_loss": f"{policy_loss.item():.3f}",
                "V_loss": f"{critic_loss.item():.3f}",
                "grad": f"{grad_norm.item():.1f}"
            })

        # Buffer management
        if self.last_done_flag():
            self.episode_count += 1
            self.clear_rollout()
        elif self.mode == 'n_step':
            self.pop_left()
        elif self.mode == 'hybrid' and self.episode_count >= self.warmup_episodes:
            if len(self.state_memory) >= self.n_steps * 2:
                self.clear_rollout()