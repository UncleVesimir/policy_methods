import numpy as np
from datetime import datetime
from utils import sanitize_file_string
import torch

from networks.actorCriticNetwork import ActorCriticNetwork
from agents.common.SlidingWindowBufferAgent import SlidingWindowBufferAgent



class ActorCriticAgent(SlidingWindowBufferAgent):
    """
        Actor Critic Agent
        Based on Williams, R. J. (1992). Simple statistical gradient-following algorithms for connectionist reinforcement learning.
    """

    def __init__(self, model_name="A2C", n_steps=6, value_coef=0.5, entropy_coef=0.03, lr=2.5e-4, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.n_steps = max(1, n_steps)
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef

        self.checkpoint_dir = f"models/{model_name}/{sanitize_file_string(self.env_name)}"
        self.filename_root = f"{model_name}_{sanitize_file_string(self.env_name)}_lr{lr}_gamma{self.gamma}_eps{self.epsilon}_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"

        self.policy = ActorCriticNetwork(
            n_actions=self.n_actions,
            input_dims=self.input_dims,
            file_name=self.filename_root + "_eval",
            lr=lr
        )


        self.networks = [self.policy] # for savings model dicts

    def choose_action(self, state):
        state = torch.tensor(state, dtype=torch.float).unsqueeze(0).to(device=self.policy.device)
        with torch.no_grad():
            actor_logits, _ = self.policy(state)

        dist = torch.distributions.Categorical(logits=actor_logits)
        action = dist.sample()

        return action.item()
    
    def _compute_n_step_returns(self, rewards, truncateds, terminals,
                                V, V_next=None, n_steps=1, gamma=0.99,
                                treat_truncation_as_terminal=False):
        T = len(rewards)
        returns = torch.zeros_like(rewards)

        for t in range(T):
            ret = 0.0
            discount = 1.0
            ended = False
            for i in range(n_steps):
                k = t + i
                if k >= T:
                    break
                ret += discount * rewards[k]

                # stop at terminal
                if terminals[k]:
                    ended = True
                    break

                # optionally stop at truncation as if terminal (time-limit)
                if truncateds[k] and treat_truncation_as_terminal:
                    ended = True
                    break

                discount *= gamma

            if not ended:
                k = t + n_steps
                if k < T:
                    # bootstrap from V(s_{t+n})
                    ret += discount * V[k]
                else:
                    # window ran off the right edge of this rollout
                    if truncateds[-1] and not treat_truncation_as_terminal and V_next is not None:
                        # bootstrap from value at the env-continued next state
                        ret += discount * V_next[-1]
                    # else: true terminal at the end → no bootstrap

            returns[t] = ret
        return returns

    
    def learn(self):
        if not self.is_ready():
            return
        
        self.policy.optimizer.zero_grad()

        states, actions, rewards, next_states, truncateds, terminals = self.sample_rollout()

        logits, values      = self.policy(states)           # logits: [T, A], values: [T, 1] or [T]

        with torch.no_grad():
            _,     next_values  = self.policy(next_states)      # next_values: [T, 1] or [T]

        values = values.squeeze(-1)
        next_values = next_values.squeeze(-1)

        # log_probs for taken actions (vectorized)
        dist = torch.distributions.Categorical(logits=logits)
        log_probs = dist.log_prob(actions)                  # [T]
        entropies = dist.entropy()      
        
        targets = self._compute_n_step_returns(
            rewards,
            truncateds,
            terminals,
            values.detach(),
            next_values,
            self.n_steps,
            self.gamma,
            treat_truncation_as_terminal=True
        )

        advantages = (targets - values).detach()
        advantages = (advantages - advantages.mean())/ (advantages.std() + 1e-8)


        policy_loss = (-log_probs * advantages).mean()
        critic_loss = self.value_coef * (targets - values).pow(2).mean()

        entropy_loss = -self.entropy_coef * entropies.mean()

        loss = policy_loss + critic_loss + entropy_loss
        loss.backward()

        # Clip gradients to prevent large updates
        grad_norm = torch.nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=0.5)
        self.policy.optimizer.step()

        self.learn_step_cnt += 1

        if self.learn_step_cnt % 20 == 0:
            print({
                "entropy": entropies.mean().item(),
                "adv_mean": advantages.mean().item(),
                "adv_std": advantages.std().item(),
                "V_mean": values.mean().item(),
                "V_std": values.std().item(),
                "pi_loss": policy_loss.item(),
                "V_loss": critic_loss.item(),
                "grad_norm": grad_norm.item()
            })

        if self.last_done_flag():
            self.clear_rollout()
        else:
            self.pop_left()