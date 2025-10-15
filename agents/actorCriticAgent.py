import time
import numpy as np
import torch
import matplotlib.pyplot as plt

from datetime import datetime
from utils import sanitize_file_string
from collections import deque

from networks.actorCriticNetwork import ActorCriticNetwork
from agents.common.FlexibleBufferAgent import FlexibleBufferAgent
from utils import MovingAvg





class ActorCriticAgent(FlexibleBufferAgent):
    """
    Actor Critic Agent with flexible learning modes.
    """

    def __init__(self, model_name="A2C", n_steps=20, value_coef=1,
                 max_grad_norm=0.5, entropy_coef=0.001, lr=2.5e-4, mode='n_step', 
                 warmup_episodes=50, n_envs: int = 1, vectorize_env: bool = False,
                 *args, **kwargs):
        
        super().__init__(n_steps=n_steps, mode=mode, warmup_episodes=warmup_episodes, *args, **kwargs)
        self.model_name = model_name
        self.n_steps = n_steps
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.mode = mode
        self.warmup_episodes = warmup_episodes
        self.max_grad_norm = max_grad_norm

        # Multi-env values
        self.vectorize_env = vectorize_env
        self.n_envs = n_envs

        # Model saving
        self.checkpoint_dir = f"models/{model_name}/{sanitize_file_string(self.env_name)}"
        self.filename_root = f"{model_name}_{sanitize_file_string(self.env_name)}_lr{lr}_gamma{self.gamma}_eps{self.epsilon}_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"


        # Network
        self.policy = ActorCriticNetwork(
            n_actions=self.n_actions,
            input_dims=self.input_dims,
            file_name=self.filename_root + "_eval",
            lr=lr
        )
        self.networks = [self.policy]


                
        # =========================
        # Tracking Metrics
        # =========================
        self.start_time = time.time()
        self.recent_returns = deque(maxlen=100)
        self.ep_ret = np.zeros(self.n_envs, dtype=np.float32)


        self.hist_steps   = []
        self.hist_rmean   = []   # mean100 episode reward
        self.hist_pi_loss = []
        self.hist_v_loss  = []
        self.hist_ent     = []
        self.ma_pi  = MovingAvg(k=20)
        self.ma_v   = MovingAvg(k=20)
        self.ma_ent = MovingAvg(k=20)


    def is_ready(self):
        if self.vectorize_env:
            return len(self.state_memory[0]) >= self.n_steps or self.terminal_memory[-1].any()
        """Override to handle warmup logic."""
        if self.mode == 'episode':
            return self.last_done_flag()
        
        elif self.mode == 'n_step':
            print("mem length: ", len(self.state_memory))
            return len(self.state_memory) >= self.n_steps or self.last_done_flag()
        
        return False

    def choose_action(self, state):
        state = torch.tensor(state, dtype=torch.float).to(device=self.policy.device)
        if not self.vectorize_env:
            state = state.unsqueeze(0)

        with torch.no_grad():
            actor_logits, _ = self.policy(state)

        dist = torch.distributions.Categorical(logits=actor_logits)
        action = dist.sample()

        if not self.vectorize_env:
            return action.item()
        else:
            return action.cpu().numpy()
    
    
    def _compute_gae(
        self,
        rewards,        
        truncateds,     
        terminals,      
        values,         
        next_values,
        gamma=0.99,
        gae_lambda=1,
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
        done = terminals | truncateds if treat_truncation_as_terminal else terminals

        # Make 2D time-major views: [T, N]
        T = rewards.shape[0]
        N = int(rewards.numel() // T)
        rew  = rewards.view(T, N)
        val  = values.view(T, N)
        d    = done.view(T, N).float()
        # last_values must be [N]
        if next_values.ndim == 0:
            next_values = next_values.unsqueeze(0)
        lastv = next_values.view(N)

        adv = torch.zeros_like(rew)
        last_gae = torch.zeros(N, dtype=torch.float32, device=device)

        # Backward-time recursion
        for t in range(T - 1, -1, -1):
            v_next = lastv if t == T - 1 else val[t + 1]
            delta = rew[t] + gamma * v_next * (1.0 - d[t]) - val[t]
            last_gae = delta + gamma * gae_lambda * (1.0 - d[t]) * last_gae
            adv[t] = last_gae

        ret = adv + val

        # Restore original shapes
        returns    = ret.view_as(values)
        advantages = adv.view_as(values)

        return returns, advantages

    def learn(self):
        if not self.is_ready():
            return
        self.start_time = time.time()

        states, actions, rewards, next_states, truncateds, terminals = self.sample_rollout()

        if self.vectorize_env:
            states = states.squeeze(0)
            actions = actions.squeeze(0)
            rewards = rewards.squeeze(0) # This line was already corrected in the last diff
            next_states = next_states.squeeze(0)
            truncateds = truncateds.squeeze(0)
            terminals = terminals.squeeze(0)

            states = states.flatten(0, 1)
            actions = actions.flatten(0, 1)

        with torch.no_grad():
            if self.vectorize_env:
                _, next_values = self.policy(next_states[-1])
            else:
                _, next_values = self.policy(next_states)

        logits, values = self.policy(states)

        dist = torch.distributions.Categorical(logits=logits)
        log_probs = dist.log_prob(actions)
        entropies = dist.entropy()

        # Use GAE for stable advantage estimation
        returns, advantages = self._compute_gae(
            rewards, truncateds, terminals,
            values.detach(), next_values,
            self.gamma, gae_lambda=1.0
        )
        
        if self.vectorize_env:
            rewards = rewards.sum(axis=0)
            truncateds = truncateds.any(axis=0)
            terminals = terminals.any(axis=0)
            advantages = advantages.flatten(0, 1)
            returns = returns.flatten(0, 1)
        
        
        with torch.no_grad():
            # values is [B], flat_returns is [B]
            # Move to CPU for numpy ops if you like, but torch works fine:
            vr = returns
            vpred = values.squeeze(-1)
            var_y = vr.var(unbiased=False)
            # print(((vr - vpred).var(unbiased=False) / (var_y + 1e-8)))
            ev = 1.0 - ((vr - vpred).var(unbiased=False) / (var_y + 1e-8))
            # print(ev)



        # Normalize advantages
        if len(advantages) > 1:
        # Check for pathological cases
            if advantages.std() > 1e-6:
                advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)
            else:
                # Rare: all advantages identical (shouldn't happen with GAE)
                advantages = advantages - advantages.mean()

        policy_loss = -(log_probs * advantages).mean()
        value_loss = self.value_coef * (returns.detach() - values).pow(2).mean()
        entropy_loss = -self.entropy_coef * entropies.mean()

        loss = policy_loss + value_loss + entropy_loss
        self.policy.optimizer.zero_grad()
        loss.backward()

        total_grad_norm = self.calc_total_grad_norm()

        grad_norm = torch.nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=self.max_grad_norm)
        self.policy.optimizer.step()

        step_size = self.n_envs if (self.vectorize_env and self.n_envs) else 1
        self.learn_step_cnt += step_size

        dones = torch.logical_or(terminals, truncateds).cpu().numpy()

        self.ep_ret += rewards.cpu().numpy()

        if dones.any():
                finished = np.where(dones)[0]
                for i in finished:
                    self.recent_returns.append(float(self.ep_ret[i]))
                    # reset counters for that sub-env
                    self.ep_ret[i] = 0.0

        r_mean100 = float(np.mean(self.recent_returns)) if len(self.recent_returns) > 0 else np.nan

        self.hist_steps.append(self.learn_step_cnt)
        self.hist_rmean.append(r_mean100)
        self.hist_pi_loss.append(self.ma_pi.add(policy_loss.item()))
        self.hist_v_loss.append(self.ma_v.add(value_loss.item()))
        self.hist_ent.append(self.ma_ent.add((-entropy_loss).item())) 


        if self.learn_step_cnt % 20 == 0:
            # print({
            #     "ep": self.episode_count,
            #     "buffer": len(rewards),
            #     "entropy": f"{entropies.mean().item():.3f}",
            #     "adv": f"{advantages.mean().item():.2f}±{advantages.std().item():.2f}",
            #     "V": f"{values.mean().item():.1f}±{values.std().item():.1f}",
            #     "ret": f"{returns.mean().item():.1f}±{returns.std().item():.1f}",
            #     "R_sum": f"{rewards.sum().item():.0f}",
            #     "pi_loss": f"{policy_loss.item():.3f}",
            #     "V_loss": f"{value_loss.item():.3f}",
            #     "grad": f"{grad_norm.item():.1f}"
            # })
            
            sps = int(self.learn_step_cnt / (time.time() - self.start_time))

            if len(self.recent_returns) > 0:
                r_last   = self.recent_returns[-1]
                r_mean   = float(np.mean(self.recent_returns))
                r_median = float(np.median(self.recent_returns))
                r_count  = len(self.recent_returns)
                reward_msg = f"R/last={r_last:.1f} | R/mean100={r_mean:.1f} | R/med100={r_median:.1f} | ep_count={r_count}"
            else:
                reward_msg = "R/last=NA | R/mean100=NA | R/med100=NA | ep_count=0"

            print(
                f"steps={self.learn_step_cnt:,} | "
                f"loss={loss.item():.3f} | pi={policy_loss.item():.3f} | "
                f"v_loss={value_loss.item():.3f} | ent={-entropy_loss.item():.3f} | "
                f"sps={sps} | {reward_msg}"
                f" | grad_norm={total_grad_norm:.3f} | ev={ev:.3f}"
            )

        if self.learn_step_cnt % 100_000 == 0:
            _, axs = plt.subplots(2, 1, figsize=(9, 8), sharex=True)

            # ----------------------------
            # (1) Episodic reward curve
            # ----------------------------
            axs[0].plot(self.hist_steps, self.hist_rmean, color="tab:blue", label="Reward (mean100)")
            axs[0].set_ylabel("Reward")
            axs[0].set_title("A2C Training — Reward")
            axs[0].grid(True, linewidth=0.3)
            axs[0].legend(loc="lower right")

            # ----------------------------
            # (2) Losses + entropy curve
            # ----------------------------
            axs[1].plot(self.hist_steps, self.hist_pi_loss, label="Policy loss (MA)", color="tab:orange", alpha=0.9)
            axs[1].plot(self.hist_steps, self.hist_v_loss, label="Value loss (MA)", color="tab:green", alpha=0.9)
            axs[1].plot(self.hist_steps, self.hist_ent, label="Entropy (MA)", color="tab:red", alpha=0.9)
            axs[1].set_xlabel("Env steps")
            axs[1].set_ylabel("Loss / Entropy")
            axs[1].set_title(f"{self.model_name} — Losses & Entropy")
            axs[1].grid(True, linewidth=0.3)
            axs[1].legend(loc="upper right")

            plt.tight_layout()
            #TODO make this filepath dynamc
            plt.savefig(f"models/{self.model_name}/figures/{self.filename_root}.png", dpi=130)
            plt.close()

        # Buffer management
        if self.last_done_flag():
            self.episode_count += 1
            self.clear_rollout()
        elif self.mode == 'n_step':
            self.pop_left()
        elif self.mode == 'hybrid' and self.episode_count >= self.warmup_episodes:
            if len(self.state_memory) >= self.n_steps * 2:
                self.clear_rollout()

    def calc_total_grad_norm(self):
        total_grad_norm = 0.0
        for p in self.policy.parameters():
            if p.grad is not None:
                total_grad_norm += p.grad.data.norm(2).item() ** 2
        total_grad_norm = total_grad_norm ** 0.5

        return total_grad_norm

    