# a2c_cartpole.py
import time
from dataclasses import dataclass

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

import math
import matplotlib.pyplot as plt
from collections import deque
import os

# -------------------------
# Checkpointing
# -------------------------
CHECKPOINT_DIR = "models/a2c_cartpole_checkpoints"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
MODEL_PATH = os.path.join(CHECKPOINT_DIR, "a2c_cartpole_model.pt")
OPT_PATH   = os.path.join(CHECKPOINT_DIR, "a2c_cartpole_optimizer.pt")

# =========================
# Hyperparameters
# =========================
ENV_ID         = "CartPole-v1"
TOTAL_STEPS    = 1_000_000
N_ENVS         = 16
N_STEPS        = 5                    # rollout horizon
GAMMA          = 0.99
GAE_LAMBDA     = 1.0
ENTROPY_COEF   = 0.01
VALUE_COEF     = 0.5
LR             = 7e-4
RMS_ALPHA      = 0.99
RMS_EPSILON    = 1e-5
MAX_GRAD_NORM  = 0.5
DEVICE         = torch.device("cpu") #torch.device("mps" if torch.backends.mps.is_available() else 'cuda:0' if torch.cuda.is_available() else 'cpu')
SEED           = 1

torch.backends.cudnn.benchmark = True

# =========================
# Env creation (CartPole)
# =========================
def make_cartpole_env(env_id: str, seed: int, idx: int):
    def _thunk():
        env = gym.make(env_id)
        env.reset(seed=seed + idx)
        env.action_space.seed(seed + idx)
        env.observation_space.seed(seed + idx)
        return env
    return _thunk

def make_vec_env(env_id: str, n_envs: int, seed: int):
    thunks = [make_cartpole_env(env_id, seed, i) for i in range(n_envs)]
    return gym.vector.SyncVectorEnv(thunks)

# =========================
# Model (MLP)
# =========================
class MlpA2C(nn.Module):
    """
    Small MLP torso + separate actor/critic heads for low-dim obs.
    Expects input as float [B, obs_dim].
    """
    def __init__(self, obs_dim: int, n_actions: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 128), nn.ReLU(inplace=True),
            nn.Linear(128, 128),     nn.ReLU(inplace=True),
        )
        self.policy = nn.Linear(128, n_actions)
        self.value  = nn.Linear(128, 1)

    def forward(self, x):
        z = self.net(x)
        logits = self.policy(z)
        value  = self.value(z).squeeze(-1)
        return logits, value

    def act(self, x):
        logits, value = self.forward(x)
        dist = torch.distributions.Categorical(logits=logits)
        action = dist.sample()
        logprob = dist.log_prob(action)
        entropy = dist.entropy()
        return action, logprob, entropy, value

# =========================
# Helpers
# =========================
def obs_to_tensor(obs_np) -> torch.Tensor:
    """
    Accepts vectorized obs as either:
      - [N, obs_dim]
      - [obs_dim]
    Returns float32 with shape [N, obs_dim].
    """
    x = torch.as_tensor(obs_np, device=DEVICE, dtype=torch.float32)
    if x.ndim == 1:
        x = x.unsqueeze(0)
    return x

def vector_reset_done(env, next_obs, done):
    """Compatibility reset for vector envs without `reset_done`."""
    if hasattr(env, "reset_done"):
        return env.reset_done(next_obs, done)
    done_idx = np.where(done)[0]
    if len(done_idx) > 0:
        for i in done_idx:
            obs_i, _ = env.envs[i].reset()
            next_obs[i] = np.asarray(obs_i)
    return next_obs, {}

@torch.no_grad()
def bootstrap_value(model: nn.Module, obs_tensor: torch.Tensor) -> torch.Tensor:
    _, v = model(obs_tensor)
    return v  # [N_ENVS]

def compute_gae(rewards, dones, values, last_values, gamma, gae_lambda):
    """
    rewards:    [T, N]
    dones:      [T, N]  (1.0 where episode ended at t)
    values:     [T, N]  critic V(s_t) estimated during rollout (no grad)
    last_values:[N]     V(s_{T}) for bootstrap
    returns:    [T, N]
    advantages: [T, N]
    """
    T, N = rewards.shape
    advantages = torch.zeros_like(rewards)
    last_gae = torch.zeros(N, device=rewards.device, dtype=rewards.dtype)
    for t in reversed(range(T)):
        v_next = last_values if t == T - 1 else values[t + 1]
        delta = rewards[t] + gamma * v_next * (1.0 - dones[t]) - values[t]
        last_gae = delta + gamma * gae_lambda * (1.0 - dones[t]) * last_gae
        advantages[t] = last_gae
    returns = advantages + values
    return returns, advantages

# Smoothing helper for logs
class MovingAvg:
    def __init__(self, k=20):
        self.buf = deque(maxlen=k)
    def add(self, x):
        self.buf.append(float(x))
        return self.value()
    def value(self):
        if not self.buf: return math.nan
        return sum(self.buf) / len(self.buf)

# =========================
# Training
# =========================
def main():
    env = make_vec_env(ENV_ID, N_ENVS, SEED)
    obs, _ = env.reset(seed=SEED)
    obs_t = obs_to_tensor(obs)

    n_actions = env.single_action_space.n
    obs_dim   = env.single_observation_space.shape[0]
    model = MlpA2C(obs_dim, n_actions).to(DEVICE)
    optimizer = optim.RMSprop(model.parameters(), lr=LR, alpha=RMS_ALPHA, eps=RMS_EPSILON, weight_decay=0)

    global_steps = 0
    start_time = time.time()

    # Rolling stats
    recent_returns = deque(maxlen=100)
    ep_ret = np.zeros(N_ENVS, dtype=np.float32)
    ep_len = np.zeros(N_ENVS, dtype=np.int32)

    hist_steps   = []
    hist_rmean   = []
    hist_pi_loss = []
    hist_v_loss  = []
    hist_ent     = []
    ma_pi  = MovingAvg(k=20)
    ma_v   = MovingAvg(k=20)
    ma_ent = MovingAvg(k=20)

    while global_steps < TOTAL_STEPS:
        # -------- Rollout buffers --------
        obs_buf     = torch.empty((N_STEPS, N_ENVS, obs_dim), dtype=torch.float32, device=DEVICE)
        actions_buf = torch.empty((N_STEPS, N_ENVS),          dtype=torch.long,    device=DEVICE)
        rew_buf     = torch.empty((N_STEPS, N_ENVS),          dtype=torch.float32, device=DEVICE)
        done_buf    = torch.empty((N_STEPS, N_ENVS),          dtype=torch.float32, device=DEVICE)
        val_buf     = torch.empty((N_STEPS, N_ENVS),          dtype=torch.float32, device=DEVICE)

        for t in range(N_STEPS):
            obs_buf[t] = obs_t  # store float32 features

            # act WITHOUT building a graph (sampling step)
            with torch.no_grad():
                logits, v_t = model(obs_t)
                dist = torch.distributions.Categorical(logits=logits)
                action = dist.sample()
                val_buf[t] = v_t

            actions_buf[t] = action

            # env step
            next_obs, reward, terminated, truncated, _ = env.step(action.detach().cpu().numpy())
            done = np.logical_or(terminated, truncated)

            rew_buf[t]  = torch.as_tensor(reward, dtype=torch.float32, device=DEVICE)
            done_buf[t] = torch.as_tensor(done,   dtype=torch.float32, device=DEVICE)

            # reset finished sub-envs
            next_obs, _ = vector_reset_done(env, next_obs, done)

            # next obs tensor
            obs_t = obs_to_tensor(next_obs)
            global_steps += N_ENVS

            # episodic logging state update
            ep_ret += reward.astype(np.float32)
            ep_len += 1

            # Record finished episodes
            if done.any():
                finished = np.where(done)[0]
                for i in finished:
                    recent_returns.append(float(ep_ret[i]))
                    ep_ret[i] = 0.0
                    ep_len[i] = 0

        # -------- Bootstrap value --------
        with torch.no_grad():
            _, last_v = model(obs_t)  # [N_ENVS]

        # -------- Returns & advantages --------
        returns, advantages = compute_gae(rew_buf, done_buf, val_buf, last_v, GAMMA, GAE_LAMBDA)  # [T,N]

        # -------- Recompute logits/values with grad on the whole rollout --------
        flat_obs = obs_buf.reshape(N_STEPS * N_ENVS, obs_dim)   # [B, obs_dim]
        logits, values = model(flat_obs)                        # values: [B]
        dist = torch.distributions.Categorical(logits=logits)
        flat_actions = actions_buf.reshape(-1)                  # [B]
        logprobs = dist.log_prob(flat_actions)                  # [B]
        entropies = dist.entropy()                              # [B]

        flat_adv     = advantages.reshape(-1).detach()
        flat_returns = returns.reshape(-1)

        with torch.no_grad():
            vr = flat_returns
            vpred = values
            var_y = vr.var(unbiased=False)
            ev = 1.0 - ((vr - vpred).var(unbiased=False) / (var_y + 1e-8))

        # normalize advantages
        flat_adv = (flat_adv - flat_adv.mean()) / (flat_adv.std(unbiased=False) + 1e-8)

        # -------- Losses --------
        policy_loss  = -(logprobs * flat_adv).mean()
        value_loss   = 0.5 * (flat_returns - values).pow(2).mean()
        entropy_loss = -entropies.mean()

        loss = policy_loss + VALUE_COEF * value_loss + ENTROPY_COEF * entropy_loss

        optimizer.zero_grad()
        loss.backward()

        total_grad_norm = 0.0
        for p in model.parameters():
            if p.grad is not None:
                total_grad_norm += p.grad.data.norm(2).item() ** 2
        total_grad_norm = total_grad_norm ** 0.5

        nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
        optimizer.step()

        cur_step = global_steps
        r_mean100 = float(np.mean(recent_returns)) if len(recent_returns) > 0 else np.nan

        hist_steps.append(cur_step)
        hist_rmean.append(r_mean100)
        hist_pi_loss.append(ma_pi.add(policy_loss.item()))
        hist_v_loss.append(ma_v.add(value_loss.item()))
        hist_ent.append(ma_ent.add((-entropy_loss).item()))

        if cur_step % 100_000 == 0:
            fig, axs = plt.subplots(2, 1, figsize=(9, 8), sharex=True)

            # (1) Episodic reward curve
            axs[0].plot(hist_steps, hist_rmean, label="Reward (mean100)")
            axs[0].set_ylabel("Reward")
            axs[0].set_title("A2C Training — Reward")
            axs[0].grid(True, linewidth=0.3)
            axs[0].legend(loc="lower right")

            # (2) Losses + entropy curve
            axs[1].plot(hist_steps, hist_pi_loss, label="Policy loss (MA)", alpha=0.9)
            axs[1].plot(hist_steps, hist_v_loss,  label="Value loss (MA)",  alpha=0.9)
            axs[1].plot(hist_steps, hist_ent,     label="Entropy (MA)",     alpha=0.9)
            axs[1].set_xlabel("Env steps")
            axs[1].set_ylabel("Loss / Entropy")
            axs[1].set_title("A2C Training — Losses & Entropy")
            axs[1].grid(True, linewidth=0.3)
            axs[1].legend(loc="upper right")

            plt.tight_layout()
            plt.savefig("training_curves_cartpole.png", dpi=130)
            plt.close()

        if cur_step % 500_000 == 0:  # save every 500k env steps
            torch.save(model.state_dict(), MODEL_PATH)
            torch.save(optimizer.state_dict(), OPT_PATH)
            print(f"✅ Saved checkpoint at {cur_step:,} steps")

        # -------- Logging --------
        if (global_steps // (N_ENVS * N_STEPS)) % 20 == 0:
            sps = int(global_steps / (time.time() - start_time))
            if len(recent_returns) > 0:
                r_last   = recent_returns[-1]
                r_mean   = float(np.mean(recent_returns))
                r_median = float(np.median(recent_returns))
                r_count  = len(recent_returns)
                reward_msg = f"R/last={r_last:.1f} | R/mean100={r_mean:.1f} | R/med100={r_median:.1f} | ep_count={r_count}"
            else:
                reward_msg = "R/last=NA | R/mean100=NA | R/med100=NA | ep_count=0"

            print(
                f"steps={global_steps:,} | "
                f"loss={loss.item():.3f} | pi={policy_loss.item():.3f} | "
                f"v={value_loss.item():.3f} | ent={-entropy_loss.item():.3f} | "
                f"sps={sps} | {reward_msg}"
                f" | grad_norm={total_grad_norm:.3f} | ev={ev:.3f}"
            )

    env.close()

if __name__ == "__main__":
    main()
