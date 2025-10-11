# train.py
import os
from datetime import datetime
from typing import Optional

import gymnasium as gym
import ale_py
import numpy as np
import torch
from gymnasium.wrappers import AtariPreprocessing, TransformObservation, FrameStackObservation

import agents.Agents as Agents  # your agents package (adjust import path if needed)
from utils import plot_learning_curve, sanitize_file_string

import arguably
import json


def _make_env(
    env_name: str,
    render: bool,
    screen_size: int,
    grayscale: bool,
    frame_skip: int,
    noop_max: int,
    terminal_on_life_loss: bool,
    scale_obs: bool,
) -> gym.Env:
    """Builds an Atari env with preprocessing -> (4, 84, 84) float32 in [0,1]."""
    # Important: disable base frameskip so the wrapper controls it
    env = gym.make(env_name, render_mode=("human" if render else None), frameskip=1)
    env = AtariPreprocessing(
        env,
        screen_size=screen_size,
        grayscale_obs=grayscale,
        frame_skip=frame_skip,
        noop_max=noop_max,
        terminal_on_life_loss=terminal_on_life_loss,
        scale_obs=scale_obs,
    )
    # Stack along channel axis (CHW): (4, H, W) uint8
    env = FrameStackObservation(env, stack_size=4)
    # Normalize only (already CHW)
    env = TransformObservation(
        env,
        lambda o: np.asarray(o, dtype=np.float32) / 255.0,
        observation_space=gym.spaces.Box(
            low=0.0, high=1.0, shape=(4, screen_size, screen_size), dtype=np.float32
        ),
    )
    return env


def _figure_path(models_dir: str, model_type: str, env_name: str, lr: float, gamma: float, eps: float) -> str:
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    safe_env = sanitize_file_string(env_name)
    fig_dir = os.path.join(models_dir, model_type, "figures")
    os.makedirs(fig_dir, exist_ok=True)
    return os.path.join(
        fig_dir, f"{model_type}_{safe_env}_lr{lr}_gamma{gamma}_eps{eps}_{ts}.png"
    )


def _ensure_model_dir(models_dir: str, model_type: str, env_name: str):
    base = os.path.join(models_dir, model_type, sanitize_file_string(env_name))
    os.makedirs(base, exist_ok=True)
    return base


def _checkpoint_path(models_dir: str, model_type: str, env_name: str, lr: float, gamma: float) -> str:
    safe_env = sanitize_file_string(env_name)
    stats_dir = os.path.join(models_dir, model_type, "stats")
    os.makedirs(stats_dir, exist_ok=True)
    # Note: No timestamp, so we can resume from it.
    filename = f"{model_type}_{safe_env}_lr{lr}_gamma{gamma}_stats.json"
    return os.path.join(stats_dir, filename)


@arguably.command
def train(
    *,
    env: str = "ALE/Pong-v5",
    episodes: int = 10_000,
    render: bool = False,
    model: str = "REINFORCE",  # choices: Double_DQN, DQN, etc. (must exist in your Agents module)
    epsilon: float = 1.0,
    gamma: float = 0.99,
    lr: float = 1e-4,
    batch_size: int = 32,
    replace_target_every: int = 1000,
    models_dir: str = "models",
    mem_size: int = 100000,
    window_step: int = 5,
    # Atari preprocessing knobs:
    screen_size: int = 84,
    grayscale: bool = True,
    frame_skip: int = 4,
    noop_max: int = 30,
    terminal_on_life_loss: bool = False,
    scale_obs: bool = False,
    # Resume:
    load_model_checkpoint: Optional[bool] = False,
    resume_training: Optional[bool] = False,
):
    """
    Train a (Double) DQN agent on an Atari environment.

    Args:
        env: Gymnasium env id (e.g., ALE/Pong-v5)
        episodes: Number of episodes to train
        render: If True, render with 'human' (slow)
        model: Model type from your Agents module (e.g., 'Double_DQN' or 'DQN')
        epsilon: Initial epsilon for ε-greedy
        gamma: Discount factor
        lr: Learning rate
        batch_size: Batch size for updates
        replace_target_every: Hard update frequency for the target net
        models_dir: Root directory for model files & figures
        mem_size: Replay buffer size
        window_step: Step size for n_step returns (only for n_step agents)
        screen_size: Preprocess to square size (default 84)
        grayscale: Use grayscale frames
        frame_skip: Frameskip inside AtariPreprocessing (base env frameskip is set to 1)
        noop_max: Random no-op at reset (exploration)
        terminal_on_life_loss: Treat life loss as terminal (Atari)
        scale_obs: If True, AtariPreprocessing scales to [0,1] (we already normalize later)
        load_model_checkpoint: Optional flag to load previous models for inference ONLY (choose in cli)
        resume_training: Optional flag to load previous models for further training (choose in cli)
    """
    # --- Env & figure file ---
    env_obj = _make_env(
        env_name=env,
        render=render,
        screen_size=screen_size,
        grayscale=grayscale,
        frame_skip=frame_skip,
        noop_max=noop_max,
        terminal_on_life_loss=terminal_on_life_loss,
        scale_obs=scale_obs,
    )

    fig_path = _figure_path(models_dir, model, env, lr, gamma, epsilon)
    _ensure_model_dir(models_dir, model, env)
    checkpoint_file = _checkpoint_path(models_dir, model, env, lr, gamma)

    # --- Agent selection (expects names in Agents) ---
    Agent = Agents.agents_dict.get(model)
    if Agent is None:
        raise ValueError(f"Unknown model '{model}'. Available: {list(Agents.agents_dict.keys())}")


    agent = Agent(
        n_actions=env_obj.action_space.n,
        input_dims=env_obj.observation_space.shape,
        env_name=env,
        epsilon=epsilon,
        gamma=gamma,
        learning_rate=lr,
        batch_size=batch_size,
        replace_limit=replace_target_every,
        mem_size=mem_size,
        n_step=window_step
    )

    # --- State Initialization ---
    scores, eps_history, steps_array = [], [], []
    n_steps, best_score = 0, -np.inf
    start_episode = 0

    if resume_training and os.path.exists(checkpoint_file):
        print(f"Resuming training from {checkpoint_file}")
        with open(checkpoint_file, 'r') as f:
            state = json.load(f)
        agent.load_models()
        start_episode = state['episode'] + 1
        scores = state['scores']
        eps_history = state['eps_history']
        steps_array = state['steps_array']
        n_steps = state['n_steps']
        agent.epsilon = state['epsilon']
        best_score = state.get('best_score', -np.inf)
    elif load_model_checkpoint:
        agent.load_models()


    train_start_datetime = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

    print(f"Commencing {model} training on {env} at {train_start_datetime}")
    for i in range(start_episode, episodes):
        state, _ = env_obj.reset()
        score, done = 0.0, False


        while not done:
            action = agent.choose_action(state)
            next_state, reward, terminated, truncated, _info = env_obj.step(action)
            done = terminated or truncated
            score += float(reward)

            if render:
                env_obj.render()
            if not load_model_checkpoint:
                agent.store_transition(state, action, reward, next_state, truncated, terminated)
                agent.learn()

            state = next_state
            n_steps += 1

        scores.append(score)
        steps_array.append(n_steps)
        avg_score = float(np.mean(scores[-100:])) if scores else score

        print(
            f"episode: {i} | score: {score:.1f} | avg(100): {avg_score:.1f} "
            f"| best: {best_score:.1f} | epsilon: {agent.epsilon:.2f} | steps: {n_steps}"
        )

        if avg_score > best_score:
            best_score = avg_score
            if not load_model_checkpoint:
                agent.save_models()

        eps_history.append(agent.epsilon)

        if not load_model_checkpoint:
            training_state = {
                'episode': i,
                'scores': scores,
                'eps_history': eps_history,
                'steps_array': steps_array,
                'n_steps': n_steps,
                'epsilon': agent.epsilon,
                'best_score': best_score
            }
            with open(checkpoint_file, 'w') as f:
                json.dump(training_state, f, indent=4)

    plot_learning_curve(steps_array, scores, eps_history, filename=fig_path)
    print(f"Saved learning curve to: {fig_path}")


if __name__ == "__main__":
    arguably.run()