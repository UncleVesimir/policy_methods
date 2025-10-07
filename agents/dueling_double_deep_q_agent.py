from collections import defaultdict
import torch
import numpy as np
from datetime import datetime
from utils import sanitize_file_string


from agents.common.PrioritizedReplayBufferAgent import PERBufferAgent
from networks.DuelingDeepQNetwork import DuelingDeepQNetwork


class DuelingDoubleDeepQAgent(PERBufferAgent):
    """
    Dueling Double DQN

    Implements a Double DQN agent with a dueling head (Wang et al., 2015).
    The Q-function is decomposed into a state-value stream V(s) and an
    advantage stream A(s, a), combined with an identifiability constraint:

        Q(s, a) = V(s) + [ A(s, a) - mean_a A(s, a) ]

    Key ideas
    ---------
    • Dueling head (representation):
        - V(s): captures how good a state is regardless of action.
        - A(s, a): captures how much better/worse an action is relative to others in that state.
        - Mean-subtraction enforces E_a[A(s,a)] = 0 so V and A are identifiable and stable.

    • Double DQN (estimation):
        - Action selection uses the online (eval) network:
              a* = argmax_a Q_online(s', a)
        - Action evaluation uses the target network:
              y = r + γ (1 - done) · Q_target(s', a*)
        - This reduces overestimation bias vs. single-network max-backups.

    Why dueling helps
    -----------------
    • In many states, actions have similar effects; learning V(s) directly improves value
      estimates there, while A(s,a) focuses on relative action differences.
    • Leads to better sample efficiency and more robust learning in action-insensitive states.

    Notes / Pitfalls
    ----------------
    • Use the same dueling head in BOTH online and target networks.
    • Prefer mean-subtraction to max-subtraction for stability.
    • Ensure the combine op is part of the forward pass; forgetting it reintroduces
      the V/A unidentifiability problem.

    References
    ----------
    • Dueling Network Architectures for Deep Reinforcement Learning (Wang et al., 2015)
    • Deep Reinforcement Learning with Double Q-learning (van Hasselt et al., 2015)
    """

    def __init__(self, *, model_name="Double_DQN", **kwargs):
        super().__init__(**kwargs)

        self.checkpoint_dir = f"models/{model_name}/{sanitize_file_string(self.env_name)}"
        self.filename_root = f"{model_name}_{sanitize_file_string(self.env_name)}_lr{self.learning_rate}_gamma{self.gamma}_eps{self.epsilon}_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"

        self.q_eval = DuelingDeepQNetwork( 
            self.n_actions, 
            self.input_dims, 
            self.filename_root + "_eval", 
            lr=self.learning_rate,
            checkpoint_dir=self.checkpoint_dir
        )

        self.q_target = DuelingDeepQNetwork( 
            self.n_actions, 
            self.input_dims, 
            self.filename_root + "_target", 
            lr=self.learning_rate,
            checkpoint_dir=self.checkpoint_dir
        )

        self.networks = [self.q_eval, self.q_target] # for savings model dicts

    
    def decrement_epsilon(self):
        self.epsilon = self.epsilon - self.epsilon_dec if self.epsilon > self.min_epsilon else self.min_epsilon
    

    def check_if_target_network_needs_replace(self):
        if self.learn_step_cnt % self.replace_limit == 0:
            self.q_target.load_state_dict(self.q_eval.state_dict())

    def learn(self):
        if self.memory.mem_cntr < self.batch_size: ## skip until we have batch_size in memory buffer (from main.py env loop)
            return
        self.q_eval.optimizer.zero_grad()

        self.check_if_target_network_needs_replace() ## check if we need to update target network weights to eval weights

        states, actions, rewards, next_states, terminals, idxs, is_weights = self.sample_from_buffer()

        with torch.no_grad(): ##avoid gradients passing through q_target
            best_actions = self.q_eval(next_states).argmax(dim=1).unsqueeze(1) # [B,1] tensor of best action indexes
            q_next_all_actions = self.q_target(next_states)
            q_next_state = q_next_all_actions.gather(1, best_actions).squeeze(1)
            q_next_state[terminals] = 0.0
            q_targ = rewards + self.gamma *  q_next_state

        q_pred_all = self.q_eval(states)
        q_pred = q_pred_all.gather(1, actions.unsqueeze(1)).squeeze(1)

        loss = self.q_eval.loss(q_pred, q_targ).to(self.q_eval.device)
        loss = (is_weights * loss).mean()
        loss.backward()
        self.q_eval.optimizer.step()

        #update transition priorities
        with torch.no_grad():
            td_errs = (q_targ - q_pred).detach().cpu().numpy()
            self.update_priorities(idxs, td_errs)

        self.learn_step_cnt += 1

        self.decrement_epsilon()