from collections import defaultdict
import torch
import numpy as np
from datetime import datetime
from utils import sanitize_file_string

from networks.deepQNetwork import DeepQNetwork
from agents.common.ReplayBufferAgent import ReplayBufferAgent


class DoubleDeepQAgent(ReplayBufferAgent):
    """
    Double DQN Agent, based on https://arxiv.org/pdf/1509.06461
    The core idea in Double DQ learning is to decouple the computation of Q values
    and next_action by the Eval Network, which has been shown to
    accumulate errors (noise, bias, non-stationarity etc.) 
    and overestimation of Q values.

    Here, the Eval network is used to select the greedy action but
    now the the Taret Network is used to evaluate the action
    """

    def __init__(self, model_name="Double_DQN", *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.checkpoint_dir = f"models/{model_name}/{sanitize_file_string(self.env_name)}"
        self.filename_root = f"{model_name}_{sanitize_file_string(self.env_name)}_lr{self.learning_rate}_gamma{self.gamma}_eps{self.epsilon}_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"

        self.q_eval = DeepQNetwork( 
            self.n_actions, 
            self.input_dims, 
            self.filename_root + "_eval", 
            lr=self.learning_rate,
            checkpoint_dir=self.checkpoint_dir
        )

        self.q_target = DeepQNetwork( 
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

        states, actions, rewards, next_states, terminals = self.sample_from_buffer()

        best_actions = self.q_eval(next_states).argmax(dim=1).unsqueeze(1) # [B,1] tensor of best action indexes
        q_next_all_actions = self.q_target(next_states)
        q_next_state = q_next_all_actions.gather(1, best_actions).squeeze(1)
        
        q_next_state[terminals] = 0.0

        q_targ = rewards + self.gamma *  q_next_state
        q_pred_all = self.q_eval(states)
        q_pred = q_pred_all.gather(1, actions.unsqueeze(1)).squeeze(1)

        loss = self.q_eval.loss(q_pred, q_targ).to(self.q_eval.device)
        loss.backward()
        self.q_eval.optimizer.step()

        self.learn_step_cnt += 1

        self.decrement_epsilon()