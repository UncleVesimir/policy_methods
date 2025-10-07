from collections import defaultdict
import torch
import torch.nn.functional as F
import numpy as np
from datetime import datetime
from utils import sanitize_file_string


from agents.common.PrioritizedReplayBufferAgent import PERBufferAgent
from networks.distributionalQNetwork import DistributionalQNetwork

class DistributionalDoubleDeepQAgent(PERBufferAgent):
    """
    Distributional Double DQN

    Implements a """

    def __init__(self, *, model_name="Distributional_Double_DQN", n_atoms=51, v_min=-10, v_max=10, **kwargs):
        super().__init__(**kwargs)

        self.checkpoint_dir = f"models/{model_name}/{sanitize_file_string(self.env_name)}"
        self.filename_root = f"{model_name}_{sanitize_file_string(self.env_name)}_lr{self.learning_rate}_gamma{self.gamma}_eps{self.epsilon}_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"

        self.theta_eval = DistributionalQNetwork( 
            self.n_actions, 
            self.input_dims, 
            self.filename_root + "_eval", 
            lr=self.learning_rate,
            n_atoms=n_atoms,
            checkpoint_dir=self.checkpoint_dir
        )

        self.theta_target = DistributionalQNetwork( 
            self.n_actions, 
            self.input_dims, 
            self.filename_root + "_target", 
            lr=self.learning_rate,
            n_atoms=n_atoms,
            checkpoint_dir=self.checkpoint_dir
        )

        self.networks = [self.theta_eval, self.theta_target] # for savings model dicts
    
        self.n_atoms = n_atoms
        self.v_min = v_min
        self.v_max = v_max
        self.delta_z = (v_max - v_min) / (n_atoms - 1)
        self.z = torch.linspace(v_min, v_max, n_atoms).to(self.theta_eval.device)
    
    
    def decrement_epsilon(self):
        self.epsilon = self.epsilon - self.epsilon_dec if self.epsilon > self.min_epsilon else self.min_epsilon
    

    def check_if_target_network_needs_replace(self):
        if self.learn_step_cnt % self.replace_limit == 0:
            self.theta_target.load_state_dict(self.theta_eval.state_dict())
    

    def learn(self):

        if self.memory.mem_cntr < self.batch_size: ## skip until we have batch_size in memory buffer (from main.py env loop)
            return
        self.theta_eval.optimizer.zero_grad()

        self.check_if_target_network_needs_replace() ## check if we need to update target network weights to eval weights

        states, actions, rewards, next_states, terminals, idxs, is_weights = self.sample_from_buffer()

        with torch.no_grad(): ##avoid gradients passing through q_target
            eval_logits = self.theta_eval(next_states)  # [B, A, N]
            eval_probs = F.softmax(eval_logits, dim=-1)
            q_eval = (eval_probs * self.z.view(1, 1, -1)).sum(dim=-1) 
            best_actions = q_eval.argmax(dim=-1)

            target_logits = self.theta_target(next_states)
            target_probs = F.softmax(target_logits, dim=-1) # [B, A, N]
            idx = best_actions.view(-1, 1, 1).expand(-1, 1, self.n_atoms) # [B, 1, N]
            p_next = target_probs.gather(1, index=idx).squeeze(1)  # [B, N]
            
            m = self.project_distribution(p_next, rewards, terminals)


            # ---- CE loss on taken action
        logits = self.theta_eval(states)     
        idx = actions.view(-1, 1, 1).expand(-1, 1, self.n_atoms)                                # [B, A, N]
        logits_a = logits.gather(1, idx).squeeze(1) # [B, N]
        log_probs = F.log_softmax(logits_a, dim=-1)
        loss_per_sample = -(m * log_probs).sum(dim=-1)
        loss = (loss_per_sample * is_weights).mean() if is_weights is not None else loss_per_sample.mean()

        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.theta_eval.parameters(), 10.0)
        self.theta_eval.optimizer.step()

        # PER priorities (optional)
        if idxs is not None:
            self.update_priorities(idxs, loss_per_sample.detach().abs().cpu().numpy())

        self.learn_step_cnt += 1
        self.decrement_epsilon()


    def project_distribution(self, p_next, rewards, terminals):
        B,N = p_next.size() # used to avoid writing self.batch_size and self.n_atoms everywhere

        terminals = terminals.float()
        m = torch.zeros(B, N).to(self.theta_eval.device)
        Tz = rewards.view(B, 1) + self.gamma * self.z.view(1, N) * (1 - terminals.view(B, 1))
        Tz = Tz.clamp(self.v_min, self.v_max)

        b = (Tz - self.v_min) / self.delta_z
        l = b.floor().clamp(0, N - 1).long()
        u = b.ceil().clamp(0, N - 1).long()

        w_l = (u.to(b.dtype) - b)
        w_u = (b - l.to(b.dtype))

        m.scatter_add_(dim=1, index=l, src=p_next * w_l)
        m.scatter_add_(dim=1, index=u, src=p_next * w_u)

        m = m + 1e-8 # avoid log(0)
        m = m / m.sum(dim=1, keepdim=True) # normalize

        return m
    
    def choose_action(self, observation):
        device = self.networks[0].device
        if np.random.random() > self.epsilon:
            state = torch.tensor(observation, dtype=torch.float).unsqueeze(0).to(device=device) # unsqueeze to add required batch dimension
            logits = self.networks[0](state)
            probs = F.softmax(logits, dim=-1)
            q = (probs * self.z.view(1, 1, -1).to(device)).sum(dim=-1)
            action = q.argmax(dim=-1).item()
        else:
            action = np.random.choice(self.n_actions)
        return action