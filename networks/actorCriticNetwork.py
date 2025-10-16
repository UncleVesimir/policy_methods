import os
import re
from datetime import datetime
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class ActorCriticNetwork(nn.Module):
    def __init__(self, n_actions=None, input_dims=None, file_name=None, lr=2.5e-4, checkpoint_dir="models/Unknown"):
        super().__init__()
        if n_actions is None or input_dims is None or file_name is None:
            raise ValueError("n_actions, input_dims, and file_name must be provided.")
        
        self.mdl_checkpoint_dir = checkpoint_dir
        self.mdl_checkpoin_filename = os.path.join(self.mdl_checkpoint_dir, file_name)

        #Main Network setup
        self.conv1 = nn.Conv2d(input_dims[0], 32, 8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, 3, stride=2)
        self.conv3 = nn.Conv2d(64, 64, 3, stride=1)

        fc_input_dims = self.calc_conv_out_dims(input_dims)
        self.fc1 = nn.Linear(fc_input_dims, 512)
        self.actor = nn.Linear(512, n_actions) 
        self.critic = nn.Linear(512, 1)

        ## Optimizer, loss function, device setup
        # self.optimizer = optim.RMSProp(self.parameters(), lr=lr, alpha=0.99, eps=1e-5)
        self.optimizer = optim.AdamW(self.parameters(), lr=lr, weight_decay=0)
        # self.loss = nn.MSELoss()
        self.device = torch.device("mps" if torch.backends.mps.is_available() else 'cuda:0' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        self.to(self.device)

    def calc_conv_out_dims(self, input_dims):
        """ utility to calculate the output dimensions of conv layers """
        with torch.no_grad():
            state = torch.zeros(1, *input_dims)
            dims = self.conv1(state)
            dims = self.conv2(dims)
            dims = self.conv3(dims)
            return int(np.prod(dims.size()))
    

    def forward(self, state) -> tuple[torch.Tensor, torch.Tensor]:
        """ forward pass """
        
        conv1 = F.relu(self.conv1(state))
        conv2 = F.relu(self.conv2(conv1))
        conv3 = F.relu(self.conv3(conv2)) # conv3 output is shape batch_size x filters x H x W
        flat = conv3.view(conv3.size()[0], -1)  # flatten 
        fc1 = F.relu(self.fc1(flat))
        
        action_logits = self.actor(fc1)
        critic_value = self.critic(fc1)

        return action_logits, critic_value
    
    def save_checkpoint(self):
        print("...saving checkpoint...")
        os.makedirs(os.path.dirname(self.mdl_checkpoin_filename), exist_ok=True)
        torch.save(self.state_dict(), self.mdl_checkpoin_filename)

    def load_checkpoint(self):
        print("...loading checkpoint...")
        checkpoint_dir = os.path.abspath(self.mdl_checkpoint_dir)

        if not os.path.isdir(checkpoint_dir):
            raise ValueError(f"Checkpoint directory not found: {checkpoint_dir}")


        target_stub = None
        base_filename = os.path.basename(self.mdl_checkpoin_filename)

        for stub in ("_eval", "_target"):
            if base_filename.endswith(stub):
                target_stub = stub
                break

        if target_stub is None:
            raise ValueError(
                f"Unsupported checkpoint filename stub in {base_filename}. Expected '_eval' or '_target'."
            )

        all_entries = [
            entry
            for entry in os.listdir(checkpoint_dir)
            if os.path.isfile(os.path.join(checkpoint_dir, entry))
        ]

        matching_entries = [entry for entry in all_entries if entry.endswith(target_stub)]

        if not matching_entries:
            print(f"No checkpoints found in {checkpoint_dir} matching '{target_stub}'.")
            return

        def extract_timestamp(entry_name: str) -> Optional[datetime]:
            matches = re.findall(r"_(\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2})", entry_name)
            if not matches:
                return None
            try:
                return datetime.strptime(matches[-1], "%Y-%m-%d_%H-%M-%S")
            except ValueError:
                return None

        entries_with_metadata = []
        for entry in matching_entries:
            timestamp = extract_timestamp(entry)
            base_without_stub = entry[: -len(target_stub)] if entry.endswith(target_stub) else entry
            entries_with_metadata.append((entry, base_without_stub, timestamp))

        entries_with_metadata.sort(
            key=lambda item: item[2] or datetime.min,
            reverse=True,
        )

        print("Available checkpoints:")
        for idx, (_, base_without_stub, timestamp) in enumerate(entries_with_metadata, start=1):
            timestamp_display = timestamp.strftime("%Y-%m-%d %H:%M:%S") if timestamp else "Unknown timestamp"
            print(f"  {idx}. {base_without_stub}{target_stub} [{timestamp_display}]")

        selection = None
        while selection is None:
            raw_input = input(f"Select checkpoint to load (1-{len(entries_with_metadata)}): ")
            try:
                choice = int(raw_input)
                if 1 <= choice <= len(entries_with_metadata):
                    selection = entries_with_metadata[choice - 1]
                else:
                    print("Selection out of range. Please try again.")
            except ValueError:
                print("Invalid input. Please enter a number.")

        _, base_without_stub, _ = selection
        chosen_filename = f"{base_without_stub}{target_stub}"
        chosen_path = os.path.join(checkpoint_dir, chosen_filename)

        print(f"Loading checkpoint: {chosen_path}")
        checkpoint_data = torch.load(chosen_path, map_location=self.device)
        self.load_state_dict(checkpoint_data)
        self.mdl_checkpoin_filename = chosen_path
    
