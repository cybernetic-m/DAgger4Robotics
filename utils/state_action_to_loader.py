from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
import torch
import numpy as np

def state_action_to_loader(data, batch_size):
    x_list = []  # observations (input for the model)
    y_list = []  # actions (target)
    for ep in data:
        obs = ep.observations[:-1]  # (50, 10) (removing the last one)
        acts = ep.actions           # (50, 2)
        
        # consistency check
        assert obs.shape[0] == acts.shape[0], f"Shape mismatch: {obs.shape[0]} vs {acts.shape[0]}"
        
        x_list.append(obs)
        y_list.append(acts)
    
    s = np.concatenate(x_list, axis=0)  # shape (len(data)*50, 10)
    a = np.concatenate(y_list, axis=0)  # shape (len(data)*50, 2)

    dataset = TensorDataset(torch.tensor(s, dtype=torch.float32), torch.tensor(a, dtype=torch.float32))
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)