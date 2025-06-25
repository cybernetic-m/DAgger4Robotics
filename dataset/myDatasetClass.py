from torch.utils.data import Dataset
import torch
import numpy as np

class myDatasetClass(Dataset):
    def __init__(self, dataset_dict, env_mode):

        self.obs_list = []  # observations (input for the model)
        self.act_list = []  # actions (target)
        if env_mode == 'reacher':
            for ep in dataset_dict:
                obs = ep.observations[:-1]  # (50, 10) (removing the last one)
                acts = ep.actions           # (50, 2)
                
                # consistency check
                assert obs.shape[0] == acts.shape[0], f"Shape mismatch: {obs.shape[0]} vs {acts.shape[0]}"
                
                self.obs_list.extend(obs)
                self.act_list.extend(acts)
            
            s = np.array(self.obs_list)  # shape (len(data)*50, 10)
            a = np.array(self.act_list)  # shape (len(data)*50, 2)

            self.obsTensor = torch.tensor(s, dtype=torch.float32)
            self.actTensor = torch.tensor(a, dtype=torch.float32)

        elif env_mode == 'kitchen':
            for ep_id in dataset_dict['observations']:
                obs = dataset_dict['observations'][ep_id]
                acts = dataset_dict['actions'][ep_id]

                self.obs_list.extend(obs)
                self.act_list.extend(acts)
                
            s = np.array(self.obs_list)
            a = np.array(self.act_list)

            self.obsTensor = torch.tensor(s, dtype=torch.float32)
            self.actTensor = torch.tensor(a, dtype=torch.float32)

        else:
            raise ValueError(f"Unsupported environment mode: {env_mode}. Choose 'reacher' or 'kitchen'.")
        
    def __len__(self):
        return len(self.obs_list)
    
    def __getitem__(self, idx):
        return self.obsTensor[idx], self.actTensor[idx]