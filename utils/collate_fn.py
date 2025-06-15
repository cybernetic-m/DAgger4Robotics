import torch
from torch.nn.utils.rnn import pad_sequence 

def collate_fn(batch):

    return {
        "id": torch.Tensor([x.id for x in batch]),
        "observations": pad_sequence([torch.as_tensor(x.observations) for x in batch], batch_first=True),
        "actions": pad_sequence([torch.as_tensor(x.actions) for x in batch], batch_first=True),
        "rewards": pad_sequence([torch.as_tensor(x.rewards) for x in batch], batch_first=True),
        "terminations": pad_sequence([torch.as_tensor(x.terminations) for x in batch], batch_first=True),
        "truncations": pad_sequence([torch.as_tensor(x.truncations) for x in batch], batch_first=True),
    }
