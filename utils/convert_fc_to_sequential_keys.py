import torch

# Converts old state_dict keys (e.g., "fc1.weight") to the new sequential format
# (e.g., "net.0.weight"). Needed due to a refactor where layer naming changed,
# even though the architecture remained the same. This enable us to use the old
# trained simple Networks of experts_reacher

def convert_fc_to_sequential_keys(dict_path, device='cpu'):
    old_state_dict = torch.load(dict_path, map_location=torch.device(device))

    if "net.0.weight" in old_state_dict: #If the state_dict is already in the correct format do not change it
        return old_state_dict
    
    print("Necessary conversion to the new layer naming convention.")
    # Maps the old names to the new ones
    key_mapping = {
        "fc1.weight": "net.0.weight",
        "fc1.bias": "net.0.bias",
        "fc2.weight": "net.2.weight",
        "fc2.bias": "net.2.bias",
        "fc3.weight": "net.4.weight",
        "fc3.bias": "net.4.bias",
    }

    # Construct the correct state_dict
    new_state_dict = {}
    for old_key, value in old_state_dict.items():
        if old_key in key_mapping:
            new_key = key_mapping[old_key]
            new_state_dict[new_key] = value

    return new_state_dict