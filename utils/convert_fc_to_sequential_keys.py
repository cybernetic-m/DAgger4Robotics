import torch
def convert_fc_to_sequential_keys(dict_path, device='cpu'):
    old_state_dict = torch.load(dict_path, map_location=torch.device(device))

    if "net.0.weight" in old_state_dict:
        print("No conversion needed")
        return old_state_dict

    # Mappa vecchi nomi → nuovi nomi
    key_mapping = {
        "fc1.weight": "net.0.weight",
        "fc1.bias": "net.0.bias",
        "fc2.weight": "net.2.weight",
        "fc2.bias": "net.2.bias",
        "fc3.weight": "net.4.weight",
        "fc3.bias": "net.4.bias",
    }

    # Costruisci un nuovo dizionario con i nomi rinominati
    new_state_dict = {}
    for old_key, value in old_state_dict.items():
        if old_key in key_mapping:
            new_key = key_mapping[old_key]
            new_state_dict[new_key] = value

    return new_state_dict