import json

def count_non_zero_episodes(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    count = 0
    for key, value in data.items():
        if key.startswith("Episode") and value != 0.0:
            count += 1
    return count

# Inserisci i percorsi dei due file JSON
file1 = 'experts_kitchen/deep/kitchen_complete/batch_size_64_lr_1e-3/mean_rewards_big_noise.json'
file2 = 'students_kitchen/simple/kitchen_complete/batch_size_512_lr_1e-3_iterations_20_rollouts_per_iteration_20_num_epochs_5_betaMode_exponential_exponential_beta_k_0.3/mean_rewards_big_noise.json'

count1 = count_non_zero_episodes(file1)
count2 = count_non_zero_episodes(file2)

print(f"Episodi diversi da zero in {file1}: {count1}")
print(f"Episodi diversi da zero in {file2}: {count2}")