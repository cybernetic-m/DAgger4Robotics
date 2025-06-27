import gymnasium as gym
from gymnasium.wrappers import RecordVideo #INSTALL pip install "gymnasium[other]"
from model.NetworkInterface import NetworkInterface
import gymnasium_robotics
import torch
import cv2
import json
import re
import numpy as np

kitchen_dataset_type = 'complete'
env_mode='kitchen'
net_type='simple'

batch_size=512
lr=1e-3
lr_str = format(lr, ".0e")  # '1e-03'
lr_str = re.sub(r"e-0*(\d+)", r"e-\1", lr_str)

student= True
n_iterations=20
rollouts_per_iteration=20
num_epochs=5
betaMode="exponential"
exponential_beta_k=0.3
idx=15

render = True  # Set to True to render the environment
video_saving= True # Set to True to save the videos
if render == False and video_saving==False:
    env = gym.make("FrankaKitchen-v1",tasks_to_complete=['microwave']) # Cration of the environment only for gathering data
else: #at least one True of render or video_saving
    env = gym.make("FrankaKitchen-v1",tasks_to_complete=['microwave'], render_mode="rgb_array", robot_noise_ratio=0.04) # Creation of the Reacher-v5 environment with rgb_array mode
    if video_saving == True:
        #Folder for saving video
        video_folder = "./videos"
        env = RecordVideo(env, video_folder=video_folder, episode_trigger=lambda e: True, name_prefix="student-kitchen-noise")

#Load the pi_star
net_wrapper = NetworkInterface(net_type=net_type,input_dim=20,output_dim=9)
net_wrapper.summary()
pi_star = net_wrapper.get_model()

if student:
    #Load the student
    pi_star.load_state_dict(torch.load(f"students_kitchen/{net_type}/kitchen_{kitchen_dataset_type}/batch_size_{batch_size}_lr_{lr_str}_iterations_{n_iterations}_rollouts_per_iteration_{rollouts_per_iteration}_num_epochs_{num_epochs}_betaMode_{betaMode}_exponential_beta_k_{exponential_beta_k}/student_policy_{idx}.pt",map_location=torch.device('cpu')))
else:
    #Load the best expert weights
    #pi_star.load_state_dict(torch.load(f"expert_policy_simple.pt",map_location=torch.device('cpu')))
    pi_star.load_state_dict(torch.load(f"experts_kitchen/{net_type}/kitchen_{kitchen_dataset_type}/batch_size_{batch_size}_lr_{lr_str}/expert_policy.pt",map_location=torch.device('cpu')))
    #pi_star.load_state_dict(torch.load(f"experts_kitchen/{net_type}/kitchen_{kitchen_dataset_type}/expert_policy_simple.pt",map_location=torch.device('cpu')))
pi_star.eval()


n_episodes = 1
mean_reward_for_episode = {}
for ep in range(n_episodes):
    observation, _ = env.reset(seed=ep)
    observation = torch.tensor(observation['observation'], dtype=torch.float32)
    # selected_obs = torch.cat([
    # observation[0:18],          # joint angles, gripper translations, joint velocities
    # observation[31].unsqueeze(0),  # microwave door angle
    # observation[52].unsqueeze(0)   # microwave door angular velocity
    # ])

    selected_obs = torch.cat([
        observation[0:9],
        torch.tensor([0.0 for _ in range(11)]),
        #observation[9:16],
        #torch.tensor([0.0 for _ in range(2)]),
        #observation[31].unsqueeze(0),  # microwave door angle
        #observation[52].unsqueeze(0)
        ])
    
    print(selected_obs)
    done = False
    total_reward = 0.0
    step = 0

    while not done:
        action = pi_star(selected_obs)

        # Step the environment with the sampled action
        observation, reward, terminated, truncated, info = env.step(action.detach().cpu().numpy())
        observation = torch.tensor(observation['observation'], dtype=torch.float32)
        if step > 27:
            selected_obs = torch.cat([
            observation[0:18],          # joint angles, gripper translations, joint velocities
            observation[31].unsqueeze(0),  # microwave door angle
            observation[52].unsqueeze(0)   # microwave door angular velocity
            ])
        else:
            selected_obs = torch.cat([
                observation[0:9],
                torch.tensor([0.0 for _ in range(11)]),
            ])
        total_reward += reward
        step +=1
        # Render the environment
        if render == True:
            frame=env.render()
            cv2.imshow("Reacher", frame[:, :, ::-1])  # Convert RGB → BGR
            if cv2.waitKey(30) & 0xFF == ord('q'):
                break

        # Check if the episode is done
        done = terminated or truncated
    mean_reward_episode=total_reward/step
    mean_reward_for_episode[f"Episode {ep}"]=mean_reward_episode
total_mean = sum(mean_reward_for_episode.values()) / len(mean_reward_for_episode)
print(total_mean)
env.close()
cv2.destroyAllWindows()
mean_reward_for_episode["mean_of_means"] = total_mean
#Saving mean rewards in a json file
if student == True:
    with open(f"students_kitchen/{net_type}/kitchen_{kitchen_dataset_type}/batch_size_{batch_size}_lr_{lr_str}_iterations_{n_iterations}_rollouts_per_iteration_{rollouts_per_iteration}_num_epochs_{num_epochs}_betaMode_{betaMode}_exponential_beta_k_{exponential_beta_k}/mean_rewards.json", "w") as f:
        json.dump(mean_reward_for_episode, f, indent=4)
else:
    with open(f"experts_kitchen/{net_type}/kitchen_{kitchen_dataset_type}/batch_size_{batch_size}_lr_{lr_str}/mean_rewards.json", "w") as f:
        json.dump(mean_reward_for_episode, f, indent=4)