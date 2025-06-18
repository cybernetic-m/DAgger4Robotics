import gymnasium as gym
from gymnasium.wrappers import RecordVideo #INSTALL pip install "gymnasium[other]"
from model.ExpertPolicyNet import ExpertPolicyNet
import torch
import cv2
import json

render = True  # Set to True to render the environment
video_saving=True # Set to True to save the videos
if render == False and video_saving==False:
    env = gym.make("Reacher-v5") # Cration of the environment only for gathering data
else: #at least one True of render or video_saving
    env = gym.make("Reacher-v5", render_mode="rgb_array") # Creation of the Reacher-v5 environment with rgb_array mode
    if video_saving == True:
        #Folder for saving video
        video_folder = "./videos"
        env = RecordVideo(env, video_folder=video_folder, episode_trigger=lambda e: True)

#Load the pi_star
pi_star = ExpertPolicyNet(10,2)

#Load the expert weights
pi_star.load_state_dict(torch.load('super_expert_policy.pt',map_location=torch.device('cpu')))
pi_star.eval()


n_episodes = 5
mean_reward_for_episode = {}

for ep in range(n_episodes):
    observation, _ = env.reset(seed=ep)
    observation = torch.tensor(observation, dtype=torch.float32)
    done = False
    total_reward = 0.0
    step = 0

    while not done:
        action = pi_star(observation)

        # Step the environment with the sampled action
        observation, reward, terminated, truncated, info = env.step(action.detach().cpu().numpy())
        observation = torch.tensor(observation, dtype=torch.float32)
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
    mean_reward_for_episode[str(ep)]=mean_reward_episode

print(mean_reward_for_episode)
env.close()
cv2.destroyAllWindows()
#Saving mean rewards in a json file
with open("mean_rewards.json", "w") as f:
    json.dump(mean_reward_for_episode, f, indent=4)