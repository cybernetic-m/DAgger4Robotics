import gymnasium as gym
from gymnasium.wrappers import RecordVideo #INSTALL pip install "gymnasium[other]"
from model.NetworkInterface import NetworkInterface
import torch
import cv2
import json
import time

render = False  # Set to True to render the environment
video_saving=True # Set to True to save the videos
student= True
if render == False and video_saving==False:
    env = gym.make("Reacher-v5") # Cration of the environment only for gathering data
else: #at least one True of render or video_saving
    env = gym.make("Reacher-v5", render_mode="rgb_array") # Creation of the Reacher-v5 environment with rgb_array mode
    if video_saving == True:
        #Folder for saving video
        video_folder = "./videos"
        env = RecordVideo(env, video_folder=video_folder, episode_trigger=lambda e: True, name_prefix="ep_deep_student")

#Load the pi_star
net_wrapper = NetworkInterface(net_type='simple',input_dim=10,output_dim=2)
net_wrapper.summary()
pi_star = net_wrapper.get_model()

#Load the best expert/student weights
if student == True:
    pi_star.load_state_dict(torch.load('students_reacher/trained_on_deep_batch_32/student_policy_inverse.pt',map_location=torch.device('cpu')))
    #pi_star.load_state_dict(torch.load('students_reacher/student_policy_batch_32/student_policy_inverse.pt',map_location=torch.device('cpu')))
else:
    pi_star.load_state_dict(torch.load('experts_reacher/deep_expert_policy/expert_policy_512.pt',map_location=torch.device('cpu')))
pi_star.eval()


n_episodes = 100
mean_reward_for_episode = {}
start = time.time()
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
    mean_reward_for_episode[f"Episode {ep}"]=mean_reward_episode
end = time.time()
total= (end - start)/50
print(f"Tempo impiegato: {total:.4f} secondi")
total_mean = sum(mean_reward_for_episode.values()) / len(mean_reward_for_episode)
print(total_mean)
env.close()
cv2.destroyAllWindows()
mean_reward_for_episode["mean_of_means"] = total_mean
#Saving mean rewards in a json file
with open("mean_rewards.json", "w") as f:
    json.dump(mean_reward_for_episode, f, indent=4)