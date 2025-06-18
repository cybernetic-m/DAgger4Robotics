import gymnasium as gym
from gymnasium.wrappers import RecordVideo #INSTALL pip install "gymnasium[other]"
from model.ExpertPolicyNet import ExpertPolicyNet
import torch
import cv2

render = True  # Set to True to render the environment
video_saving=True

# Creation of the Reacher-v5 environment with human rendering mode
if render:
    env = gym.make("Reacher-v5", render_mode="rgb_array")#,max_episode_steps=200)
else:
    env = gym.make("Reacher-v5",max_episode_steps=200)

#Folder for saving video
video_folder = "./videos"

# Wrapper for registering videos
if video_saving:
    env = RecordVideo(env, video_folder=video_folder, episode_trigger=lambda e: True)

#Load the model
model = ExpertPolicyNet(10,2)

#LOad the expert weights
model.load_state_dict(torch.load('expert_policy.pt',map_location=torch.device('cpu')))
model.eval()


# Reset the environment to start a new episode
# Return the initial observation and info dictionary (if available)

n_episodes = 5
reward_for_episode = []
# observation, info = env.reset()
# observation = torch.tensor(observation, dtype=torch.float32)

for ep in range(n_episodes):
    observation, _ = env.reset()
    observation = torch.tensor(observation, dtype=torch.float32)
    done = False
    total_reward = 0.0

    while not done:
    
        # Sample a random action from the action space
        action = model(observation)

        # Step the environment with the sampled action
        observation, reward, terminated, truncated, info = env.step(action.detach().cpu().numpy())
        observation = torch.tensor(observation, dtype=torch.float32)
        total_reward += reward
        # Render the environment
        if render == True:
            frame=env.render()
            cv2.imshow("Reacher", frame[:, :, ::-1])  # Convert RGB → BGR
            if cv2.waitKey(30) & 0xFF == ord('q'):
                break

        # Check if the episode is done
        done = terminated or truncated

    reward_for_episode.append(total_reward)

print(reward_for_episode)
env.close()
cv2.destroyAllWindows()