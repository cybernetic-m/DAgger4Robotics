import gymnasium as gym
from gymnasium.wrappers import RecordVideo
import sys
import os

# Get the absolute paths of the directories containing the utils functions 
utils_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../utils'))

# Get the absolute paths of the directories containing the model
model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../model'))

# Add these directories to sys.path
sys.path.append(utils_path)
sys.path.append(model_path)

# Import section
from NetworkInterface import NetworkInterface
from convert_fc_to_sequential_keys import convert_fc_to_sequential_keys
from is_running_in_colab import is_running_in_colab
import gymnasium_robotics
import torch
import cv2
import json

# Import to be used in colab that enable to rendering 
if is_running_in_colab():
    #Display on colab
    from pyvirtualdisplay import Display
    from IPython.display import Video, display, clear_output
    display = Display(visible=0, size=(200, 200))

    display.start()
    from google.colab.patches import cv2_imshow

class Simulator:
    def __init__(self,env_mode, net_type, path_to_model, n_episodes=1, render=False, video_saving=True, robot_noise=0.01, device='cpu', framerate_per_episode=5):
        self.env_mode = env_mode
        self.render = render
        self.video_saving = video_saving
        self.robot_noise = robot_noise #Only used in Franka-Kitchen
        self.n_episodes = n_episodes
        self.net_type = net_type
        self.path_to_model = path_to_model
        self.device=device
        self.framerate_per_episode=framerate_per_episode #Only used in colab
    
        self.env = self._make_env()
        self.pi_star = self._load_policy()

    def _make_env(self):
        if self.env_mode == 'kitchen':
            kwargs = {
                "tasks_to_complete": ['microwave'],
                "robot_noise_ratio": self.robot_noise,
            }
            self.env_name="FrankaKitchen-v1"
            self.input_dim=20 #Input of the Network
            self.output_dim=9 #Output of the Network
        else:
            self.env_name="Reacher-v5"
            kwargs = {}
            self.input_dim=10 #Input of the Network
            self.output_dim=2 #Output of the Network

        if self.render or self.video_saving:
            kwargs["render_mode"] = "rgb_array"

        env = gym.make(self.env_name, **kwargs)

        if self.video_saving:
            video_folder = "./new_videos"
            os.makedirs(video_folder, exist_ok=True)
            env = RecordVideo(env, video_folder=video_folder, episode_trigger=lambda e: True, name_prefix=f"{self.env_mode}")
        return env
    
    def _load_policy(self):
        net_wrapper = NetworkInterface(net_type=self.net_type, input_dim=self.input_dim, output_dim=self.output_dim)
        net_wrapper.summary()
        pi_star = net_wrapper.get_model().to(self.device)

        if "experts_reacher/simple" in self.path_to_model:
            state_dict = convert_fc_to_sequential_keys(self.path_to_model,self.device) #Manage the old naming of the simple Network
        else:
            state_dict = torch.load(self.path_to_model, map_location=torch.device(self.device))

        pi_star.load_state_dict(state_dict)
        pi_star.eval()
        return pi_star
    
    def _process_observation(self, obs):
        if self.env_mode == 'kitchen':
            obs = torch.tensor(obs['observation'], dtype=torch.float32, device=self.device)
            selected = torch.cat([      # we do not need all the observation, but only:
                obs[0:18],              # proprioception
                obs[31].unsqueeze(0),   # microwave angle
                obs[52].unsqueeze(0)    # microwave angular velocity
            ])
            return selected
        else:
            return torch.tensor(obs, dtype=torch.float32, device=self.device) #For reacher all the Obs Space
    
    def run(self):
        mean_reward_for_episode = {}
        for ep in range(self.n_episodes):
            obs, _ = self.env.reset(seed=ep)
            obs = self._process_observation(obs)
            done = False
            total_reward = 0.0
            step = 0
            if self.render == True and is_running_in_colab():
                clear_output(wait=True)

            while not done:
                action = self.pi_star(obs)
                obs, reward, terminated, truncated, _ = self.env.step(action.detach().cpu().numpy())
                obs = self._process_observation(obs)

                total_reward += reward
                step += 1

                if self.render: #Managing rendering
                    if is_running_in_colab():
                        if step % self.framerate_per_episode == 0:
                            frame=self.env.render()
                            cv2_imshow(frame[:, :, ::-1])
                    else:
                        frame = self.env.render()
                        cv2.imshow("Env", frame[:, :, ::-1])
                        if cv2.waitKey(30) & 0xFF == ord('q'):
                            break

                done = terminated or truncated

            mean_reward_for_episode[f"Episode {ep}"] = total_reward / step

        total_mean = sum(mean_reward_for_episode.values()) / len(mean_reward_for_episode)
        mean_reward_for_episode["mean_of_means"] = total_mean

        print(f"Mean of means: {total_mean}")
        self.env.close()
        if not is_running_in_colab():
            cv2.destroyAllWindows()

        with open(f"mean_rewards_{self.env_mode}.json", "w") as f:
            json.dump(mean_reward_for_episode, f, indent=4)