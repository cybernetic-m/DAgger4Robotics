import gymnasium as gym
import minari # needed for dataset

render = True  # Set to True to render the environment

# Creation of the Reacher-v5 environment with human rendering mode
if render:
    env = gym.make("Reacher-v5", render_mode="human",max_episode_steps=200)
else:
    env = gym.make("Reacher-v5",max_episode_steps=200)

dataset = minari.load_dataset('mujoco/reacher/medium-v0')
print("Observation space:", dataset.observation_space)
print("Action space:", dataset.action_space)
print("Total episodes:", dataset.total_episodes)
print("Total steps:", dataset.total_steps)

# Reset the environment to start a new episode
# Return the initial observation and info dictionary (if available)
observation, info = env.reset()

done = False

while not done:
   
    # Sample a random action from the action space
    action = env.action_space.sample()

    # Step the environment with the sampled action
    observation, reward, terminated, truncated, info = env.step(action)

    # Render the environment
    if render == True:
        env.render()

    # Check if the episode is done
    done = terminated or truncated

env.close()