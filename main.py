import gymnasium as gym
import minari # needed for dataset

# Creation of the Reacher-v5 environment with human rendering mode
env = gym.make("Reacher-v5")

dataset = minari.load_dataset('mujoco/reacher/medium-v0')
print("Observation space:", dataset.observation_space)
print("Action space:", dataset.action_space)
print("Total episodes:", dataset.total_episodes)
print("Total steps:", dataset.total_steps)

# Reset the environment to start a new episode
# Return the initial observation and info dictionary (if available)
observation, info = env.reset()

done = False

render = False  # Set to True to render the environment

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