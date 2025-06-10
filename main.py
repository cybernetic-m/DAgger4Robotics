import gymnasium as gym

env = gym.make("Reacher-v5", render_mode="human")

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
    env.render()

    # Check if the episode is done
    done = terminated or truncated

env.close()