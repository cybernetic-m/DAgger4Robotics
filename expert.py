import gymnasium as gym
import minari # needed for dataset

render = False  # Set to True to render the environment

# Creation of the Reacher-v5 environment with human rendering mode
if render:
    env = gym.make("Reacher-v5", render_mode="human",max_episode_steps=200)
else:
    env = gym.make("Reacher-v5",max_episode_steps=200)

# Load the dataset for the Reacher environment
dataset = minari.load_dataset('mujoco/reacher/medium-v0')
dataset.set_seed(42)  # Set a seed for reproducibility
print("Dataset loaded successfully!")

# To sample episodes from the dataset
#episodes = dataset.sample_episodes(6)
#ids = [episode.id for episode in episodes]
#print(ids)

# get episodes with mean reward greater than 2
#expert_dataset = dataset.filter_episodes(lambda episode: episode.rewards.mean() > -0.1)
#print(f'TOTAL EPISODES FILTER DATASET: {filter_dataset.total_episodes}')

# Split the dataset into training, evaluation and test sets with percentage sizes 0.7, 0.2, 0.1
# The original dataset is of size 10000, we split it into 7000 for training and 2000 for evaluation and 1000 for testing
dataset_split = minari.split_dataset(dataset, sizes=[7000, 2000, 1000], seed=42) 
training_dataset = dataset_split[0]
evaluation_dataset = dataset_split[1]
test_dataset = dataset_split[2]
print(f"Training episodes: {len(training_dataset)}")
print(f"Evaluation episodes: {len(evaluation_dataset)}")
print(f"Test episodes: {len(test_dataset)}")

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