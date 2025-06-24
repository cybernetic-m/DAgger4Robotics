import numpy as np

def preprocess_dataset(kitchen_minari_dataset):

  # This function takes the 'D4RL/kitchen/complete-v2' and it will cut trajectories extracting only the values until the
  # microwave task is reached. This because the microwave task is the first, and our objective is to do only the microwave opening task.
  # Args: - kitchen_minari_dataset: it should be a dataset of Minari (use 'dataset = minari.load_dataset('D4RL/kitchen/complete-v2', download = True)')
  #
  # Output: - microwave_dataset: it is a dict of this type (where each list has a len that is a cutted version of the original dataset)
  # {'observations': {'1': [[obs1, ..., obsN], [obs1, ..., obsN], ...], ...}, 'actions': {'1': [[act1, ..., actN], [act1, ..., actN], ...]} }
  
  # Tolerance is a parameter used to check the differences between values in achieved_goal_list
  tolerance = 1e-20

  # Returning dictionary
  microwave_dataset = {
      'observations': {},
      'actions': {},
  }

  # Sample all the episodes from the dataset
  episodes = kitchen_minari_dataset.sample_episodes(n_episodes=len(kitchen_minari_dataset))

  # Loop over all the episodes
  for ep in episodes:
    
    # Create two lists of observation of episode i and action of episode i, in which we append each step action/observation
    # Ex. obs = [[obs1, ..., obsN], [obs1, ..., obsN],...] where each list inside is 59 dimensional in case of Kitchen Env, 1 for each step
    obs = []
    acts = []

    # Takes the achieved_goal list for the microwave task that is used to understand when (step) the task is completed
    # The achieved goal list is formed by N values (where N is the number of step for that episode)
    # Ex. [5, 4, 3, 2, 1, 1, 1, 1, 1] (Goal is 1)
    # Then we will compare pairs of values seeing when the difference is below a tolerance value 
    # (no other changes means that the microwave is opened at that step)
    achieved_goal_list = np.array(ep.observations['achieved_goal']['microwave']).flatten()
    observation_list = ep.observations['observation']  # List of lists [[obs1, ..., obsN], [obs1, ..., obsN],...]
    actions_list = ep.actions

    # Loop to check the stop_index point
    for i in range(len(achieved_goal_list) - 1):
      # Compute the difference between the next value and the actual value of the list
      diff = np.abs(achieved_goal_list[i+1] - achieved_goal_list[i])

      # If the difference is less then the very small value (tolerance) then we need to stop the list
      if diff < tolerance:
        break
      # Otherwise we append the obs_i and act_i to the obs and acts lists
      # The appending procedure will be stopped if the previous if will be verified
      obs_list = observation_list[i]
      # We take a subset of observations, in particular:
      # 0-6 joint angles
      # 7-8 gripper joint translation value
      # 9-15 joint angular velocities
      # 16-17 gripper joint linear velocity 
      # 31  Rotation of the joint in the microwave door (angle) 
      # 52 Angular velocity of the microwave door joint
      sub_obs_list = np.concatenate([obs_list[0:18], [obs_list[31]], [obs_list[52]]])
      obs.append(sub_obs_list)
      acts.append(actions_list[i])
      
    microwave_dataset['observations'][ep.id.item()] = obs
    microwave_dataset['actions'][ep.id.item()] = acts
    
  return microwave_dataset