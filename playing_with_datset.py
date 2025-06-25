import minari
import gymnasium as gym
from PIL import Image
import os
import numpy as np
from utils.state_action_to_loader import state_action_to_loader
from utils.preprocess_dataset import preprocess_dataset

dictio={'tasks_to_complete': ['microwave']}
dataset = minari.load_dataset('D4RL/kitchen/complete-v2', download=True)

print(dataset.total_episodes)
microwave_dataset = preprocess_dataset(dataset)
loader=state_action_to_loader(microwave_dataset,256)

for obs_batch, act_batch in loader:
    print("Obs batch shape:", obs_batch.shape)
    print("Action batch shape:", act_batch.shape)
    break


#env = dataset.recover_environment()
#print(env.spec)
#env = gym.make(env.spec, render_mode = 'rgb_array')

p1=dataset[14].observations['achieved_goal']['microwave'].flatten()
p2=dataset[14].observations["observation"][:,31].flatten()

#print(dataset[4].observations['desired_goal']['microwave'])
x=dataset[4]
x=[2,3,4]
print(hasattr(x, '__class__') and x.__class__.__module__ != 'builtins')
print()
print(dataset[4])
"""
i=0
for ep in dataset:
    if ep.observations["achieved_goal"]["microwave"][-1] > -0.1: #236 to be eliminated and 385 useful
        print(ep.id)
        i=1+i
print(f'totali episodi: {i}')
"""
"""
for step in range(len(dataset[3].observations["observation"])):
    print(dataset[4].observations["observation"][step][31])
"""
#print(dataset[3].observations["observation"][200][31])