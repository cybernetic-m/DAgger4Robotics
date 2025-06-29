from utils.Simulator import Simulator
sim = Simulator(
        env_mode='kitchen',
        net_type='deep',
        path_to_model='experts_kitchen/deep/kitchen_complete/batch_size_64_lr_1e-3/expert_policy.pt',
        n_episodes=5,
        render=True,
        video_saving=False,
        robot_noise=0.01,
        device='cpu'
    )
sim.run()

"""
EXPERT KITCHEN DEEP
env_mode='kitchen',
net_type='deep',
path_to_model='experts_kitchen/deep/kitchen_complete/batch_size_64_lr_1e-3/expert_policy.pt',

EXPERT KITCHEN SIMPLE
env_mode='kitchen',
net_type='simple',
path_to_model='experts_kitchen/simple/kitchen_complete/batch_size_64_lr_1e-3/expert_policy.pt'

---STUDENT KITCHEN SIMPLE
env_mode='kitchen',
net_type='simple',
path_to_model='students_kitchen/simple/kitchen_complete/batch_size_512_lr_1e-3_iterations_20_rollouts_per_iteration_20_num_epochs_5_betaMode_exponential_exponential_beta_k_0.3/student_policy_15.pt'


----------------REACHER--------------------

EXPERT REACHER DEEP
env_mode='reacher',
net_type='deep',
path_to_model='experts_reacher/deep/reacher_expert_not_filtered/batch_size_128_lr_1e-4/expert_policy.pt'

EXPERT REACHER DEEP (FILTERED)
env_mode='reacher',  # o 'reacher'
net_type='deep',
path_to_model='experts_reacher/deep/reacher_expert_filtered/batch_size_128_lr_1e-4/expert_policy.pt'

MEDIUM REACHER DEEP
env_mode='reacher',  # o 'reacher'
net_type='deep',
path_to_model='experts_reacher/deep/reacher_medium_not_filtered/batch_size_128_lr_1e-4/expert_policy.pt'
--------SIMPLE-------
EXPERT REACHER SIMPLE (OLD)
env_mode='reacher',  # o 'reacher'
net_type='simple',
path_to_model='experts_reacher/simple/reacher_expert_not_filtered/batch_size_512_lr_1e-3/expert_policy.pt'

EXPERT REACHER SIMPLE (OLD) FILTERED AND FROM WHICH WE TOOK VIDEOS
env_mode='reacher',  # o 'reacher'
net_type='simple',
path_to_model='experts_reacher/simple/reacher_expert_filtered/batch_size_512_lr_1e-3/expert_policy.pt'

----STUDENT REACHER SIMPLE





"""