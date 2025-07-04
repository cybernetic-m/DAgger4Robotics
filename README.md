# 👨‍🍳 Imitation Learning in a kitchen environment 🤖

<img src="./images/image.png" alt="Description" width="400" height = "300" />

## 📖 Description
<p align="left">
  <img src="images/reacher_1_new.gif" width="45%" />
  <img src="images/kitchen_1.gif" width="45%" />
</p>
In this project we have implemented Behavioral Cloning (BCO) and DAgger (Dataset Aggregation) in two reinforcement learning environments:

1. [Reacher](https://gymnasium.farama.org/environments/mujoco/reacher/) (Gymnasium)
   
<img src="./images/reacher.png" alt="Description" width="200" height = "200" />

3. [Franka Kitchen](https://robotics.farama.org/envs/franka_kitchen/franka_kitchen/) (Gymnasium-Robotics)
   
<img src="./images/FrankaKitchen.png" alt="Description" width="200" height = "200" />

**BCO**

For BCO we have used an MLP deep network.

<img src="./images/slide_7.png" alt="Description" />

**DAgger**

To train the student we have used DAgger algorithm with a Shallow MLP network (for Policy Distillation).

<img src="./images/slide_8.png" alt="Description" />

In particular we have used firstly the Reacher environment in order to test DAgger in a more "simple" env, while after that we have tested its performance in a more complicated one, Franka Kitchen. 

## 📚 Datasets
In order to use the BCO method, that is a supervised learning method, we have used [Minari](https://minari.farama.org/) datasets for both environments:

1. [Reacher Expert](https://minari.farama.org/datasets/mujoco/reacher/expert-v0/) and [Reacher Medium](https://minari.farama.org/datasets/mujoco/reacher/medium-v0/) datasets
2. [Franka Kitchen Complete](https://minari.farama.org/datasets/D4RL/kitchen/complete-v2/) (we have cutted the complete trajectories in order to do only the "microwave" task!)

---

## 🔧 Instructions

### 1. Clone the repository

```sh
git clone "https://github.com/cybernetic-m/DAgger4Robotics.git"
cd DAgger4Robotics
```
To run the following notebooks, you can upload the notebook directly on [Colab](https://colab.research.google.com/).

### 2. Run the "expert.ipynb" notebook (Behavioral Cloning Training)
In this notebook you can train with BCO both an expert for Reacher and Franka Kitchen.

### 3. Run the "reacher.ipynb" notebook (DAgger for Reacher)
In this notebook you can train with DAgger a student for Reacher env.

### 4. Run the "kitchen.ipynb" notebook (DAgger for Franka Kitchen)
In this notebook you can train with DAgger a student for Franka Kitchen env.


## 🗂 Folder Structure

```
├── dagger
│   └── DAgger.py
├── dataset
│   └── myDatasetClass.py
├── expert.ipynb
├── experts_kitchen
│   ├── deep
│   │   └── kitchen_complete
│   │       ├── batch_size_128_lr_1e-3
│   │       │   ├── expert_policy.pt
│   │       │   ├── test_metrics.json
│   │       │   ├── train_metrics.json
│   │       │   └── val_metrics.json
│   │       ├── batch_size_128_lr_1e-4
│   │       │   ├── expert_policy.pt
│   │       │   ├── test_metrics.json
│   │       │   ├── train_metrics.json
│   │       │   └── val_metrics.json
│   │       ├── batch_size_256_lr_1e-3
│   │       │   ├── expert_policy.pt
│   │       │   ├── test_metrics.json
│   │       │   ├── train_metrics.json
│   │       │   └── val_metrics.json
│   │       ├── batch_size_256_lr_1e-4
│   │       │   ├── expert_policy.pt
│   │       │   ├── test_metrics.json
│   │       │   ├── train_metrics.json
│   │       │   └── val_metrics.json
│   │       ├── batch_size_32_lr_1e-3
│   │       │   ├── expert_policy.pt
│   │       │   ├── test_metrics.json
│   │       │   ├── train_metrics.json
│   │       │   └── val_metrics.json
│   │       ├── batch_size_32_lr_1e-4
│   │       │   ├── expert_policy.pt
│   │       │   ├── test_metrics.json
│   │       │   ├── train_metrics.json
│   │       │   └── val_metrics.json
│   │       ├── batch_size_512_lr_1e-3
│   │       │   ├── expert_policy.pt
│   │       │   ├── test_metrics.json
│   │       │   ├── train_metrics.json
│   │       │   └── val_metrics.json
│   │       ├── batch_size_512_lr_1e-4
│   │       │   ├── expert_policy.pt
│   │       │   ├── test_metrics.json
│   │       │   ├── train_metrics.json
│   │       │   └── val_metrics.json
│   │       ├── batch_size_64_lr_1e-3
│   │       │   ├── expert_policy.pt
│   │       │   ├── mean_rewards_cold_start.json
│   │       │   ├── mean_rewards_cold_start_noise_0.04.json
│   │       │   ├── mean_rewards_cold_start_noise_0.31.json
│   │       │   ├── mean_rewards_cold_start_noise_0.3378.json
│   │       │   ├── mean_rewards_noise_0.15.json
│   │       │   ├── mean_rewards_noise_0.3378.json
│   │       │   ├── mean_rewards_noise_0.3.json
│   │       │   ├── mean_rewards_normal.json
│   │       │   ├── test_metrics.json
│   │       │   ├── train_metrics.json
│   │       │   └── val_metrics.json
│   │       └── batch_size_64_lr_1e-4
│   │           ├── expert_policy.pt
│   │           ├── test_metrics.json
│   │           ├── train_metrics.json
│   │           └── val_metrics.json
│   └── simple
│       └── kitchen_complete
│           └── batch_size_64_lr_1e-3
│               ├── expert_policy.pt
│               ├── mean_rewards_normal.json
│               ├── test_metrics.json
│               ├── train_metrics.json
│               └── val_metrics.json
├── experts_reacher
│   ├── deep
│   │   ├── reacher_expert_filtered
│   │   │   └── batch_size_128_lr_1e-4
│   │   │       ├── expert_policy.pt
│   │   │       ├── mean_rewards_reacher.json
│   │   │       ├── test_metrics.json
│   │   │       ├── test_metrics_on_expert_test_set.json
│   │   │       ├── train_metrics.json
│   │   │       └── val_metrics.json
│   │   ├── reacher_expert_not_filtered
│   │   │   ├── batch_size_128_lr_1e-3
│   │   │   │   ├── expert_policy.pt
│   │   │   │   ├── test_metrics.json
│   │   │   │   ├── train_metrics.json
│   │   │   │   └── val_metrics.json
│   │   │   ├── batch_size_128_lr_1e-4
│   │   │   │   ├── expert_policy.pt
│   │   │   │   ├── mean_rewards_reacher.json
│   │   │   │   ├── test_metrics.json
│   │   │   │   ├── train_metrics.json
│   │   │   │   └── val_metrics.json
│   │   │   ├── batch_size_256_lr_1e-3
│   │   │   │   ├── expert_policy.pt
│   │   │   │   ├── test_metrics.json
│   │   │   │   ├── train_metrics.json
│   │   │   │   └── val_metrics.json
│   │   │   ├── batch_size_512_lr_1e-3
│   │   │   │   ├── expert_policy.pt
│   │   │   │   ├── test_metrics.json
│   │   │   │   ├── train_metrics.json
│   │   │   │   └── val_metrics.json
│   │   │   └── batch_size_64_lr_1e-3
│   │   │       ├── expert_policy.pt
│   │   │       ├── test_metrics.json
│   │   │       ├── train_metrics.json
│   │   │       └── val_metrics.json
│   │   └── reacher_medium_not_filtered
│   │       └── batch_size_128_lr_1e-4
│   │           ├── expert_policy.pt
│   │           ├── mean_rewards_reacher.json
│   │           ├── test_metrics.json
│   │           ├── test_metrics_on_expert_test_set.json
│   │           ├── train_metrics.json
│   │           └── val_metrics.json
│   └── simple
│       ├── reacher_expert_filtered
│       │   ├── batch_size_128_lr_1e-3
│       │   │   ├── expert_policy.pt
│       │   │   ├── test_metrics.json
│       │   │   ├── train_metrics.json
│       │   │   └── val_metrics.json
│       │   ├── batch_size_256_lr_1e-3
│       │   │   ├── expert_policy.pt
│       │   │   ├── test_metrics.json
│       │   │   ├── train_metrics.json
│       │   │   └── val_metrics.json
│       │   ├── batch_size_32_lr_1e-3
│       │   │   ├── expert_policy.pt
│       │   │   ├── test_metrics.json
│       │   │   ├── train_metrics.json
│       │   │   └── val_metrics.json
│       │   ├── batch_size_512_lr_1e-3
│       │   │   ├── expert_policy.pt
│       │   │   ├── test_metrics.json
│       │   │   ├── train_metrics.json
│       │   │   └── val_metrics.json
│       │   └── batch_size_64_lr_1e-3
│       │       ├── expert_policy.pt
│       │       ├── test_metrics.json
│       │       ├── train_metrics.json
│       │       └── val_metrics.json
│       └── reacher_expert_not_filtered
│           ├── batch_size_128_lr_1e-3
│           │   ├── expert_policy.pt
│           │   ├── test_metrics.json
│           │   ├── train_metrics.json
│           │   └── val_metrics.json
│           ├── batch_size_256_lr_1e-3
│           │   ├── expert_policy.pt
│           │   ├── test_metrics.json
│           │   ├── train_metrics.json
│           │   └── val_metrics.json
│           ├── batch_size_32_lr_1e-3
│           │   ├── expert_policy.pt
│           │   ├── test_metrics.json
│           │   ├── train_metrics.json
│           │   └── val_metrics.json
│           ├── batch_size_512_lr_1e-3
│           │   ├── expert_policy.pt
│           │   ├── test_metrics.json
│           │   ├── train_metrics.json
│           │   └── val_metrics.json
│           └── batch_size_64_lr_1e-3
│               ├── expert_policy.pt
│               ├── test_metrics.json
│               ├── train_metrics.json
│               └── val_metrics.json
├── images
│   ├── FrankaKitchen.png
│   ├── image.png
│   ├── kitchen_10.gif
│   ├── kitchen_1.gif
│   ├── kitchen_1.png
│   ├── kitchen_2.gif
│   ├── kitchen_2.png
│   ├── kitchen_3.gif
│   ├── kitchen_4.gif
│   ├── kitchen_5.gif
│   ├── kitchen_6.gif
│   ├── kitchen_7.gif
│   ├── kitchen_8.gif
│   ├── kitchen_9.gif
│   ├── reacher_1.gif
│   ├── reacher_1_new.gif
│   ├── reacher_1.png
│   ├── reacher_2.gif
│   ├── reacher_2_new.gif
│   ├── reacher_2.png
│   ├── reacher_3.gif
│   ├── reacher_3_new.gif
│   ├── reacher_4.gif
│   ├── reacher_4_new.gif
│   ├── reacher_5.gif
│   ├── reacher_5_new.gif
│   ├── reacher.png
│   ├── slide_7.png
│   └── slide_8.png
├── kitchen.ipynb
├── LICENSE
├── model
│   ├── DeepPolicyNet.py
│   ├── NetworkInterface.py
│   └── SimplePolicyNet.py
├── reacher.ipynb
├── README.md
├── simulator
│   └── Simulator.py
├── students_kitchen
│   └── simple
│       └── kitchen_complete
│           ├── batch_size_10_lr_1e-3_iterations_20_rollouts_per_iteration_20_num_epochs_5_betaMode_inverse_exponential_beta_k_0.5
│           │   ├── student_policy_19.pt
│           │   ├── train_metrics_student_19.json
│           │   └── val_metrics_student_19.json
│           ├── batch_size_128_lr_1e-3_iterations_20_rollouts_per_iteration_20_num_epochs_5_betaMode_inverse_exponential_beta_k_0.5
│           │   ├── student_policy_18.pt
│           │   ├── train_metrics_student_18.json
│           │   └── val_metrics_student_18.json
│           ├── batch_size_256_lr_1e-3_iterations_20_rollouts_per_iteration_20_num_epochs_5_betaMode_inverse_exponential_beta_k_0.5
│           │   ├── student_policy_18.pt
│           │   ├── train_metrics_student_18.json
│           │   └── val_metrics_student_18.json
│           ├── batch_size_3_lr_1e-3_iterations_20_rollouts_per_iteration_20_num_epochs_5_betaMode_inverse_exponential_beta_k_0.5
│           │   ├── student_policy_11.pt
│           │   ├── train_metrics_student_11.json
│           │   └── val_metrics_student_11.json
│           ├── batch_size_512_lr_1e-3_iterations_20_rollouts_per_iteration_20_num_epochs_5_betaMode_exponential_exponential_beta_k_0.1
│           │   ├── student_policy_18.pt
│           │   ├── train_metrics_student_18.json
│           │   └── val_metrics_student_18.json
│           ├── batch_size_512_lr_1e-3_iterations_20_rollouts_per_iteration_20_num_epochs_5_betaMode_exponential_exponential_beta_k_0.2
│           │   ├── student_policy_17.pt
│           │   ├── train_metrics_student_17.json
│           │   └── val_metrics_student_17.json
│           ├── batch_size_512_lr_1e-3_iterations_20_rollouts_per_iteration_20_num_epochs_5_betaMode_exponential_exponential_beta_k_0.3
│           │   ├── mean_rewards_cold_start.json
│           │   ├── mean_rewards_cold_start_noise_0.04.json
│           │   ├── mean_rewards_cold_start_noise_0.31.json
│           │   ├── mean_rewards_cold_start_noise_0.3378.json
│           │   ├── mean_rewards_noise_0.15.json
│           │   ├── mean_rewards_noise_0.3378.json
│           │   ├── mean_rewards_noise_0.3.json
│           │   ├── mean_rewards_normal.json
│           │   ├── student_policy_15.pt
│           │   ├── test_metrics_on_expert_test_set.json
│           │   ├── train_metrics_student_15.json
│           │   └── val_metrics_student_15.json
│           ├── batch_size_512_lr_1e-3_iterations_20_rollouts_per_iteration_20_num_epochs_5_betaMode_exponential_exponential_beta_k_0.4
│           │   ├── student_policy_19.pt
│           │   ├── train_metrics_student_19.json
│           │   └── val_metrics_student_19.json
│           ├── batch_size_512_lr_1e-3_iterations_20_rollouts_per_iteration_20_num_epochs_5_betaMode_exponential_exponential_beta_k_0.5
│           │   ├── mean_rewards_big_noise.json
│           │   ├── student_policy_16.pt
│           │   ├── train_metrics_student_16.json
│           │   └── val_metrics_student_16.json
│           ├── batch_size_512_lr_1e-3_iterations_20_rollouts_per_iteration_20_num_epochs_5_betaMode_exponential_exponential_beta_k_0.6
│           │   ├── student_policy_18.pt
│           │   ├── train_metrics_student_18.json
│           │   └── val_metrics_student_18.json
│           ├── batch_size_512_lr_1e-3_iterations_20_rollouts_per_iteration_20_num_epochs_5_betaMode_exponential_exponential_beta_k_0.7
│           │   ├── student_policy_19.pt
│           │   ├── train_metrics_student_19.json
│           │   └── val_metrics_student_19.json
│           ├── batch_size_512_lr_1e-3_iterations_20_rollouts_per_iteration_20_num_epochs_5_betaMode_exponential_exponential_beta_k_0.8
│           │   ├── student_policy_19.pt
│           │   ├── train_metrics_student_19.json
│           │   └── val_metrics_student_19.json
│           ├── batch_size_512_lr_1e-3_iterations_20_rollouts_per_iteration_20_num_epochs_5_betaMode_exponential_exponential_beta_k_0.9
│           │   ├── student_policy_17.pt
│           │   ├── train_metrics_student_17.json
│           │   └── val_metrics_student_17.json
│           ├── batch_size_512_lr_1e-3_iterations_20_rollouts_per_iteration_20_num_epochs_5_betaMode_exponential_exponential_beta_k_2
│           │   ├── student_policy_18.pt
│           │   ├── train_metrics_student_18.json
│           │   └── val_metrics_student_18.json
│           ├── batch_size_512_lr_1e-3_iterations_20_rollouts_per_iteration_20_num_epochs_5_betaMode_inverse_exponential_beta_k_0.5
│           │   ├── student_policy_18.pt
│           │   ├── train_metrics_student_18.json
│           │   └── val_metrics_student_18.json
│           ├── batch_size_64_lr_1e-3_iterations_20_rollouts_per_iteration_20_num_epochs_5_betaMode_exponential_exponential_beta_k_0.5
│           │   ├── student_policy_16.pt
│           │   ├── train_metrics_student_16.json
│           │   └── val_metrics_student_16.json
│           ├── batch_size_64_lr_1e-3_iterations_20_rollouts_per_iteration_20_num_epochs_5_betaMode_inverse_exponential_beta_k_0.5
│           │   ├── student_policy_14.pt
│           │   ├── train_metrics_student_14.json
│           │   └── val_metrics_student_14.json
│           ├── batch_size_64_lr_1e-3_iterations_20_rollouts_per_iteration_20_num_epochs_5_betaMode_linear_exponential_beta_k_0.5
│           │   ├── student_policy_19.pt
│           │   ├── train_metrics_student_19.json
│           │   └── val_metrics_student_19.json
│           ├── batch_size_64_lr_1e-3_iterations_50_rollouts_per_iteration_20_num_epochs_5_betaMode_exponential_exponential_beta_k_0.5
│           │   ├── student_policy_48.pt
│           │   ├── train_metrics_student_48.json
│           │   └── val_metrics_student_48.json
│           └── batch_size_64_lr_1e-3_iterations_50_rollouts_per_iteration_20_num_epochs_5_betaMode_inverse_exponential_beta_k_0.5
│               ├── student_policy_46.pt
│               ├── train_metrics_student_46.json
│               └── val_metrics_student_46.json
├── students_reacher
│   └── simple
│       └── reacher_expert_not_filtered
│           ├── batch_size_32_lr_1e-3_iterations_20_rollouts_per_iteration_20_num_epochs_5_betaMode_exponential_exponential_beta_k_0.5
│           │   ├── mean_rewards_reacher.json
│           │   ├── others_param.json
│           │   ├── student_policy_19.pt
│           │   ├── train_metrics_student_19.json
│           │   └── val_metrics_student_19.json
│           ├── batch_size_32_lr_1e-3_iterations_20_rollouts_per_iteration_20_num_epochs_5_betaMode_inverse_exponential_beta_k_0.5
│           │   ├── mean_rewards_reacher.json
│           │   ├── others_param.json
│           │   ├── student_policy_17.pt
│           │   ├── test_metrics_on_expert_test_set.json
│           │   ├── train_metrics_student_17.json
│           │   └── val_metrics_student_17.json
│           └── batch_size_32_lr_1e-3_iterations_20_rollouts_per_iteration_20_num_epochs_5_betaMode_linear_exponential_beta_k_0.5
│               ├── mean_rewards_reacher.json
│               ├── others_param.json
│               ├── student_policy_19.pt
│               ├── train_metrics_student_19.json
│               └── val_metrics_student_19.json
├── test
│   └── test.py
├── training
│   ├── one_epoch.py
│   └── train.py
└── utils
    ├── calculate_metrics.py
    ├── convert_fc_to_sequential_keys.py
    ├── is_running_in_colab.py
    └── preprocess_dataset.py


```

---

## 📊 Reacher Performance
For Reacher we have tested three agents using the Expert dataset (the left), a filtered dataset (homemade) taking only mean reward over -0.1 (the center), and finally the Medium dataset (the right).

<p align="left">
  <img src="images/reacher_1_new.gif" width="33%" />
  <img src="images/reacher_2_new.gif" width="33%" />
   <img src="images/reacher_3_new.gif" width="33%" />
</p>

This is a comparison between the Deep MLP Teacher (the left) and the Shallow MLP student (the right).

<p align="left">
  <img src="images/reacher_4_new.gif" width="45%" />
  <img src="images/reacher_5_new.gif" width="45%" />
</p>

The following are the performance of the metrics and the mean reward on 1k episodes:

<img src="./images/reacher_1.png" alt="Description"/>

<img src="./images/reacher_2.png" alt="Description"/>

---

## 📊 Franka Kitchen Performance

For Franka Kitchen we have tested a Deep Teacher (BCO) using the Complete Minari dataset (the left), a Shallow Teacher (BCO) using also the Complete Minari dataset (center), and a Shallow Student (DAgger) (the right).

<p align="left">
  <img src="images/kitchen_1.gif" width="33%" />
  <img src="images/kitchen_2.gif" width="33%" />
   <img src="images/kitchen_3.gif" width="33%" />
</p>

We have tested the Deep Teacher in presence of noise (setting the "robot_noise_ratio" variable of Franka Kitchen Environment). You can see the `robot_noise_ratio=0.29` (the left) and `robot_noise_ratio=0.30`.

<p align="left">
  <img src="images/kitchen_5.gif" width="45%" />
  <img src="images/kitchen_6.gif" width="45%" />
</p>

We have tested also the Shallow Student obtained with DAgger training seeing that is more robust to the noise: in fact, you can see the `robot_noise_ratio=0.29` (the left) and `robot_noise_ratio=0.3378` (that it is able to recover the initial error).

<p align="left">
  <img src="images/kitchen_7.gif" width="45%" />
  <img src="images/kitchen_8.gif" width="45%" />
</p>

Finally, we have tested a "Cold Start" situation, in which for the first 27 steps of the episode the joint/end_effector velocities are zeroed out. Both the Deep Teacher (the left) and the Shallow Student (the right) behave correctly to this situation.

<p align="left">
  <img src="images/kitchen_9.gif" width="45%" />
  <img src="images/kitchen_10.gif" width="45%" />
</p>

The following are the tables with the metrics performance and the mean reward on 1k episodes:

<img src="./images/kitchen_1.png" alt="Description"/>

<img src="./images/kitchen_2.png" alt="Description"/>


---

## 👤 Author

**Massimo Romano**  
GitHub: [@cybernetic-m](https://github.com/cybernetic-m)  

LinkedIn: [Massimo Romano](https://www.linkedin.com/in/massimo-romano-01/)

Website: [Massimo Romano](https://sites.google.com/studenti.uniroma1.it/romano/home-page?authuser=0)

**Luca Del Signore**
GitHub: [@Puaison](https://github.com/Puaison)  

---

## 📄 License

This project is licensed under the MIT License.  
See the [LICENSE](./LICENSE) file for details.
