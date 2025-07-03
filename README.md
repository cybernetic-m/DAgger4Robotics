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

<img src="./images/slide_7.png" alt="Description" width="700" height = "200" />

**DAgger**
To train the student we have used DAgger algorithm with a Shallow MLP network.

<img src="./images/slide_8.png" alt="Description" width="700" height = "200" />



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

```

---

## 📊 Reacher Performance
For Reacher we have tested three agent using the Expert dataset (the left), a filtered dataset (homemade) taking only mean reward over -0.1 (the center), and finally the Medium dataset (the right).

<p align="left">
  <img src="images/reacher_1.gif" width="33%" />
  <img src="images/reacher_2.gif" width="33%" />
   <img src="images/reacher_3.gif" width="33%" />
</p>

This is a comparison between the Deep MLP Teacher (the left) and the Shallow MLP student (the right).

<p align="left">
  <img src="images/reacher_4.gif" width="45%" />
  <img src="images/reacher_5.gif" width="45%" />
</p>

---

## 📊 Franka Kitchen Performance

![Demo GIF](path/to/demo.gif) 

---

## 👤 Author

**Massimo Romano**  
GitHub: [@yourhandle](https://github.com/yourhandle)  
LinkedIn: [yourprofile](https://linkedin.com/in/yourprofile)
Website: [yourprofile](https://linkedin.com/in/yourprofile)

---

## 📄 License

This project is licensed under the MIT License.  
See the [LICENSE](./LICENSE) file for details.
