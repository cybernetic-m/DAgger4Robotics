# 👨‍🍳 Imitation Learning in a kitchen environment 🤖

<img src="./images/image.png" alt="Description" width="400" height = "300" />


## 📖 Description
In this project we have implemented Behavioral Cloning (BCO) and DAgger (Dataset Aggregation) in two reinforcement learning environments:

1. [Reacher](https://gymnasium.farama.org/environments/mujoco/reacher/) (Gymnasium)
   
   ![Banner](images/reacher.png)
   
3. [Franka Kitchen](https://robotics.farama.org/envs/franka_kitchen/franka_kitchen/) (Gymnasium-Robotics)
   
   ![Banner](images/FrankaKitchen.png)

In particular we have used firstly the Reacher environment in order to test DAgger in a more "simple" env, while after that we have tested its performance in a more complicated one, Franka Kitchen. 

## 📚 Datasets
In order to use the BCO method, that is a supervised learning method, we have used [Minari](https://minari.farama.org/) datasets for both environments:

1. [Reacher Expert](https://minari.farama.org/datasets/mujoco/reacher/expert-v0/) and [Reacher Medium](https://minari.farama.org/datasets/mujoco/reacher/medium-v0/) datasets
2. [Franka Kitchen Complete](https://minari.farama.org/datasets/D4RL/kitchen/complete-v2/) (we have cutted the complete trajectories in order to do only the "microwave" task!)


---

## 🔧 Instructions

### 1. Clone the repository

```sh
git clone https://github.com/yourusername/yourproject.git
cd yourproject
```

### 2. Run the 

#### On Ubuntu/Linux:

```sh
sudo apt-get install python3-venv  # If not already installed
python3 -m venv venv
source venv/bin/activate
```
## 🗂 Folder Structure

```
yourproject/
├── public/
├── src/
│   ├── components/
│   ├── pages/
│   └── ...
├── .env.example
└── README.md
```

---

## 🎬 Demo

[🌐 Live Demo](https://your-demo-link.com)  
[📘 Documentation](https://your-docs-link.com)

![Demo GIF](path/to/demo.gif) <!-- Replace with actual demo GIF -->

---

## 📸 Screenshots

| UI View | Description |
|--------|-------------|
| ![Home](path/to/screenshot1.png) | Home screen |
| ![Feature](path/to/screenshot2.png) | Feature showcase |

---


---

## 🧪 Running Tests

```sh
npm test  # or pytest / unittest depending on your project
```

---

## ⚙️ Configuration

If your project uses environment variables, copy `.env.example` to `.env` and configure as needed:

```sh
cp .env.example .env
```

---

## 🧱 Built With

- [React](https://reactjs.org/)
- [Tailwind CSS](https://tailwindcss.com/)
- [Node.js](https://nodejs.org/)
- Any other major libraries/frameworks

---

## 👤 Author

**Your Name**  
GitHub: [@yourhandle](https://github.com/yourhandle)  
LinkedIn: [yourprofile](https://linkedin.com/in/yourprofile)

---

## 📄 License

This project is licensed under the MIT License.  
See the [LICENSE](./LICENSE) file for details.

---

## 🙌 Acknowledgments

- Inspiration, libraries, or tools used  
- Tutorials or reference material  
- Contributors

---

> "_A short motivational quote or development philosophy._"
