# IL4ReacherEnv

# Instructions
1. **Clone the repository**:

 ```sh 
 git clone "https://github.com/cybernetic-m/IL4ReacherEnv.git"
cd IL4ReacherEnv 
 ```

2. Create a virtual environment:

- Install venv (You can skip if you have it):
 ```sh 
sudo apt-get install python3-venv
```

- Create a venv:
 ```sh 
python -m venv venv
```

- Activate the venv (on Linux/MacOS):
```sh 
source venv/bin/activate
 ```

- Activate the venv  (on Windows Command Prompt):

 ```sh 
venv\Scripts\activate
 ```

3. Install the Mujoco Gymnasium Environments:

```sh 
pip install "gymnasium[mujoco]"
 ```

4. Install the Minari dataset "medium" for the Reacher Env:

```sh 
pip install "minari[all]"
minari download mujoco/reacher/medium-v0
 ```

