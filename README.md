# IL4ReacherEnv

# Instructions
1. **Clone the repository**:

 ```sh 
 git clone "https://github.com/cybernetic-m/IL4ReacherEnv.git"
cd IL4ReacherEnv 
 ```

2. Create a virtual environemnt:

 ```sh 
sudo apt-get install python3-venv
python -m venv venv
```

- on Linux/MacOS:
```sh 
source venv/bin/activate
 ```

- on Windows (Command Prompt):

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

