🧠 Reinforcement_Learning

A comprehensive collection of Reinforcement Learning implementations built from the ground up using PyTorch and PyTorch Lightning — ranging from foundational algorithms to advanced deep RL methods and real projects.
---

## 🧭 Table of Contents
- [Overview](#overview)
- [Implemented Algorithms](#implemented-algorithms)
- [Repository Structure](#repository-structure)
- [Install & Setup](#install--setup)
- [Usage](#usage)
- [Results & Media](#results--media)
- [Future Work](#future-work)
- [References](#references)
- [License](#license)

---

📂 Repository Overview

This repository is divided into four main sections for clarity and learning progression:

🔹 1. Fundamentals

Covers essential concepts of Reinforcement Learning, including Markov Decision Processes (MDPs), policy and value iteration, Monte Carlo methods, SARSA, and Q-learning.

Key highlights:

MDP_introduction.ipynb

policy_iteration_complete.ipynb

value_iteration_complete.ipynb

n_step_sarsa_complete.ipynb

qlearning_complete.ipynb

---

🔹 2. Policy Gradient Methods

Implements algorithms based on policy gradient and actor-critic methods:

REINFORCE (CartPole)

Advantage Actor-Critic (A2C)

🔹 3. Deep Reinforcement Learning (DRL)

Advanced deep RL algorithms implemented using PyTorch Lightning for scalability and reproducibility.

Includes:

DQN

DDPG

Twin Delayed DDPG (TD3)

Soft Actor-Critic (SAC)

Normalized Advantage Function (NAF)

Hindsight Experience Replay (HER)

Hyperparameter tuning experiments

🔹 4. Projects

Practical implementations and applications of RL in games and custom environments.

SnakeRL Project: Training an RL agent to play Snake.

Armed Bandit Gym: Exploration strategies and reward analysis.

🎥 Media & Visualizations

In the media/soft_actor_critic folder:

FetchReachDense environment simulation videos (SAC agent)

Training metrics: Episode return curves, policy and Q-loss graphs

Model checkpoints: Saved PyTorch Lightning weights

⚙️ Technologies Used

Language: Python 3.10+

Frameworks: PyTorch, PyTorch Lightning

Libraries: NumPy, Matplotlib, Gymnasium/OpenAI Gym, TensorBoard

Tools: Jupyter Notebook

🚀 Future Work

Integration with PyBullet and MuJoCo environments

Visual RL with CNN-based policies

Additional environments (LunarLander, FetchPush, etc.)

Deployment-ready API for trained agents

Reinforcement_Learning/
│
├── 01_Fundamentals/
│   ├── MDP_introduction.ipynb
│   ├── armed_bandit_problem.ipynb
│   ├── policy_iteration_complete.ipynb
│   ├── value_iteration_complete.ipynb
│   ├── on_policy_control_complete.ipynb
│   ├── off_policy_control_complete.ipynb
│   ├── n_step_sarsa_complete.ipynb
│   ├── qlearning_complete.ipynb
│   ├── on_policy_constant_alpha_mc_complete.ipynb
│   └── continuous_observation_spaces_complete.ipynb
│
├── 02_Policy_Gradient/
│   ├── reinforce_CartPole_complete.ipynb
│   └── advantage_actor_critic_complete.ipynb
│
├── 03_Deep_RL/
│   ├── RL_DQ-SAC/
│   │   ├── dqn_pytorch_lightning.ipynb
│   │   ├── deep_deterministic_policy_gradient.ipynb
│   │   ├── twin_delayed_ddpg.ipynb
│   │   ├── soft_actor_critic.ipynb
│   │   ├── normalized_advantage_function.ipynb
│   │   ├── hindsight_experience_replay.ipynb
│   │   └── hyperparameter_tuning.ipynb
│
├── 04_Projects/
│   ├── RL_projects_1/
│   │   ├── armedBanditGym.ipynb
│   │   ├── armed_bandit_gym_env.ipynb
│   │   ├── reward_and_transitions.ipynb
│   │   └── Policy_evaluation.ipynb
│   │
│   └── SnakeRL_project/
│       ├── agent.py
│       ├── model.py
│       ├── game.py
│       ├── helper.py
│       ├── RlSnakeProject.ipynb
│       └── arial.ttf
│
└── media/
    └── soft_actor_critic/
        ├── FetchReachDense_ep_0.mp4
        ├── FetchReachDense_ep_500.mp4
        ├── FetchReachDense_ep_1000.mp4
        ├── FetchReachDense_last_ep.mp4
        ├── episode_return.jpeg
        ├── episode_policy_loss.jpeg
        ├── Q_loss.jpeg
        └── checkpoints/
            └── epoch=1999-step=16000.ckpt


---

## ⚙️ Install & Setup
# Clone the repo
git clone https://github.com/<your-username>/Reinforcement_Learning.git
cd Reinforcement_Learning

# Create virtual environment
python3 -m venv rl_env
source rl_env/bin/activate   # (Linux/Mac)
rl_env\Scripts\activate      # (Windows)

# Install dependencies
pip install -r requirements.txt
gymnasium==1.2.1       # or a version compatible with your code
gym==0.26.2            # for some older environments if needed
torch>=2.1.0           # PyTorch
pytorch-lightning==1.9.5
numpy>=1.24.0
brax==0.12.3
optuna==2.7.0
ipython
matplotlib              # if you use plots anywhere
Optional / helper packages you might also include:
pyvirtualdisplay        # for headless rendering
pillow                  # if you save frames/images



💡 How to Run

# Clone the repository
git clone https://github.com/<your-username>/Reinforcement_Learning.git
cd Reinforcement_Learning

# Install dependencies
pip install -r requirements.txt

# Open notebooks
jupyter notebook

🏆 Author

Abubakar Adam
Reinforcement Learning & Robotics Enthusiast
Passionate about Safe Robot Learning, Gaming AI, and Robotic Simulation Control.
📫 Contact: https://www.linkedin.com/in/abubakarx-adam
