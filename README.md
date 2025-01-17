# RL_project-Hindsight-Goal-Ranking-for-Pick-and-Place-tasks

The goal of this project is to implement and evaluate a reinforcement learning
(RL) approach using Hindsight Goal Ranking (HGR) in the context of sparse
reward environments. The approach will be applied to a fixed-base manipulator
performing a pick-and-place operation in simulation. The experiments will be
conducted using the OpenAI Gym Fetch environment.


# Dependancies 

- In this project the package `gymnasium-robotics` is required. 
Be careful on the installation because the download via PyPI with `pip install gymnasium-robotics` present a bug in which the base of the manipulator has an offset resulting in some goal reach configuarations to be outside from the workspace.
I managed this issue by installing the package directly from GitHub via:
`git clone https://github.com/Farama-Foundation/Gymnasium-Robotics.git
cd Gymnasium-Robotics`
`pip install -e .`

- Install the requirements:
`pip install requirements.txt`

- Mujoco engine is required check the installation procedure in the official site: [https://robotics.farama.org/content/installation/](https://robotics.farama.org/content/installation/)