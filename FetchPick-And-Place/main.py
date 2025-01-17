import argparse
import gymnasium as gym
import mujoco
import gymnasium_robotics

import numpy as np

#from DDPG import Policy

gym.register_envs(gymnasium_robotics)

def evaluate(env=None, n_episodes=6, render=False):
    #agent = Policy()
    #agent.load()

    max_episode_steps=100
    env = gym.make("FetchPickAndPlace-v4", max_episode_steps=max_episode_steps)
    if render:
        env = gym.make("FetchPickAndPlace-v4", max_episode_steps=max_episode_steps, render_mode = 'human')
        
    rewards = []
    for episode in range(n_episodes):
        total_reward = 0
        done = False
        s, _ = env.reset()
        print('initial state', s)
        for _ in range(max_episode_steps):
            #action = agent.act(s)
            action = env.action_space.sample()
            s, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            total_reward += reward
        
        rewards.append(total_reward)
        
    print('Mean Reward:', np.mean(rewards))
    env.close()


def train():
    agent = Policy()
    agent.train()
    agent.save()

def plot_graphics():
    agent = Policy()
    agent.plot_training_logs()

def main():
    parser = argparse.ArgumentParser(description='Run training and evaluation')
    parser.add_argument('--render', action='store_true')
    parser.add_argument('-t', '--train', action='store_true')
    parser.add_argument('-e', '--evaluate', action='store_true')
    parser.add_argument('-p', '--plot', action='store_true')
    args = parser.parse_args()

    if args.train:
        train()

    if args.evaluate:
        evaluate(render=args.render)
    if args.plot:
        plot_graphics()

    
if __name__ == '__main__':
    main()
