import argparse
import gymnasium as gym
import mujoco
import gymnasium_robotics

import numpy as np

from DDPG import Policy


def evaluate(env=None, n_episodes=1, render=False):
    agent = Policy()
    agent.load()

    env = gym.make("FetchReach-v3", max_episode_steps=200)
    if render:
        env = gym.make("FetchReach-v3", max_episode_steps=200, render_mode='human')
        
    rewards = []
    for episode in range(n_episodes):
        total_reward = 0
        done = False
        s, _ = env.reset()
        while not done:
            action = agent.act(s)
            
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


def main():
    parser = argparse.ArgumentParser(description='Run training and evaluation')
    parser.add_argument('--render', action='store_true')
    parser.add_argument('-t', '--train', action='store_true')
    parser.add_argument('-e', '--evaluate', action='store_true')
    args = parser.parse_args()

    if args.train:
        train()

    if args.evaluate:
        evaluate(render=args.render)

    
if __name__ == '__main__':
    main()
