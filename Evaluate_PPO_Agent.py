import os
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy

enviroment_name = 'CarRacing-v3'
env = gym.make(enviroment_name, render_mode='human')

ppo_path = os.path.join('Training', 'Saved Models', 'PPO_491k_CarRacing_Model')
model = PPO.load(ppo_path, env=env)

rew, std = evaluate_policy(model, env, n_eval_episodes=1, render=True)

env.close()

print(rew, std)