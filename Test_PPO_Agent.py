import os
import hashlib
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

ppo_path = os.path.join('Learnings', 'Training', 'Saved Models', 'PPO_491k_CarRacing_Model')

enviroment_name = 'CarRacing-v3'
env = gym.make(enviroment_name, render_mode='human')

env = DummyVecEnv([lambda: env]) # Wrapper
model = PPO.load(ppo_path, env=env)

episodes = 5
for i in range(1, episodes + 1):
    observation = env.reset()
    done = False
    score = 0

    while not done:
        action, _ = model.predict(observation) # Now using Model to predict
        observation, reward, done, info = env.step(action)
        score += reward
    print('Episode:{} Score:{}'.format(i, score))
env.close()