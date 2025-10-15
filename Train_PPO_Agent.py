import os
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.monitor import Monitor

def make_env():
    def _init():
        env = gym.make("CarRacing-v3")  # no render_mode here (slows training)
        env = Monitor(env)              # logs reward/episode length
        return env
    return _init

num_envs = 8   # adjust based on your CPU
env = SubprocVecEnv([make_env() for _ in range(num_envs)])

log_path = os.path.join("Training", "Logs")

model = PPO("CnnPolicy", env, verbose=1, tensorboard_log=log_path)

ppo_path = os.path.join('Training', 'Saved Models', 'PPO_491k_CarRacing_Model')

model.learn(total_timesteps=491520)
model.save(ppo_path)