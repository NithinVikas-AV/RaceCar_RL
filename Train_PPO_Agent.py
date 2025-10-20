import os
import hashlib
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv

def make_env():
    def _init():
        env = gym.make("CarRacing-v3")  # no render_mode here (slows training)
        env = Monitor(env)              # logs reward/episode length
        return env
    return _init

num_envs = 8   # adjust based on your CPU
env = SubprocVecEnv([make_env() for _ in range(num_envs)])

log_path = os.path.join('Learnings', 'Training', 'Logs')

model = PPO("CnnPolicy", env, verbose=1, tensorboard_log=log_path)

ppo_path = os.path.join('Learnings', 'Training', 'Saved Models', 'PPO_491k_CarRacing_Model')

model.learn(total_timesteps=491520)
model.save(ppo_path)

# Generate a SHA-256 hash for model integrity verification
with open(ppo_path + ".zip", "rb") as f:
    file_bytes = f.read()
    hash_value = hashlib.sha256(file_bytes).hexdigest()
    with open('hash_value.txt', 'w') as hash_file:
        hash_file.write(hash_value)