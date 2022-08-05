from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3 import PPO
import tensorboard
import gym
import os 
import time

model_name = "CartPole-v1"
model_dir = f"model/{model_name}"
log_dir = f"./logs"

if not os.path.exists(model_dir):
    os.makedirs(model_dir)

if not os.path.exists(log_dir):
    os.makedirs(log_dir)

# Hyper params
num_envs = 1
train = False
learning_rate =  0.021601781885390386
num_steps = 256
batch_size = 256
n_epochs = 1
gamma = 0.95

env = make_vec_env(model_name, num_envs)
policy = "MlpPolicy"
total_timestamps = 30000

if train:
    env.reset()
    model = PPO(policy= policy,
                env = env,
                learning_rate= learning_rate,
                n_steps = num_steps,
                batch_size= batch_size,
                n_epochs= n_epochs,
                tensorboard_log= log_dir,
                verbose=1)
    model.learn(total_timesteps= total_timestamps)
    err = model.save(f"{model_dir}/{model_name}")
    print("model saved: "+ err)

if not train:
    state = env.reset()
    model = PPO.load(f"{model_dir}/{model_name}")
    for i in range(2000):
        action, _state = model.predict(state, deterministic= True)
        state, reward, done, info = env.step(action)
        env.render()