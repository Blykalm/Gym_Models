import gym
import time
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
import os
import tensorboard

#constant params
model_name = "LunarLander-v2"
model_dir = f"model/{model_name}"
log_dir = "./logs"

if not os.path.exists(model_dir):
    os.makedirs(model_dir)


# hyper params
train = False
num_envs = 1
num_timestamps = 1e6
policy = 'MlpPolicy'
num_steps = 1024
batch_size = 32
lam = 0.98
gamma = 0.999
num_epochs = 4
ent_coef = 0.01
learning_rate = 0.0004

env = make_vec_env(model_name, num_envs)

if train:
    model = PPO(policy = policy,
                env = env,
                n_steps= num_steps,
                batch_size= batch_size,
                gamma= gamma,
                gae_lambda= lam,
                n_epochs = num_epochs,
                ent_coef= ent_coef,
                verbose= 1,
                tensorboard_log=log_dir)
    model.learn(total_timesteps=num_timestamps, log_interval=4)
    model.save(f"{model_dir}/{model_name}")

if not train:
    state = env.reset()
    model = PPO.load(f"{model_dir}/{model_name}")
    for i in range(1000):
        action , _state = model.predict(state, deterministic= True)
        state, reward, done, info = env.step(action)
        time.sleep(0)
        env.render()

        


