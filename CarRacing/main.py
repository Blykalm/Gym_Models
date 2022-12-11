import os
import time
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
import tensorboard
import optuna
import gym

model_name = "CarRacing-v0"
model_dir = f"model/{model_name}"
log_dir = "./logs"

if not os.path.exists(model_dir):
    os.makedirs(model_dir)

# hyper params
num_envs = 2
train = True
learning_rate = 7.830628567129431e-05
num_steps = 32
n_epochs = 20
gamma = 0.98
clip_range = 0.4
batch_size = 16
target_kl = 0.99
gae_lambda = 0.9144383227774225


if train == True:
    num_envs = 2
else:
    num_envs = 1

env = make_vec_env(model_name, num_envs)
#env = gym.make(model_name)
print(env.action_space)
policy = 'MlpPolicy'
total_timestamps = 100000

if train:
    model = PPO(policy=policy,
                env=env,
                learning_rate=learning_rate,
                n_steps=num_steps,
                n_epochs=n_epochs,
                gamma=gamma,
                clip_range=clip_range,
                batch_size=batch_size,
                target_kl= target_kl,
                gae_lambda=gae_lambda,
                tensorboard_log=log_dir,
                verbose=1)
    model.learn(total_timesteps=total_timestamps)
    model.save(f"{model_dir}/{model_name}")
    print("model saved")

if not train:
    state = env.reset()
    model = PPO.load(f"{model_dir}/{model_name}")
    for i in range(1000):
        action, _state = model.predict(state, deterministic=True)
        state, reward, done, info = env.step(action)
        time.sleep(0)
        env.render()
