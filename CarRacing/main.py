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
num_envs = 1
train = False
learning_rate =  0.5
num_steps = 2048
batch_size = 32
n_epochs = 20
gamma = 0.99

env = make_vec_env(model_name, num_envs)
#env = gym.make(model_name)
policy = 'MlpPolicy'
total_timestamps = 50000

if train:
    env.reset()
    model = PPO(policy=policy,
                env=env,
                learning_rate = learning_rate,
                n_steps = num_steps,
                batch_size= batch_size,
                n_epochs= n_epochs,
                gamma= gamma,
                tensorboard_log=log_dir,
                verbose=1)
    model.learn(total_timesteps=total_timestamps)
    err = model.save(f"{model_dir}/{model_name}")
    print("model saved: " + err)

if not train:
    state = env.reset()
    model = PPO.load(f"{model_dir}/{model_name}")
    for i in range(1000):
        action , _state = model.predict(state, deterministic= True)
        state, reward, done, info = env.step(action)
        time.sleep(0)
        env.render()
