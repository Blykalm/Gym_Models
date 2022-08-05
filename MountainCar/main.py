import time
import gym
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3 import DQN
import os

# constant params
model_name = "DQN"
train = False
model_dir = f"model/{model_name}"
num_envs = 1

# hyper params
learning_rate = 0.0004

if not os.path.exists(model_dir):
    os.makedirs(model_dir)

env = make_vec_env('MountainCar-v0', n_envs= num_envs)
env.seed(42)

if train:
    #training DQN model
    model = DQN(policy= "MlpPolicy",
                env = env,
                verbose=1,
                learning_rate = learning_rate)
    model.learn(total_timesteps=2000000, log_interval=4)
    model.save(f'{model_dir}/dqn_MountainCar')


if not train:
    state = env.reset()
    model = DQN.load(f'{model_dir}/dqn_MountainCar')
    #evaluating
    for i in range(1500):
        action, _state = model.predict(state, deterministic= True)
        state, reward, done, info = env.step(action)
        time.sleep(0.01)
        env.render()

