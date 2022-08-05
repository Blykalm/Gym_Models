import os
import gym
import optuna
import tensorboard
from optuna.samplers import TPESampler
from optuna import trial
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from typing import Any, Dict


def objective(trial)-> Dict[str, Any]:
    # Define the search space
    learning_rate =  trial.suggest_loguniform("learning_rate", 1e-5, 1)
    n_steps = trial.suggest_categorical("n_steps", [8, 16, 32, 64, 128, 256, 512, 1024, 2048])
    n_epochs = trial.suggest_categorical("n_epochs", [1, 5, 10, 20])
    gamma = trial.suggest_categorical("gamma", [0.9, 0.95, 0.98, 0.99, 0.995, 0.999, 0.9999])
    batch_size = trial.suggest_categorical("batch_size", [8, 16, 32, 64, 128, 256, 512])
    
    model_name = "CartPole-v1"
    model_dir = f"model/{model_name}"
    log_dir = "./logs"
    env = make_vec_env(model_name, 1)
    #env = gym.make(model_name)

    policy = 'MlpPolicy'
    total_timestamps = 1000
    model = PPO(policy=policy,
                env=env,
                verbose=1,
                learning_rate=learning_rate,
                n_steps = n_steps,
                n_epochs = n_epochs,
                gamma = gamma,
                batch_size = batch_size)
    model.learn(total_timesteps=total_timestamps)

    state = env.reset()
    action , _state = model.predict(state)
    state, reward, done, info = env.step(action)

    env.close()
    return reward


study = optuna.create_study(study_name="CarRacing",
                            direction='maximize',
                            sampler=TPESampler())
study.optimize(objective,show_progress_bar=True, n_trials=10)

print("##############################################################")
print(study.best_params)
print("##############################################################")
print(study.best_value)
print("##############################################################")

optuna.visualization.plot_optimization_history(study)
