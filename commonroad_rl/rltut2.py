import os
import yaml
import copy

log_path = "tutorials/logs/"

# Read in environment configurations
env_configs = {}
with open(os.path.join(log_path, "environment_configurations.yml"), "r") as config_file:
    env_configs = yaml.safe_load(config_file)

# Read in model hyperparameters
hyperparams = {}
with open(os.path.join(log_path, "model_hyperparameters.yml"), "r") as hyperparam_file:
    hyperparams = yaml.safe_load(hyperparam_file)

# Deduce `policy` from the pretrained model
if "policy" in hyperparams:
    del hyperparams["policy"]

# Remove `normalize` as it will be handled explicitly later
if "normalize" in hyperparams:
    del hyperparams["normalize"]

import gymnasium as gym
gym.envs.registration.register(
    id="commonroad-v1-safe",
    entry_point="commonroad_rl.gym_commonroad.safe_commonroad_env:SafetyLayer",
)


from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env  import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback

# Create a Gym-based RL environment with specified data paths and environment configurations
meta_scenario_path = "tutorials/data/highD/pickles/meta_scenario"
training_data_path = "tutorials/data/highD/pickles/problem_train"
training_env = gym.make("commonroad-v1-safe",
                        meta_scenario_path=meta_scenario_path,
                        train_reset_config_path=training_data_path,
                        **env_configs)

# Wrap the environment with a monitor to keep an record of the learning process
info_keywords = tuple(["is_collision", \
                       "is_time_out", \
                       "is_off_road", \
                       "is_friction_violation", \
                       "is_goal_reached"])
training_env = Monitor(training_env, log_path + "0", info_keywords=info_keywords)

# Vectorize the environment with a callable argument
def make_training_env():
    return training_env


training_env = DummyVecEnv([make_training_env])

# Append the additional key for the testing environment
env_configs_test = copy.deepcopy(env_configs)
env_configs_test["test_env"] = True

# Create the testing environment
testing_data_path = "tutorials/data/highD/pickles/problem_test"
testing_env = gym.make("commonroad-v1-safe",
                       meta_scenario_path=meta_scenario_path,
                       test_reset_config_path=testing_data_path,
                       **env_configs_test)

# Wrap the environment with a monitor to keep an record of the testing episodes
log_path_test = "tutorials/logs/test"
os.makedirs(log_path_test, exist_ok=True)

testing_env = Monitor(testing_env, log_path_test + "/0", info_keywords=info_keywords)


# Vectorize the environment with a callable argument
def make_testing_env():
    return testing_env


testing_env = DummyVecEnv([make_testing_env])

# Normalize only observations during testing
testing_env = VecNormalize(testing_env, norm_obs=True, norm_reward=False, training=False)


# Define a customized callback function to save the vectorized and normalized training environment
class SaveVecNormalizeCallback(BaseCallback):
    def __init__(self, save_path: str, verbose=1):
        super(SaveVecNormalizeCallback, self).__init__(verbose)
        self.save_path = save_path

    def _init_callback(self) -> None:
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self) -> bool:
        save_path_name = os.path.join(self.save_path, "vecnormalize.pkl")
        self.model.get_vec_normalize_env().save(save_path_name)
        print("Saved vectorized and normalized environment to {}".format(save_path_name))


# Pass the testing environment and customized saving callback to an evaluation callback
# Note that the evaluation callback will triggers three evaluating episodes after every 500 training steps
save_vec_normalize_callback = SaveVecNormalizeCallback(save_path=log_path)
eval_callback = EvalCallback(testing_env,
                             log_path=log_path,
                             eval_freq=500,
                             n_eval_episodes=3,
                             callback_on_new_best=save_vec_normalize_callback)

from stable_baselines3.common.vec_env import VecNormalize
from stable_baselines3 import PPO

# Load saved environment
training_env = VecNormalize.load("tutorials/logs/vecnormalize.pkl", training_env)

# Load pretrained model
model_continual = PPO.load("tutorials/logs/intermediate_model", env=training_env, **hyperparams)

# Set learning steps and trigger learning with the evaluation callback
n_timesteps=5000
model_continual.learn(n_timesteps, eval_callback)

# Save the continual-learning model
# Note that we use the name "best_model" here as it will be fetched in the next tutorials
model_continual.save("tutorials/logs/best_model")
model_continual.get_vec_normalize_env().save("tutorials/logs/vecnormalize.pkl")




