import os
import yaml
import copy

# Read in environment configurations
env_configs = {}
with open("gym_commonroad/configs.yaml","r") as config_file:
    env_configs = yaml.safe_load(config_file)["env_configs"]
env_configs["reward_type"] = "hybrid_reward"
log_path = "tutorials/logs/"
os.makedirs(log_path, exist_ok=True)
with open(os.path.join(log_path, "environment_configurations.yml"), "w") as config_file:
    yaml.dump(env_configs, config_file)
hyperparams = {}
with open("hyperparams/ppo2.yml","r") as hyperparam_file:
    hyperparams = yaml.safe_load(hyperparam_file)["commonroad-v1"]
with open(os.path.join(log_path, "model_hyperparameters.yml"), "w") as hyperparam_file:
    yaml.dump(hyperparams, hyperparam_file)
if "normalize" in hyperparams:
    del hyperparams["normalize"]
# import gym
import gymnasium as gym
from stable_baselines3.common.monitor import Monitor

# try:
#     gym.envs.register(
#         id="commonroad-v1",
#         entry_point="commonroad_rl.gym_commonroad.commonroad_env:CommonroadEnv",
#         kwargs=None,
#     )
#     print("[gym_commonroad/__init__.py] Registered commonroad-v1")
# except gym.error.Error:
#     print("[gym_commonroad/__init__.py] Error occurs while registering commonroad-v1")
#     pass

gym.envs.registration.register(
    id="commonroad-v1-safe",
    entry_point="commonroad_rl.gym_commonroad.safe_commonroad_env:SafetyLayer",
)

import gymnasium as gym
# from stable_baselines.bench import Monitor
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env  import DummyVecEnv, VecNormalize

#print(gym.envs.registry['commonroad-v1'])

# Create a Gym-based RL environment with specified data paths and environment configurations
meta_scenario_path = "tutorials/data/inD-dataset-v1.0/pickles/meta_scenario"
training_data_path = "tutorials/data/inD-dataset-v1.0/pickles/problem_train"
training_env = gym.make("commonroad-v1-safe",
                        meta_scenario_path=meta_scenario_path,
                        train_reset_config_path= training_data_path,
                        **env_configs)

# Wrap the environment with a monitor to keep an record of the learning process
info_keywords=tuple(["is_collision", \
                     "is_time_out", \
                     "is_off_road", \
                     "is_friction_violation", \
                     "is_goal_reached"])
training_env = Monitor(training_env, log_path + "0", info_keywords=info_keywords)

# Vectorize the environment with a callable argument
def make_training_env():
    return training_env
training_env = DummyVecEnv([make_training_env])

training_env = VecNormalize(training_env, norm_obs=True, norm_reward=True)

from stable_baselines3.common.callbacks import BaseCallback, EvalCallback

env_configs_test = copy.deepcopy(env_configs)
env_configs_test["test_env"] = True

testing_data_path = "tutorials/data/inD-dataset-v1.0/pickles/problem_test"
testing_env = gym.make("commonroad-v1-safe",
                       meta_scenario_path=meta_scenario_path,
                       test_reset_config_path=testing_data_path,
                       **env_configs_test)

log_path_test = "tutorials/logs/test"
os.makedirs(log_path_test, exist_ok=True)

testing_env = Monitor(testing_env, log_path_test + "/0", info_keywords=info_keywords)

def make_testing_env():
    return testing_env

testing_env = DummyVecEnv([make_testing_env])

testing_env = VecNormalize(testing_env, norm_obs=True, norm_reward=False, training=False)

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
        # # Return True added to always continue training
        # return True

save_vec_normalize_callback = SaveVecNormalizeCallback(save_path=log_path)
eval_callback = EvalCallback(testing_env,
                             log_path=log_path,
                             eval_freq=500,
                             n_eval_episodes=15,
                             callback_on_new_best=save_vec_normalize_callback,
                             verbose=1)
from stable_baselines3 import PPO
model = PPO(env=training_env, **hyperparams)
model.learn(
    total_timesteps=100000,
    callback=eval_callback
)

model.save("tutorials/logs/intermediate_model")

