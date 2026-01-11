import gymnasium as gym
import commonroad_rl.gym_commonroad
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback, CallbackList, BaseCallback
from tqdm import trange
import os

import warnings
warnings.filterwarnings("ignore")

# kwargs overwrites configs defined in commonroad_rl/gym_commonroad/configs.yaml
# env = gym.make("commonroad-v1",
# 		        action_configs={"action_type": "continuous"},
#                 goal_configs={"observe_distance_goal_long": True, "observe_distance_goal_lat": True},
#                 surrounding_configs={"observe_lane_circ_surrounding": True,
#                		                "fast_distance_calculation": False,
#                                     "observe_lidar_circle_surrounding": True,
#                                     "lidar_circle_num_beams": 20})

meta_scenario_path = "/home/shuaiyi/Documents/Lab/commonroad/commonroad-rl/pickles/meta_scenario"
training_data_path = "/home/shuaiyi/Documents/Lab/commonroad/commonroad-rl/pickles/problem_train"
testing_data_path = "/home/shuaiyi/Documents/Lab/commonroad/commonroad-rl/pickles/problem_test"
env = gym.make("commonroad-v1", 
                meta_scenario_path=meta_scenario_path,
                train_reset_config_path=training_data_path)

observation = env.reset()

if os.path.exists("./ppo_best_model_simple/"):
    print("Loading existing model")
    model = PPO.load("./ppo_commonroad_policy.zip", env=env, verbose=1, tensorboard_log="./ppo_commonroad_tensorboard/")
else:
    model = PPO("MlpPolicy", env, verbose=1, tensorboard_log="./ppo_commonroad_tensorboard/")


eval_env = gym.make("commonroad-v1", 
                meta_scenario_path=meta_scenario_path,
                train_reset_config_path=testing_data_path)


class TqdmEvalCallback(BaseCallback):
    def __init__(self, eval_env, eval_freq=1000, n_eval_episodes=5, save_path=None, verbose=1):
        super().__init__(verbose)
        self.eval_env = eval_env
        self.eval_freq = eval_freq
        self.n_eval_episodes = n_eval_episodes
        self.save_path = save_path
        self.best_mean_reward = -130.0
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self) -> bool:
        if self.n_calls % self.eval_freq == 0:
            rewards = []
            lengths = []
            for _ in range(self.n_eval_episodes):
                obs, _ = self.eval_env.reset()
                done = False
                total_reward = 0.0
                ep_len = 0
                while not done:
                    action, _ = self.model.predict(obs, deterministic=True)
                    obs, reward, terminated, truncated, _ = self.eval_env.step(action)
                    done = terminated or truncated
                    total_reward += reward
                    ep_len += 1
                rewards.append(total_reward)
                lengths.append(ep_len)
            mean_reward = sum(rewards) / len(rewards)
            mean_length = sum(lengths) / len(lengths)
            print(f"Eval at step {self.num_timesteps}: "
                  f"Mean reward = {mean_reward:.2f}, Mean length = {mean_length:.2f}")
            if self.save_path is not None and mean_reward > self.best_mean_reward:
                self.best_mean_reward = mean_reward
                model_path = os.path.join(self.save_path, "best_model"+str(int(mean_reward)))
                self.model.save(model_path)
                if self.verbose:
                    print(f"New best model saved with reward {mean_reward:.2f} at step {self.num_timesteps}")
        return True

eval_callback = TqdmEvalCallback(eval_env=eval_env, eval_freq=1000, save_path="./ppo_best_model_simple/", n_eval_episodes=10)

callback = CallbackList([eval_callback])

# 训练
model.learn(total_timesteps=100000, callback=callback, progress_bar=True)

# 保存模型
model.save("ppo_commonroad_policy")

# 关闭环境
env.close()