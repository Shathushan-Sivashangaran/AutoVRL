# Copyright 2023 Shathushan Sivashangaran

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Authors: Shathushan Sivashangaran, Apoorva Khairnar

from stable_baselines3.common.env_checker import check_env
from stable_baselines3 import SAC, PPO
from stable_baselines3.common.policies import MultiInputActorCriticPolicy
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.env_util import make_vec_env
import gym
import Env
import matplotlib.pyplot as plt
import numpy as np

env = gym.make("X10Car-v1") # v1, v2, v3, v4 and v5 correspond to the 20m x 20m outdoor, 50m x 50m outdoor, 20m x 20m urban, 50m x 50m urban and oval racetrack environments
check_env(env, warn=True)

#model = SAC('MlpPolicy', env) # initialize new, untrained model

model = SAC.load("AutoVRL_policy", env = env) # load trained policy

obs = env.reset()

episode = 0
agent_steps = 0
episode_reward = 0
max_steps = 100000000 # maximum steps to evaluate

n_steps = max_steps
for step in range(n_steps):

  action, _ = model.predict(obs, deterministic=False) # select action using the trained model
  print("Step {}".format(step + 1))
  print("Action: ", action)
  obs, reward, done, info = env.step(action) # perform action to obtain new observation and reward
  agent_steps = agent_steps + 1
  episode_reward = episode_reward + reward
  print('obs=', obs, 'reward=', reward, 'done=', done)
  env.render()
  
  if done:
    episode = episode + 1
    episode_reward = episode_reward + reward
    obs = env.reset()
    print("Obstacle Hit,", "Episode =", episode,",", "Agent Steps =", agent_steps,",", "Episode Reward =", episode_reward)
    agent_steps = 0
    episode_reward = 0