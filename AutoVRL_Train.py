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
  
env = gym.make('X10Car-v1') # v1, v2, v3, v4 and v5 correspond to the 20m x 20m outdoor, 50m x 50m outdoor, 20m x 20m urban, 50m x 50m urban and oval racetrack environments
check_env(env, warn=True)

## To evaluate a new reward function, modify step() in the gym environment file

training_steps = 1000000 # number of training steps

model = SAC('MlpPolicy', env) # initialize new model. SAC can be replaced with any RL algorithm imported from SB3. Use 'CnnPolicy' for image observations

## To continue training a saved model, load the trained policy and replay buffer if using an off-policy algorithm

#model = SAC.load('AutoVRL_policy', env = env) # load previously trained model
#model.load_replay_buffer('AutoVRL_policy_replay_buffer') # load replay buffer of previously trained model

model.learn(total_timesteps=training_steps, progress_bar=True) # train the agent

model.save('AutoVRL_policy') # save the trained policy
model.save_replay_buffer('AutoVRL_policy_replay_buffer') # save the replay buffer for off-policy algorithms
