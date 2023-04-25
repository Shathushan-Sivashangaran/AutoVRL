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

import gym
import numpy as np
import math
import pybullet as p
import matplotlib.pyplot as plt
import pandas as pd

from Env.resources.x10car_v0 import x10car_v0
from Env.resources.Walls_50 import walls_50
from Env.resources.Ground_50 import ground_50

from Env.resources.Rock_1_1 import rock_1_1
from Env.resources.Rock_2_1 import rock_2_1
from Env.resources.Rock_3_1 import rock_3_1
from Env.resources.Rock_4_1 import rock_4_1
from Env.resources.Rock_5_1 import rock_5_1
from Env.resources.Rock_6_1 import rock_6_1
from Env.resources.Rock_7_1 import rock_7_1
from Env.resources.Rock_8_1 import rock_8_1
from Env.resources.Rock_9_1 import rock_9_1
from Env.resources.Rock_10_1 import rock_10_1
from Env.resources.Rock_11_1 import rock_11_1
from Env.resources.Rock_12_1 import rock_12_1
from Env.resources.Rock_13_1 import rock_13_1
from Env.resources.Rock_14_1 import rock_14_1
from Env.resources.Rock_15_1 import rock_15_1
from Env.resources.Rock_16_1 import rock_16_1
from Env.resources.Rock_29_1 import rock_29_1
from Env.resources.Rock_30_1 import rock_30_1
from Env.resources.Rock_31_1 import rock_31_1
from Env.resources.Rock_32_1 import rock_32_1
from Env.resources.Rock_33_1 import rock_33_1
from Env.resources.Rock_34_1 import rock_34_1
from Env.resources.Rock_35_1 import rock_35_1

from Env.resources.Tree_1_1 import tree_1_1
from Env.resources.Tree_2_1 import tree_2_1
from Env.resources.Tree_3_1 import tree_3_1
from Env.resources.Tree_4_1 import tree_4_1
from Env.resources.Tree_5_1 import tree_5_1
from Env.resources.Tree_6_1 import tree_6_1
from Env.resources.Tree_7_1 import tree_7_1
from Env.resources.Tree_8_1 import tree_8_1
from Env.resources.Tree_9_1 import tree_9_1
from Env.resources.Tree_10_1 import tree_10_1
from Env.resources.Tree_11_1 import tree_11_1
from Env.resources.Tree_12_1 import tree_12_1
from Env.resources.Tree_13_1 import tree_13_1
from Env.resources.Tree_14_1 import tree_14_1
from Env.resources.Tree_15_1 import tree_15_1
from Env.resources.Tree_16_1 import tree_16_1
from Env.resources.Tree_17_1 import tree_17_1
from Env.resources.Tree_34_1 import tree_34_1
from Env.resources.Tree_35_1 import tree_35_1
from Env.resources.Tree_36_1 import tree_36_1
from Env.resources.Tree_37_1 import tree_37_1
from Env.resources.Tree_38_1 import tree_38_1
from Env.resources.Tree_39_1 import tree_39_1
from Env.resources.Tree_40_1 import tree_40_1
from Env.resources.Tree_41_1 import tree_41_1

from Env.resources.Rock_1_2 import rock_1_2
from Env.resources.Rock_2_2 import rock_2_2
from Env.resources.Rock_3_2 import rock_3_2
from Env.resources.Rock_4_2 import rock_4_2
from Env.resources.Rock_5_2 import rock_5_2
from Env.resources.Rock_6_2 import rock_6_2
from Env.resources.Rock_7_2 import rock_7_2
from Env.resources.Rock_8_2 import rock_8_2
from Env.resources.Rock_9_2 import rock_9_2
from Env.resources.Rock_10_2 import rock_10_2
from Env.resources.Rock_11_2 import rock_11_2
from Env.resources.Rock_12_2 import rock_12_2
from Env.resources.Rock_13_2 import rock_13_2
from Env.resources.Rock_14_2 import rock_14_2
from Env.resources.Rock_15_2 import rock_15_2
from Env.resources.Rock_16_2 import rock_16_2
from Env.resources.Rock_17_2 import rock_17_2
from Env.resources.Rock_29_2 import rock_29_2
from Env.resources.Rock_30_2 import rock_30_2
from Env.resources.Rock_31_2 import rock_31_2
from Env.resources.Rock_32_2 import rock_32_2
from Env.resources.Rock_33_2 import rock_33_2
from Env.resources.Rock_34_2 import rock_34_2
from Env.resources.Rock_35_2 import rock_35_2
from Env.resources.Rock_36_2 import rock_36_2

class X10Car_Env_out50(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        self.action_space = gym.spaces.Box(
            low=np.array([-1, -1], dtype=np.float32),
            high=np.array([1, 1], dtype=np.float32))

        distlow = np.zeros(100)
        disthigh = [1000]*100

        # Observation space for 90 deg FoV lidar with 100 points. For alternate FoV and number of points modify get_distance() in the x10car file
        
        self.observation_space = gym.spaces.Box(
            low=np.array(distlow, dtype=np.float32),
            high=np.array(disthigh, dtype=np.float32))
        self.np_random, _ = gym.utils.seeding.np_random()

        self.client = p.connect(p.DIRECT)#, options='--opengl2')
        # Reduce length of episodes for RL algorithms
        p.setTimeStep(1/30, self.client)

        self.car = None
        self.wall = None
        self.plane = None


        self.rock_1_1=None; self.rock_2_1=None; self.rock_3_1=None; self.rock_4_1=None; self.rock_5_1=None; self.rock_6_1=None
        self.rock_7_1=None; self.rock_8_1=None; self.rock_9_1=None; self.rock_10_1=None; self.rock_11_1=None; self.rock_12_1=None
        self.rock_13_1=None; self.rock_14_1=None; self.rock_15_1=None; self.rock_16_1=None; self.rock_17_1=None; self.rock_18_1=None
        self.rock_19_1=None; self.rock_20_1=None; self.rock_21_1=None; self.rock_22_1=None; self.rock_23_1=None; self.rock_24_1=None
        self.rock_25_1=None; self.rock_26_1=None; self.rock_27_1=None; self.rock_28_1=None; self.rock_29_1=None; self.rock_30_1=None
        self.rock_31_1=None; self.rock_32_1=None; self.rock_33_1=None; self.rock_34_1=None; self.rock_35_1=None

        self.tree_1=None; self.tree_2=None; self.tree_3=None; self.tree_4=None; self.tree_5=None; self.tree_6=None; self.tree_7=None
        self.tree_8=None; self.tree_9=None; self.tree_10=None; self.tree_11=None; self.tree_12=None; self.tree_13=None; self.tree_14=None
        self.tree_15=None; self.tree_16=None; self.tree_17=None; self.tree_34=None; self.tree_35=None; self.tree_36=None; self.tree_37=None
        self.tree_38=None; self.tree_39=None; self.tree_40=None; self.tree_41=None

        self.rock_1_2=None; self.rock_2_2=None; self.rock_3_2=None; self.rock_4_2=None; self.rock_5_2=None; self.rock_6_2=None
        self.rock_7_2=None; self.rock_8_2=None; self.rock_9_2=None; self.rock_10_2=None; self.rock_11_2=None; self.rock_12_2=None
        self.rock_13_2=None; self.rock_14_2=None; self.rock_15_2=None; self.rock_16_2=None; self.rock_17_2=None; self.rock_18_2=None
        self.rock_19_2=None; self.rock_20_2=None; self.rock_21_2=None; self.rock_22_2=None; self.rock_23_2=None; self.rock_24_2=None
        self.rock_25_2=None; self.rock_26_2=None; self.rock_27_2=None; self.rock_28_2=None; self.rock_29_2=None; self.rock_30_2=None
        self.rock_31_2=None; self.rock_32_2=None; self.rock_33_2=None; self.rock_34_2=None; self.rock_35_2=None; self.rock_36_2=None

        self.done = False
        self.prev_dist_to_wall = None
        self.rendered_img = None
        self.render_rot_matrix = None
        self.trajectory = pd.DataFrame({'Agent Steps':[], 'Car Observation X':[], 'Car Observation Y':[], 'Minimum Distance':[], 'Reward':[]})
        self.writeReward = pd.DataFrame({'Reward':[]})
        self.store_agent_steps = 0
        self.store_reward = 0
        self.episode_reward = 0
        self.writeEpisodeReward = pd.DataFrame({'Episode Reward':[]})
        self.reset()

    def step(self, action):

        self.car.apply_action(action) # perform action
        p.stepSimulation()
        car_ob = self.car.get_observation() # return state

        dist_to_wall = self.car.get_distance()
        min_dist_to_wall = min(dist_to_wall) 
        #print(min_dist_to_wall)

        self.store_agent_steps = self.store_agent_steps + 1
        self.episode_reward = self.episode_reward
        
        # Compute reward 

        reward = 0.005 * (min_dist_to_wall**2) + 5.0*((min(max(action[0], 0), 1))**2) - 2.0*((max(min(action[1], 0.36), -0.36))**2)
        
        if ((min(max(action[0], 0), 1)) < 0.1):
            reward = reward - 50.0  

        if (min_dist_to_wall > 2.0 and min_dist_to_wall < 2.5):
            reward = reward + 2.0
            
        if (min_dist_to_wall < 1.0):
            reward = reward - 50.0            
        
        self.writeReward = pd.concat([self.writeReward, pd.DataFrame({'Reward':[reward]})], ignore_index=True)
        self.episode_reward = self.episode_reward + reward
        
        # Record reward at each step every 10,000,000 steps. Modify number for higher data recording frequency. Training may be slower with higher writing frequency

        if(self.store_agent_steps == 10000000):
            self.writeReward.to_csv('reward_out50.csv', index = False)
            self.store_agent_steps = 0
        
        # Terminate episode and record episode reward

        if (min_dist_to_wall < 0.6):
            self.store_reward = 0
            self.done = True
            self.writeEpisodeReward = pd.concat([self.writeEpisodeReward, pd.DataFrame({'Episode Reward':[self.episode_reward]})], ignore_index=True)
            self.writeEpisodeReward.to_csv('episode_reward_out50.csv', index = False)
            self.episode_reward = 0


        self.store_reward = self.store_reward + reward     

        # Record trajectory position data if agent achieves 5000 steps. Modify number to store trajectories for alternate target steps

        if (self.store_agent_steps <= 5000):
            self.trajectory = pd.concat([self.trajectory, pd.DataFrame({'Agent Steps':[self.store_agent_steps], 'Car Observation X':[car_ob[0]], 'Car Observation Y': [car_ob[1]],'Minimum Distance':[min_dist_to_wall], 'Reward':[self.store_reward]})], ignore_index=True)
            if (self.store_agent_steps == 5000):
                self.trajectory.to_csv('trajectory_out50.csv', index = False)
                print(f'stored trajectory for reward {reward} and agent_steps {self.store_agent_steps}')
                self.store_agent_steps = 0       
        
        ob = np.array(dist_to_wall, dtype=np.float32)
        return ob, reward, self.done, dict()

    def seed(self, seed=None):
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        return [seed]

    def reset(self):
        p.resetSimulation(self.client)
        p.setGravity(0, 0, -9.81)
        # Reload environment assets
        self.plane = ground_50(self.client)
        self.car = x10car_v0(self.client)
        self.wall = walls_50(self.client)
        
        self.rock_1_1=rock_1_1(self.client); self.rock_2_1=rock_2_1(self.client); self.rock_3_1=rock_3_1(self.client); self.rock_4_1=rock_4_1(self.client); self.rock_5_1=rock_5_1(self.client); self.rock_6_1=rock_6_1(self.client)
        self.rock_7_1=rock_7_1(self.client); self.rock_8_1=rock_8_1(self.client); self.rock_9_1=rock_9_1(self.client); self.rock_10_1=rock_10_1(self.client); self.rock_11_1=rock_11_1(self.client); self.rock_12_1=rock_12_1(self.client)
        self.rock_13_1=rock_13_1(self.client); self.rock_14_1=rock_14_1(self.client); self.rock_15_1=rock_15_1(self.client); self.rock_16_1=rock_16_1(self.client); self.rock_29_1=rock_29_1(self.client); self.rock_30_1=rock_30_1(self.client)
        self.rock_31_1=rock_31_1(self.client); self.rock_32_1=rock_32_1(self.client); self.rock_33_1=rock_33_1(self.client); self.rock_34_1=rock_34_1(self.client); self.rock_35_1=rock_35_1(self.client)

        self.rock_1_2=rock_1_2(self.client); self.rock_2_2=rock_2_2(self.client); self.rock_3_2=rock_3_2(self.client); self.rock_4_2=rock_4_2(self.client); self.rock_5_2=rock_5_2(self.client); self.rock_6_2=rock_6_2(self.client)
        self.rock_7_2=rock_7_2(self.client); self.rock_8_2=rock_8_2(self.client); self.rock_9_2=rock_9_2(self.client); self.rock_10_2=rock_10_2(self.client); self.rock_11_2=rock_11_2(self.client); self.rock_12_2=rock_12_2(self.client)
        self.rock_13_2=rock_13_2(self.client); self.rock_14_2=rock_14_2(self.client); self.rock_15_2=rock_15_2(self.client); self.rock_16_2=rock_16_2(self.client); self.rock_17_2=rock_17_2(self.client); self.rock_29_2=rock_29_2(self.client)
        self.rock_30_2=rock_30_2(self.client); self.rock_31_2=rock_31_2(self.client); self.rock_32_2=rock_32_2(self.client); self.rock_33_2=rock_33_2(self.client); self.rock_34_2=rock_34_2(self.client); self.rock_35_2=rock_35_2(self.client)
        self.rock_36_2=rock_36_2(self.client)
        
        self.tree_1=tree_1_1(self.client); self.tree_2=tree_2_1(self.client); self.tree_3=tree_3_1(self.client); self.tree_4=tree_4_1(self.client); self.tree_5=tree_5_1(self.client); self.tree_6=tree_6_1(self.client); self.tree_7=tree_7_1(self.client)
        self.tree_8=tree_8_1(self.client); self.tree_9=tree_9_1(self.client); self.tree_10=tree_10_1(self.client); self.tree_11=tree_11_1(self.client); self.tree_12=tree_12_1(self.client); self.tree_13=tree_13_1(self.client); self.tree_14=tree_14_1(self.client)
        self.tree_15=tree_15_1(self.client); self.tree_16=tree_16_1(self.client); self.tree_17=tree_17_1(self.client); self.tree_34=tree_34_1(self.client); self.tree_35=tree_35_1(self.client); self.tree_36=tree_36_1(self.client)
        self.tree_37=tree_37_1(self.client); self.tree_38=tree_38_1(self.client); self.tree_39=tree_39_1(self.client); self.tree_40=tree_40_1(self.client); self.tree_41=tree_41_1(self.client)

        self.done = False

        self.prev_dist_to_wall = self.car.get_distance() 
        
        return np.array(self.prev_dist_to_wall, dtype=np.float32)

    def render(self, mode='human'):
        if self.rendered_img is None:
            self.rendered_img = plt.imshow(np.zeros((100, 100, 4)))

        # Base information
        car_id, client_id = self.car.get_ids()
        wall_id = self.wall.get_ids()

        proj_matrix = p.computeProjectionMatrixFOV(fov=80, aspect=1,
                                                   nearVal=0.01, farVal=100)
        pos, ori = [list(l) for l in
                    p.getBasePositionAndOrientation(car_id, client_id)]
        pos[2] = 0.2

        # Rotate camera direction
        rot_mat = np.array(p.getMatrixFromQuaternion(ori)).reshape(3, 3)
        camera_vec = np.matmul(rot_mat, [1, 0, 0])
        up_vec = np.matmul(rot_mat, np.array([0, 0, 1]))
        view_matrix = p.computeViewMatrix([pos[0], pos[1], pos[2]+7.5], pos + camera_vec, up_vec)

        # Display image
        frame = p.getCameraImage(400, 400, view_matrix, proj_matrix)[2]
        frame = np.reshape(frame, (400, 400, 4))
        self.rendered_img.set_data(frame)
        plt.draw()
        plt.pause(.00001)

    def close(self):
        p.disconnect(self.client)