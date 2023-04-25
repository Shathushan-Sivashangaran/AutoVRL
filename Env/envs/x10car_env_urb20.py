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
from Env.resources.Walls_20 import walls_20
from Env.resources.Road_20 import road_20

from Env.resources.Building_1 import building_1
from Env.resources.Building_2 import building_2
from Env.resources.Building_3 import building_3
from Env.resources.Building_4 import building_4
from Env.resources.Building_5 import building_5

from Env.resources.Vehicle_1 import vehicle_1
from Env.resources.Vehicle_2 import vehicle_2

class X10Car_Env_urb20(gym.Env):
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
        self.building_1 = None
        self.building_2 = None
        self.building_3 = None
        self.building_4 = None
        self.building_5 = None
        self.vehicle_1 = None
        self.vehicle_2 = None
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
        
        if (min_dist_to_wall > 2.0 and min_dist_to_wall < 2.5):
            reward = reward + 2.0
            
        if (min_dist_to_wall < 1.0):
            reward = reward - 50.0            
        
        self.writeReward = pd.concat([self.writeReward, pd.DataFrame({'Reward':[reward]})], ignore_index=True)
        self.episode_reward = self.episode_reward + reward
        
        # Record reward at each step every 10,000,000 steps. Modify number for higher data recording frequency. Training may be slower with higher writing frequency

        if(self.store_agent_steps == 10000000):
            self.writeReward.to_csv('reward_urb20.csv', index = False)
            self.store_agent_steps = 0
        
        # Terminate episode and record episode reward

        if (min_dist_to_wall < 0.6):
            self.store_reward = 0
            self.done = True
            self.writeEpisodeReward = pd.concat([self.writeEpisodeReward, pd.DataFrame({'Episode Reward':[self.episode_reward]})], ignore_index=True)
            self.writeEpisodeReward.to_csv('episode_reward_urb20.csv', index = False)
            self.episode_reward = 0

        self.store_reward = self.store_reward + reward       

        # Record trajectory position data if agent achieves 5000 steps. Modify number to store trajectories for alternate target steps

        if (self.store_agent_steps <= 5000):
            self.trajectory = pd.concat([self.trajectory, pd.DataFrame({'Agent Steps':[self.store_agent_steps], 'Car Observation X':[car_ob[0]], 'Car Observation Y': [car_ob[1]],'Minimum Distance':[min_dist_to_wall], 'Reward':[self.store_reward]})], ignore_index=True)
            if (self.store_agent_steps == 5000):
                self.trajectory.to_csv('trajectory_urb20.csv', index = False)
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
        self.plane = road_20(self.client)
        self.car = x10car_v0(self.client)
        self.wall = walls_20(self.client)
        self.building_1 = building_1(self.client)
        self.building_2 = building_2(self.client)
        self.building_3 = building_3(self.client)
        self.building_4 = building_4(self.client)
        self.building_5 = building_5(self.client)
        self.vehicle_1 = vehicle_1(self.client)
        self.vehicle_2 = vehicle_2(self.client)

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
        view_matrix = p.computeViewMatrix([pos[0],pos[1],10], pos + camera_vec, up_vec)

        # Display image
        frame = p.getCameraImage(400, 400, view_matrix, proj_matrix)[2]
        frame = np.reshape(frame, (400, 400, 4))
        self.rendered_img.set_data(frame)
        plt.draw()
        plt.pause(.00001)

    def close(self):
        p.disconnect(self.client)