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

from Env.resources.x10car_v0_racetrack_oval import x10car_v0
from Env.resources.Racetrack_oval import racetrack_oval
from Env.resources.Plane_racetrack_oval import plane_racetrack_oval

class X10Car_Env_raceoval(gym.Env):
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

        self.client = p.connect(p.DIRECT)
        # Reduce length of episodes for RL algorithms
        p.setTimeStep(1/30, self.client)

        self.car = None
        self.wall = None
        self.plane = None
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

        reward = 5.0*((min(max(action[0], 0), 1))**2) - 2.0*((max(min(action[1], 0.36), -0.36))**2)
        
        self.writeReward = pd.concat([self.writeReward, pd.DataFrame({'Reward':[reward]})], ignore_index=True)
        self.episode_reward = self.episode_reward + reward
        
        # Record reward at each step every 10,000,000 steps. Modify number for higher data recording frequency. Training may be slower with higher writing frequency

        if(self.store_agent_steps == 10000000):
            self.writeReward.to_csv('reward_raceoval.csv', index = False)
            self.store_agent_steps = 0
        
        # Terminate episode and record episode reward

        if (min_dist_to_wall < 0.6):
            self.store_reward = 0
            self.done = True
            self.writeEpisodeReward = pd.concat([self.writeEpisodeReward, pd.DataFrame({'Episode Reward':[self.episode_reward]})], ignore_index=True)
            self.writeEpisodeReward.to_csv('episode_reward_raceoval.csv', index = False)
            self.episode_reward = 0

        self.store_reward = self.store_reward + reward            

        # Record trajectory position data if agent achieves 5000 steps. Modify number to store trajectories for alternate target steps

        if (self.store_agent_steps <= 5000):
            self.trajectory = pd.concat([self.trajectory, pd.DataFrame({'Agent Steps':[self.store_agent_steps], 'Car Observation X':[car_ob[0]], 'Car Observation Y': [car_ob[1]],'Minimum Distance':[min_dist_to_wall], 'Reward':[self.store_reward]})], ignore_index=True)
            if (self.store_agent_steps == 5000):
                self.trajectory.to_csv('trajectory_raceoval.csv', index = False)
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
        self.plane = plane_racetrack_oval(self.client)
        self.car = x10car_v0(self.client)
        self.wall = racetrack_oval(self.client)

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
        #view_matrix = p.computeViewMatrix([8, 2, 2], pos + camera_vec, up_vec)
        view_matrix = p.computeViewMatrix([0, 0, 20], pos + camera_vec, up_vec)

        # Display image
        frame = p.getCameraImage(400, 400, view_matrix, proj_matrix)[2]
        frame = np.reshape(frame, (400, 400, 4))
        self.rendered_img.set_data(frame)
        plt.draw()
        plt.pause(.00001)

    def close(self):
        p.disconnect(self.client)