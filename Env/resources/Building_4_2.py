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

import pybullet as p
import os
import math


class building_4_2:
    def __init__(self, client):
        f_name = os.path.join(os.getcwd(), 'Env/resources/building_4_2.urdf')
        self.building_4_2 = p.loadURDF(fileName=f_name,
                   basePosition=[8.6, -12, 0],
                   physicsClientId=client, useFixedBase = True)
    
    def get_ids(self):
        return self.building_4_2