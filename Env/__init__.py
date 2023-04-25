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

from gym.envs.registration import register

register(
    id='X10Car-v1', 
    entry_point='Env.envs:X10Car_Env_out20'
)

register(
    id='X10Car-v2', 
    entry_point='Env.envs:X10Car_Env_out50'
)

register(
    id='X10Car-v3', 
    entry_point='Env.envs:X10Car_Env_urb20'
)

register(
    id='X10Car-v4', 
    entry_point='Env.envs:X10Car_Env_urb50'
)

register(
    id='X10Car-v5', 
    entry_point='Env.envs:X10Car_Env_raceoval'
)
