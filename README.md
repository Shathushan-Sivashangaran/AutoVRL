# AutoVRL

AutoVRL (AUTOnomous ground Vehicle deep Reinforcement Learning simulator) is an open-source high fidelity simulator for simulation to real-world Autonomous Ground Vehicle (AGV) Deep Reinforcement Learning (DRL) research and development built upon the Bullet physics engine utilizing OpenAI Gym and Stable Baselines3 in PyTorch. AutoVRL is equipped with sensor implementations of GPS, IMU, LiDAR and camera, actuators for AGV control, and realistic environments, with extensibility for new environments and AGV models. The simulator provides access to state-of-the-art DRL algorithms, utilizing a python interface for simple algorithm and environment customization, and simulation execution [1].


## AGV and Environment Models

AutoVRL includes a digital twin of [XTENTH-CAR](https://github.com/Shathushan-Sivashangaran/XTENTH-CAR), a proportionally 1/10th scaled Ackermann steered AGV for connected autonomy and all terrain research developed with best-in-class embedded processing to facilitate real-world AGV DRL research.

<p align="center">
<img src="https://github.com/Shathushan-Sivashangaran/AutoVRL/blob/main/images/xtenthcar.jpg" width="200" height="190">
<img src="https://github.com/Shathushan-Sivashangaran/AutoVRL/blob/main/images/xtenthcar_digitaltwin.JPG" width="200">
</p>

Five environments are included for training and evaluation. These comprise 20m x 20m and 50m x 50m outdoor and urban environments, and an oval race track. The environments contain realistic objects that include trees and boulders in the outdoor map, and buildings and passenger vehicles in the urban scenario.

<p align="center">
<img src="https://github.com/Shathushan-Sivashangaran/AutoVRL/blob/main/images/env_outdoor_20.png" width="200">
<img src="https://github.com/Shathushan-Sivashangaran/AutoVRL/blob/main/images/env_outdoor_50.png" width="200" height="180">
<img src="https://github.com/Shathushan-Sivashangaran/AutoVRL/blob/main/images/env_urban_20.png" width="200" height="180">
<img src="https://github.com/Shathushan-Sivashangaran/AutoVRL/blob/main/images/env_urban_50.png" width="200" height="180">
<img src="https://github.com/Shathushan-Sivashangaran/AutoVRL/blob/main/images/env_racetrack_oval.png" width="200" height="180">
</p>

New application specific scenarios, such as indoor household or office, and subterranean environments can be generated from Unified Robot Description Format (URDF) files that utilize open-source, or custom CAD models.


## Installation

1. Install AutoVRL dependencies: [PyBullet](https://github.com/bulletphysics/bullet3), [Gym](https://github.com/openai/gym) and [Stable Baselines3](https://github.com/DLR-RM/stable-baselines3).

2. Clone the contents of [AutoVRL](https://github.com/Shathushan-Sivashangaran/AutoVRL) to a new directory `AutoVRL`.

3. Execute simulations using the `AutoVRL_Train.py` script. Environment versions v1, v2, v3, v4 and v5 correspond to the 20m x 20m outdoor, 50m x 50m outdoor, 20m x 20m urban, 50m x 50m urban and oval racetrack environments.


## Cite

For more information on AutoVRL refer to the following paper. Cite it if AutoVRL was helpful to your research.

[1] S. Sivashangaran, A. Khairnar and A. Eskandarian, *“AutoVRL: A High Fidelity Autonomous Ground Vehicle Simulator for Sim-to-Real Deep Reinforcement Learning,”* arXiv preprint arXiv:2304.11496, 2023. [Link](https://arxiv.org/pdf/2304.11496.pdf)
