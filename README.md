# Physics-Inspired Temporal Learning of Quadrotor Dynamics for Accurate Model Predictive Trajectory Tracking

## License
Please be aware that this code was originally implemented for research purposes and may be subject to changes and any fitness for a particular purpose is disclaimed. To inquire about commercial licenses, please contact Alessandro Saviolo (alessandro.saviolo@nyu.edu), Guanrui Li (lguanrui@nyu.edu), and Prof. Giuseppe Loianno (loiannog@nyu.edu).
```
    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>.
    
```

## Citation
If you publish a paper with our simulator, please cite our paper: 
```
@article{saviolo2022pitcn,
    author    = {Saviolo, Alessandro and Li, Guanrui and Loianno, Giuseppe},
    title     = {Physics-Inspired Temporal Learning of Quadrotor Dynamics for Accurate Model Predictive Trajectory Tracking},
    journal   = {arXiv preprint arXiv:2206.03305},
    year      = {2022}
}
 ```
 
## Overview

#### Abstract
Accurately modeling quadrotor's system dynamics is critical for guaranteeing agile, safe, and stable navigation. The model needs to capture the system behavior in multiple flight regimes and operating conditions, including those producing highly nonlinear effects such as aerodynamic forces and torques, rotor interactions, or possible system configuration modifications. Classical approaches rely on handcrafted models and struggle to generalize and scale to capture these effects. In this paper, we present a novel Physics-Inspired Temporal Convolutional Network (PI-TCN) approach to learning quadrotor's system dynamics purely from robot experience. Our approach combines the expressive power of sparse temporal convolutions and dense feed-forward connections to make accurate system predictions. In addition, physics constraints are embedded in the training process to facilitate the network's generalization capabilities to data outside the training distribution. Finally, we design a model predictive control approach that incorporates the learned dynamics for accurate closed-loop trajectory tracking fully exploiting the learned model predictions in a receding horizon fashion. Experimental results demonstrate that our approach accurately extracts the structure of the quadrotor's dynamics from data, capturing effects that would remain hidden to classical approaches. To the best of our knowledge, this is the first time physics-inspired deep learning is successfully applied to temporal convolutional networks and to the system identification task, while concurrently enabling predictive control.
