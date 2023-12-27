# A DRL solution to help reduce the cost in waiting time of securing a traffic light for cyclists

Code used for the paper [A DRL solution to help reduce the cost in waiting time of securing a traffic light for cyclists](https://arxiv.org/abs/2311.13905).


## Dependencies and installation

This project uses [Python 3.8.10](https://www.python.org/downloads/release/python-3810/). Use the package manager [pip](https://pypi.org/project/pip/) to install the dependencies :

```bash
pip install -r requirements.txt
```
 SUMO also needs to be installed following the instructions available [here](https://sumo.dlr.de/docs/Installing/index.html)

## Usage

The files that can be executed are `main.py` and `graphs.py`. `main.py` can be used to train a DRL agent or to test the performance of any traffic light control methods detailed in the paper (including a trained DRL agent). The logs of the simulations launched with `main.py` are stored in the `files/` folder. `graphs.py` uses this logs to create graphs and save them in the `images/` folder. 

The parameters of `main.py` are :

1. `-m`/`--method` : specify the method to control the traffic light. Accepted methods are `3DQN` (DRL algorithm), `PPO` (DRL algorithm), `static_secured`, `unsecured` and `actuated`.

2. `--load-scenario` : the simulation(s) use the logs of previous ones to spawn the vehicles. This argument is used in order to compare different methods under the exact same conditions. The logs of the last simulation(s) using the `3DQN` method are loaded. If they do not exist, the logs of the last simulation(s) using the `actuated` method are used instead.

3. `--test` : specifiy that the simulation is a test. If a DRL method is used, a trained agent is loaded and it do not learn during the simulation. A test simulation is 1 day long (ie. 3600*24 steps).

4. `--full-test` : used to generate the second part of the results detailed in the paper. The bike trafic is linearly modified and 5 simulations are launched per bike trafic.

5. `--gui` : display the simulation(s).

`graphs.py` produces graphs for the training if executed without arguments. `--test` and `--full-test` can be used with the same purposes as for `main.py`.

You can execute the `paper.sh` script to reproduce all the results described in the paper. Execution takes time as the training of the agent lasts for more than 6000000 steps and the "full-test" results lasts for 2160000 steps.
