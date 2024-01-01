## udacity Deep Reinforcement Learning
## project 1  : navigation (Banana Collector)

### Project Details
This project is one of the Udacity Deep Reinforcement Learning Nano Degree program.
It will train an agent to navigate and collect bananas in a square world. 
The training uses a deep Q-Network algorithms.

A reward of +1 is provided for collecting a yellow banana, and a reward of -1 is provided for collecting a blue banana.  Thus, the goal of the agent is to collect as many yellow bananas as possible while avoiding blue bananas.  

The state space has 37 dimensions and contains the agent's velocity, along with ray-based perception of objects around agent's forward direction.  Given this information, the agent has to learn how to best select actions.  Four discrete actions are available, corresponding to:
- **`0`** - move forward.
- **`1`** - move backward.
- **`2`** - turn left.
- **`3`** - turn right.

The task is episodic, and in order to solve the environment, the agent must get an average score of +13 over 100 consecutive episodes.

The following is the unity agent information for this project.
```
INFO:unityagents:
'Academy' started successfully!
Unity Academy name: Academy
        Number of Brains: 1
        Number of External Brains : 1
        Lesson number : 0
        Reset Parameters :
		
Unity brain name: BananaBrain
        Number of Visual Observations (per agent): 0
        Vector Observation space type: continuous
        Vector Observation space size (per agent): 37
        Number of stacked Vector Observation: 1
        Vector Action space type: discrete
        Vector Action space size (per agent): 4
        Vector Action descriptions: , , ,
```

### Getting Started
The project has been developped in the udacity workspace.
The "python" folder at the root of the directory contains all necessary packages.

### Instructions

Following the instructions in `Navigation.ipynb` allows to train an agent.
Hyperparameters have been set to succesfully train the agent and reach the goal of scoring more than 13 in average for 100 episodes.
In our case the agent was trained to score over 15

It is possible to tune hyper-parameters in the model, for examples the number of hidden layers, the width of the layers, epsilon, the soft update parameters, etc.

The model weights are saved and van be loaded later.
