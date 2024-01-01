# Project 1 Navigation : report

In this project, a DQN learning algorithm is used to train an agent to navigate in the "banana collector" environment of "Unity".
The report describes the learning algorithm, the architecture of the neural networks and the values of the hyper parameters.

### Training Code
The code is written in PyTorch and Python3, executed in Jupyter Notebook
- Navigation.ipynb	: Main Instruction file
- dqn_agent.py	: Agent and ReplayBuffer Class
- model.py	: Build QNetwork and train function
- checkpoint.pth : Saved Model Weights


### Learning Algorithm
#### Deep Q-Network

**Q-learning** is a value-based Reinforcement Learning algorithm that is used to estimate the optimal action-selection policy using a q function, *`Q(s,a)`*

It's goal is to maximize the value function "Q" which is the maximum sum of rewards r<sub>t</sub> discounted by &gamma; at each timestep t, achievable by a behaviour policy *&pi;=P(a|s)*, after making an
observation (s) and taking an action (a)

The pseudo code of Q learning algorithm is :
1. Initialze Q-values *Q(s,a)* arbitrarily for all state-action pairs.
2. For i=1 to # num_episodes <br/>
  Choose an action A<sub>t</sub> in the current state (s) based on current Q-value estimates (e,g &epsilon;-greedy) </br>
  Take action A<sub>t</sub> and observe reward and state, R<sub>t+1</sub>, S<sub>t+1</sub>
  Update *Q(s|a)* <br/>
  
**Q-networks** approximate the Q-function with a neural network *Q(s, a, θ)*. 
Given a state, it outputs the Q-values for each possible action <br/>
The model parameters &theta are optimise by Stochastic Gradient Descent.

**DQN-Experience replay**<br/>
In order to deal with correlated states, the agent build a dataset of experience and then makes random samples from
the dataset.<br/>

**DQN-Fixed Target** <br/>
Also, the agent fixes the parameter &theta;<sup>-</sup> and then with some frequency updates them<br/>

**Neural Network Architecture**<br/>
The state space has 37 dimensions and the size of action space per state is 4.<br/>
so the number of input features of NN is 37 and the output size is 4.<br/>
And the number of hidden layers and each size is configurable in this project in the model.py file.<br/>
The hidden layers used in this project is [64,128] ie, 2 layers with 64, 128 neurons in each layer. <br/>

Number of features
* Input layers : 37
* Hidden layer 1: 64
* Hidden layer 2: 128
* Output layer : 4

~~~python
QNetwork(
  (layers): ModuleList(
    (0): Linear(in_features=37, out_features=64, bias=True)
    (1): Linear(in_features=64, out_features=128, bias=True)
  )
  (output): Linear(in_features=128, out_features=4, bias=True)
)
~~~

**Hyper-parameters**<br/>

- BUFFER_SIZE = int(1e5)  # replay buffer size
- BATCH_SIZE = 64         # minibatch size
- GAMMA = 0.99            # discount factor
- TAU = 1e-3              # for soft update of target parameters
- LR = 5e-4               # learning rate 
- UPDATE_EVERY = 4        # how often to update the network

### Plot of Rewards

Below plot of rewards per episode
- plot an average reward (over 100 episodes)
- the objective for the average score is set to 15 (above the minimum of 13 required to solve the environment)
- It shows this agent solve the environment in 774 episodes
![plot](https://github.com/ealbenque/demo-repo/assets/137990986/fe7ed600-4a37-4d68-bfbd-0b415588ac9b)

~~~python
Episode 100	Average Score: 0.84
Episode 200	Average Score: 3.40
Episode 300	Average Score: 6.71
Episode 400	Average Score: 9.94
Episode 500	Average Score: 12.99
Episode 600	Average Score: 13.59
Episode 700	Average Score: 13.82
Episode 774	Average Score: 15.04
Environment solved in 774 episodes!	Average Score: 15.04
~~~

### Ideas for Future Work
This project used simply a vanila DQN.<br/>
As a future work, improved algorithms like double DQN, dueling DQN and prioritized experince replay can be applied.
Fine-tuning the hyper parameters could improve the overall performance.
