# Report

## Implementation
The resolution of the environment involves the utilization of a deep reinforcement learning agent.
The corresponding implementation is available in the continuous_control folder :
* continuous_control.ipynb contains the main code with the training loop and results
* agent.py contains the reinforcement learning agent
* model.py includes the neural networks serving as the estimators
* the weights folder contains the saved weight of the different neural networks.

The starting point for the code was borrowed from the Udacity ddpg-pendulum exercise and subsequently modified to suit the requirements of this specific problem.

### Learning algorithm
The learning algorithm used here is [DDPG](https://arxiv.org/abs/1509.02971).

This algorithm is an actor-critic approach quite similar to the value-based DQN algorithm. Unlike DQN, it can solve tasks with continuous action spaces. It's an off-policy algorithm that uses four neural networks: two for the actor and two for the critic with in each case a local network and a target network.
Each training step the experience (state, action, reward, next state) the 20 agents gained was stored.
Then every second training step the agent learned from a random sample from the stored experience. The actor tries to estimate the
optimal policy by using the estimated state-action values from the critic while critic tries to estimate the optimal q-value function
and learns by using a normal q-learning approach. Using this approach one gains the benefits of value based and policy based
methods at the same time.

### hyperparameters
The following hyperparameters were used:
* replay buffer size: 1e6
* max timesteps: 3000 (all episodes get shutdown after 3000 timesteps)
* minibatch size: 256
* discount factor: 0.99
* tau (soft update for target networks factor): 1e-3
* learning rate: 1e-4 (actor) and 1e-3 (critic)
* update interval (how often to learn): 2
* beta start (factor for the noise added to the actions selected by the actor): 0.1
* beta decay factor: 0.995
* min beta: 0.01

### Neural networks
The actor model is a simple feedforward network:
* Batch normalization
* Input layer: 33 (input) neurons (the state size)
* 1st hidden layer: 128 neurons (leaky relu)
* 2nd hidden layer: 128 neurons (leaky relu)
* output layer: 4 neurons (1 for each action) (tanh)

The critic model:
* Batch normalization
* Input layer: 33 (input) neurons (the state size)
* 1st hidden layer: 132 neurons (action with action_size 4 added) (leaky relu)
* 2nd hidden layer: 128 neurons (leaky relu)
* output layer: 1 neuron

## Results
The agent was able to solve the environment after 133 episodes achieving an average score of 30.11 over the last 100 episodes
of the training.

The average scores of the 20 agents during the training process:
![scores](https://user-images.githubusercontent.com/9535190/78456465-2bd03180-76a4-11ea-8cb9-bedcb75827bd.png)

## possible future improvements
The algorithm could be improved in many ways. For example one could implement some DQN improvements, for example Prioritized Experience Replays
which would improve the learning effect gained from the saved experience. Also true parallel algorithms like A3C could be tried out.
