import random
import math
from collections import deque
import numpy as np

import gym

import random
import math
from collections import deque
import numpy as np
from matplotlib import pyplot
import gym

#Setup the Gym Environment

#Define training parameters
#number of games we want the agent to play
n_episodes = 200

#The number of tick to win a particular episode
winning_ticks = 100

#the maximum number of environment timesteps
max_env_steps = 1000000

#Discount Factor
gamma = 1.0

#Exploration - Used in choosing actions that 
#has best long term effect
epsilon=1.0

#we want the agent to explore at least this amount
epsilon_min = 0.01

#we want to decrease the number of explorations as it gets good at playing games
epsilon_decay=0.995

#Learning rate
alpha=0.05
alpha_decay=0.1

batch_size=64
monitor=False
quiet=False

#Building the Neural Network
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

def NNModel():
    model = Sequential()
     # 'Dense' is the basic form of a neural network layer
    # Input Layer of state size(4) and Hidden Layer with 24 nodes
    model.add(Dense(24,input_dim=4,activation='relu'))
    # Hidden layer with 64 nodes
    model.add(Dense(64,activation='relu'))
    # Output Layer with # of actions: 2 nodes (left, right)
    model.add(Dense(2,activation='relu'))
    model.compile(loss='mse',optimizer=Adam(lr=alpha,decay=alpha_decay),metrics=["accuracy"])
    return model

model = NNModel()

#Environment Parameters
memory = deque(maxlen=1000000)
env  = gym.make("CartPole-v0")
if max_env_steps is not None:
    env.max_episode_steps=max_env_steps
    

    #remember function will simply store states, actions and resulting rewards to the memory
def remember(state,action,reward,next_state,done):
     memory.append((state,action,reward,next_state,done))

def choose_action(state,epsilon):
    if np.random.sample() <= epsilon:
        return env.action_space.sample()
    else:
        return np.argmax(model.predict(state))

def get_epsilon(t):
    return max(epsilon_min, min(epsilon, 1.0 - math.log10((t+1)*epsilon_decay)))

#Transform the state in 1 X 4 Matrics
def preprocess_input(state):
    return np.reshape(state,[1,4])

#A method that trains the model with samples experiences from the memory 
def replay(batch_size,epsilon):
    x_batch,y_batch=[],[]
    mini_batch = random.sample(memory, min(len(memory),batch_size))
    
    for state, action, reward,next_state,done in mini_batch:
        y_target = model.predict(state)
        y_target[0][action] = reward if done else reward + gamma * np.max(model.predict(next_state)[0])
        x_batch.append(state[0])
        y_batch.append(y_target[0])
        
   # Train the Neural Network with batches
    model.fit(np.array(x_batch),np.array(y_batch),batch_size=len(x_batch),verbose=0)
    
    if epsilon > epsilon_min:
        epsilon *= epsilon_decay

#Define the Run function
def run():
    scores = deque(maxlen=100)
    time_steps = []
    average_scores = []
    for e in range(n_episodes):
        state = preprocess_input(env.reset())
        done=False
        i=0
        while not done:
            action = choose_action(state,get_epsilon(e))
            next_state, reward, done,_ = env.step(action)
            env.render()
            next_state = preprocess_input(next_state)
            remember(state,action,reward,next_state,done)
            state = next_state
            i += 1
        scores.append(i)
        mean_score = np.mean(scores)
        time_steps.append(i)
        average_scores.append(mean_score)
        if mean_score >= winning_ticks and e >= 100:
            if not quiet : print("Run {} times, solved after {} trials".format(e, e-100))
            return e-100,time_steps,average_scores
        if e % 20 == 0:
            print('[Episode {}] - mean Survival time over last 20 episodes was {} ticks.'.format(e,mean_score))
        
        replay(batch_size,get_epsilon(e)) 
        
    if not quite: print("Did not solve after {} epsilon".format(e))
    return e,time_steps,average_scores

for i in range(3):
    episode, timesteps, average_scores = run()
    pyplot.plot(timesteps,average_scores)
    pyplot.show()



