from agent import Agent
from monitor import interact
import gym
import numpy as np
import json

def select_action(Q,state):
    if state in Q:
        return np.argmax(Q[state])
    else:
        return np.random.choice(6)

env = gym.make('Taxi-v3')
agent = Agent()
test = True
print('Training ...')
avg_rewards, best_avg_reward,Q = interact(env, agent)

        

print()
print('Testing ...')
if test:
    state0 = env.reset()
    total_reward = 0
    while True:
        action = select_action(Q,state0)
        env.render()
        next_state,reward,done,info = env.step(action)
        state0 = next_state
        total_reward += reward
        if done:
            env.render()
            print('reached goal')
            print('Total return : ',total_reward)
            break

