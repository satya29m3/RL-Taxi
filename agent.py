import numpy as np
from collections import defaultdict

class Agent:

    def __init__(self, nA=6):
        """ Initialize agent.

        Params
        ======
        - nA: number of actions available to the agent
        """
        self.nA = nA
        self.Q = defaultdict(lambda: np.zeros(self.nA))

    def select_action(self, state,Q,eps):
        """ Given the state, select an action.

        Params
        ======
        - state: the current state of the environment

        Returns
        =======
        - action: an integer, compatible with the task's action space
        """
        action = np.random.choice(self.nA)
        probs = np.ones(self.nA)*(eps/self.nA)
        if state in self.Q:
            best_action = np.argmax(self.Q[state])
            probs[best_action]= 1-eps+(eps/self.nA)
            action = np.random.choice(self.nA,p=probs)
            return action
        return action

    def step(self, state, action, reward, next_state, done,eps,alpha=0.01):
        """ Update the agent's knowledge, using the most recently sampled tuple.

        Params
        ======
        - state: the previous state of the environment
        - action: the agent's previous choice of action
        - reward: last reward received
        - next_state: the current state of the environment
        - done: whether the episode is complete (True or False)
        """
        if not done:
            probs = np.ones(self.nA)*(eps/self.nA)
            if next_state in self.Q:
                best_action = np.argmax(self.Q[next_state])
                probs[best_action] = 1-eps+(eps/self.nA)


            self.Q[state][action] += alpha*(reward + np.dot(probs,self.Q[next_state]) - self.Q[state][action])
        else:
            self.Q[state][action] += alpha*(reward - self.Q[state][action])