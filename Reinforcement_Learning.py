import numpy as np
from gridworld import GridWorld1
import gridrender as gui

env = GridWorld1
gamma  = 0.95
States = range(env.n_states)
Policy = [0 if 0 in env.state_actions[s] else 3 for s in States]
n_MC = 100
T_max = 20

#V_Pi = [0.877, 0.928, 0.988, 0, 0.671, −0.994, 0, −0.828, −0.877, −0.934, −0.994]

def Trajectory(state,action,T_max,policy):
    '''
    Given initials state and action, T_max and a policy, we simulate a trajectory
    :param state:
    :param action:
    :param T_max:
    :param policy:
    :return: reward for the simulated trajectory
    '''
    t = 1
    next_state, reward, term = env.step(state,action)

    while((t<T_max) and not term):
        t+=1
        next_state, reward, term = env.step(next_state, Policy[next_state])

    return (gamma**(t-1))*reward

for i in range(n_MC):
    for s in States:
        for a in env.state_actions[s]:
            r = Trajectory(s,a,T_max, Policy)