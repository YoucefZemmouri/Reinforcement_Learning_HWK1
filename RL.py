import numpy as np
from gridworld import GridWorld1
import matplotlib.pyplot as plt
import gridrender as gui

env = GridWorld1
gamma  = 0.95
States = range(env.n_states)
Policy = [0 if 0 in env.state_actions[s] else 3 for s in States]
n_MC = 10000
T_max = 100
mu0 =(1/11)*np.ones(11)
V_opt = [0.87691855, 0.92820033, 0.98817903, 0.00000000, 0.67106071, -0.99447514, 0.00000000, -0.82847001, -0.87691855,-0.93358351, -0.99447514]

##Q4
def Trajectory(state,action,T_max,policy):
    t = 1
    next_state, reward, term = env.step(state,action)
    gamma_r = (gamma**(t-1))*reward
    while((t<T_max) and not term):
        t+=1
        next_state, reward, term = env.step(next_state, policy[next_state])
        gamma_r += (gamma ** (t - 1)) * reward
    return gamma_r

N = np.zeros((env.n_states,len(env.action_names)))
Q = np.zeros((env.n_states,len(env.action_names)))
Error = []
for i in range(1,n_MC):
    Q_temp = np.zeros((env.n_states,len(env.action_names)))
    for s in States:
        for a in env.state_actions[s]:
            N[s,a]+=1
            Q[s,a]+=Trajectory(s,a,T_max,Policy)
    for i in range(env.n_states):
        for j in range(len(env.action_names)):
            if N[i, j] != 0:
                Q_temp[i, j] = Q[i, j] / N[i, j]
    V = [Q_temp[s][Policy[s]] for s in States]
    Error.append(sum(np.multiply(V,mu0))-sum(np.multiply(V_opt,mu0)))

plt.plot(Error)
plt.xlabel("iterations")
plt.ylabel("Error")
plt.show()

##Q5