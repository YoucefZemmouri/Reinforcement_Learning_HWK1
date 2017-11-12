import numpy as np
import matplotlib.pyplot as plt

class MarkovDecisionProcess:
    def __init__(self,states, actions, probability, reward, gamma, precision = 0.01):
        self.states = states
        self.actions = actions
        self.N_states = len(states)
        self.N_actions = len(actions)
        self.P = probability
        self.R = reward
        self.gamma = gamma
        self.epsilon = precision*(1-gamma)/(2*gamma)

    def OptimalBellmanValue(self, V):
        V_opt = np.zeros((self.N_states))
        for s in self.states:
            V_opt[s] = np.max(self.R[s,:]+self.gamma*(self.P[:,s,:].dot(V)),axis=0)
        return V_opt

    def OptimalBellmanPolicy(self,V):
        Pi_opt= np.zeros((self.N_states))
        for s in self.states:
            Pi_opt[s] = 0
            temp = self.R[s,0]+self.gamma*(self.P[0,s,:].dot(V))
            for a in self.actions:
                if self.R[s,a]+self.gamma*(self.P[a,s,:].dot(V)) > temp:
                    Pi_opt[s] = a
                    temp = self.R[s,a]+self.gamma*(self.P[a,s,:].dot(V))
        return Pi_opt.astype(int)

    def PolicyEvaluation(self,Pi):
        p_Pi = np.zeros((self.N_states,self.N_states))
        r_Pi = np.zeros((self.N_states))
        for i in range(self.N_states):
            r_Pi[i] = self.R[i,Pi[i]]
            for j in range(self.N_states):
                p_Pi[i,j] = self.P[Pi[i],i,j]
        return np.linalg.inv(np.identity(self.N_states)-self.gamma*p_Pi).dot(r_Pi)

    def ValueIteratin(self,Init):
        V1 = Init.copy()
        V2 = self.OptimalBellmanValue(Init)
        error = np.max(np.absolute(V2-V1))
        i = 1
        while error > self.epsilon:
            i = i + 1
            V1 = V2.copy()
            V2 = self.OptimalBellmanValue(V2)
            error = np.max(np.absolute(V2 - V1))
        Pi = self.OptimalBellmanPolicy(V2)
        return  V2, Pi, i

    def PlotConvergenceValue(self,Pi,Init):
        Error = []
        V_star = self.PolicyEvaluation(Pi)
        V1 = Init.copy()
        V2 = self.OptimalBellmanValue(Init)
        error = np.max(np.absolute(V2 - V1))
        Error.append(np.max(np.absolute(V1 - V_star)))
        Error.append(np.max(np.absolute(V2 - V_star)))
        i = 1
        while error > self.epsilon:
            i = i+1
            V1 = V2.copy()
            V2 = self.OptimalBellmanValue(V2)
            error = np.max(np.absolute(V2 - V1))
            Error.append(np.max(np.absolute(V2 - V_star)))
        plt.plot(Error)
        plt.xlabel("iterations")
        plt.ylabel("Error")
        plt.show()

    def PolicyIteration(self, Init):
        Pi = Init.copy()
        V_1 = self.PolicyEvaluation(Pi)
        Pi = self.OptimalBellmanPolicy(V_1)
        V_2 = self.PolicyEvaluation(Pi)
        while not np.array_equal(V_1,V_2):
            V_1 = V_2.copy()
            Pi = self.OptimalBellmanPolicy(V_2)
            V_2 = self.PolicyEvaluation(Pi)
        return Pi

    def PlotConvergencePolicy(self,Pi,Init):
        V_star = self.PolicyEvaluation(Pi)
        Error = []
        Pi_1 = Init.copy()
        V_1 = self.PolicyEvaluation(Pi_1)
        Pi_1 = self.OptimalBellmanPolicy(V_1)
        V_2 = self.PolicyEvaluation(Pi_1)
        Error.append(np.max(np.absolute(V_1 - V_star)))
        Error.append(np.max(np.absolute(V_2 - V_star)))
        while not np.array_equal(V_1,V_2):
            print(9)
            V_1 = V_2.copy()
            Pi_opt = self.OptimalBellmanPolicy(V_2)
            V_2 = self.PolicyEvaluation(Pi_opt)
            Error.append(np.max(np.absolute(V_2 - V_star)))
        plt.plot(Error)
        plt.xlabel("iterations")
        plt.ylabel("Error")
        plt.show()


