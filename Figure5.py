#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  5 13:27:11 2023

@author: hanwen
"""

import pandas as pd
import torch
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import scipy.stats
from scipy.signal import find_peaks
import time
import gurobipy as gp
from gurobipy import GRB
np.random.seed(234198)

dummy_wind=pd.read_csv(r'/Users/hanwen/Desktop/GIS project/Code/Final1/Project1/dummy_wind_Florence.csv')
County_shelter=pd.read_csv(r'/Users/hanwen/Desktop/GIS project/Code/Final1/Project1/County_shelter.csv')
SC_population=pd.read_csv(r'/Users/hanwen/Desktop/GIS project/Code/Final1/Project1/SC_population.csv')
SC_population['CTYNAME']=SC_population['CTYNAME'].str.replace(' County', '')
County_shelter=County_shelter.sort_values(by='County')
SC_population=SC_population.sort_values(by = 'CTYNAME')
v_Shelter=pd.read_csv(r'/Users/hanwen/Desktop/GIS project/Code/Final1/Project1/vunlarable.csv')
v_Shelter=v_Shelter.iloc[:,:2]
new_names={0:'CTYNAME',1:'pop'}
for i, name in new_names.items():
    v_Shelter = v_Shelter.rename(columns={v_Shelter.columns[i]: name})

#Value function
class stock:
    def __init__(self, T, K, C, h, r, N, M,d):
        self.T = T
        self.K=K
        self.r=r
        self.N=N
        self.M=M
        self.C=C
        self.h=h
        self.d=d
        self.lim=torch.zeros(self.d)
    def g(self,n,m,X,cost,step):
        min1=sum(self.h[step,:]*(self.N-n)+sum(cost[:(int(n)+1),m,:]*(torch.clamp(X[:(int(n)+1),m,:]*100-torch.sum(self.C[:step,:],0),min=self.lim,max=self.C[step,:]))))
        return min1

def generate(N,M,y,people,sigma,d,gamma):

    wind=np.full((N,M,d),0.0)
    demand=np.full((N,M,d),0.0)
    cost=np.full((N,M,d),0.0)
    rcost=np.full((N,M,d),0.0)
    rdemand=np.full((N,M,d),0.0)
    alpha=1 
    for epoch in range(M):
        yhat=y
        c=yhat/10
        dt = 1.0 / N
        alpha=1
        dB = np.random.randn(d)
        de=np.full((N,d),0.0)
        dde=np.full((N,d),0.0)
        de[0]=0
        for i in range(N):
            if i < N-1:
                dB = np.random.randn(d)
                dde[i]= alpha*(yhat[i+1]-yhat[i])+sigma*dB
                de[i+1]=de[i]+dde[i]
        de=np.clip(de,0,99.9)        
        cost[:,epoch,:]=((de.T/gamma)).T
        wind[:,epoch,:]=de
        rcost[:,epoch,:]=((yhat.T/gamma)).T
        for j in range(d):
            demand[:,epoch,j]=people[j]*de[:,j]/64
            rdemand[:,epoch,j]=people[j]*yhat[:,j]/64
    return wind,cost,demand,y,rdemand,rcost

def generate2(N,M,y,people,sigma,d,gamma):
    wind=np.full((N,M,d),0.0)
    demand=np.full((N,M,d),0.0)
    cost=np.full((N,M,d),0.0)
    rcost=np.full((N,M,d),0.0)
    rdemand=np.full((N,M,d),0.0)
    alpha=1

    for epoch in range(M):
        yhat=y
        c=yhat/10
        dt = 1.0 / N
        alpha=1
        dB = np.random.randn(d)
        de=np.full((N,d),0.0)
        dde=np.full((N,d),0.0)
        de[0]=0
        for i in range(N):
            if i < N-1:
                dB = np.random.randn(d)
                dde[i]= alpha*(yhat[i+1]-yhat[i])+sigma*dB
                de[i+1]=de[i]+dde[i]
        de=np.clip(de,0,99.9)        
        cost[:,epoch,:]=(((de**2).T/(gamma**2))).T
        wind[:,epoch,:]=de
        rcost[:,epoch,:]=((yhat.T/gamma)).T
        for j in range(d):
            demand[:,epoch,j]=people[j]*de[:,j]/64
            rdemand[:,epoch,j]=people[j]*yhat[:,j]/64
    return wind,cost,demand,y,rdemand,rcost  

    
#SAA method
class MDP_S:
    def __init__(self, P, C, N, d, steps, states, gamma1, Trans_matrix):
        self.N = N
        self.C = C
        self.P = torch.from_numpy(P).float()
        self.steps = steps
        self.gamma1 = gamma1
        self.states = states
        self.T = Trans_matrix
        self.lim = torch.zeros(d)
        self.costs = torch.zeros((steps+1, N + 1, len(states)))

    def cost(self, k, n, L, a):  # Implement your specific cost for action k and time n here
        if a == 1:
            return sum(self.C[k, :]) * 0.5*(S.N-n) + self.costs[k + 1, n, L]
        else:
            sum_cost=0
            expected_cost=0
            for i in range(k,self.steps):
                sum_cost += sum(torch.clamp(self.P * states[L]/64-torch.sum(self.C[:i,:],0), min=self.lim, max=self.C[i, :]))*states[L]*self.gamma1
            for i in range(len(self.states)):
                expected_cost += self.T[n, L, i] * self.costs[k, n+1, i]     
            return sum_cost + expected_cost

    def backward_induction(self):
        optimal_policy = np.zeros((self.N, len(self.states),self.steps), dtype=int)
        for n in range(self.N - 1, -1, -1):
            for k in range(self.steps-1,-1,-1):
                for L in range(len(self.states)):
                    if n == self.N - 1:
                        action_costs = [self.cost(k, n, L, 1)]
                        optimal_policy[n,L,k] = 1
                    else:
                        action_costs = [self.cost(k, n, L, a) for a in [0, 1]]
                        actions=np.argmin([self.cost(k, n, L, a) for a in [0,1]]) 
                        if actions==0:
                            optimal_policy[n,L,k:] = np.zeros(self.steps-k)
                        else:
                            optimal_policy[n,L,k] =actions
                    self.costs[k, n, L] = min(action_costs)

        return optimal_policy

#Multi-Class Deep learning Method
class NeuralNet(torch.nn.Module):
    def __init__(self, d, q1, q2):
        super(NeuralNet, self).__init__()
        self.a1 = nn.Linear(d, q1) 
        self.relu = nn.ReLU()
        self.a2 = nn.Linear(q1, q2)
        self.a3 = nn.Linear(q2, 1)  
        self.sigmoid=nn.Sigmoid()
    
    def forward(self, x):
        out = self.a1(x)
        out = self.relu(out)
        out = self.a2(out)
        out = self.relu(out)
        out = self.a3(out)
        out = self.sigmoid(out)
        
        return out
    
def loss(y_pred,s,X, n,cost,step,J):
    r_n=torch.zeros((s.M))
    for m in range(0,s.M):
        r_n[m]=(s.g(n,m,X,cost,step)+J[step+1,n,m])*y_pred[m] + J[step,n+1,m]*(1-y_pred[m])
    return(r_n.mean())
  

def NN(n,X,s,cost,step,J):
    epochs=50
    model=NeuralNet(s.d,s.d+40,s.d+40)
    optimizer = torch.optim.Adam(model.parameters(), lr = 0.001)

    for epoch in range(epochs):
        F = model.forward(X[n])

        criterion = loss(F,s,X,n,cost,step,J)
        optimizer.zero_grad()
        criterion.backward()
        optimizer.step()


    return F,model      

Shelter_names=['C1','C2','C3','C4']
All=County_shelter.loc[County_shelter.Group.isin(Shelter_names)]
pro=np.full((len(All['County'].unique())),1.0)
cap=np.zeros((4,len(All['County'].unique())))
names=All['County'].unique()
c_num=len(names)
a=0
for i in Shelter_names:
    ingroup=All.loc[All.Group==i,:]
    b=np.zeros(c_num)
    for j in ingroup['County'].unique():
        b[names==j]=All.loc[(All.Group==i)&(All.County==j),'Capacity']
    cap[a,:]=b
    a=a+1
people=np.zeros(c_num)
for i in names:
    pop_str = v_Shelter.loc[v_Shelter.CTYNAME==i,'pop']
    pop_str = pop_str.str.replace(',', '')  
    pop = pd.to_numeric(pop_str, errors='coerce')
    pop = pop.fillna(0)  
    people[names==i]=pop*0.05

    
y=dummy_wind[All['County'].unique()]
y=np.array(y.iloc[list(range(0, 25, 2)),:])
y[0,:]=0
y[y<0]=0    

M=5000
C=cap
h=cap*0.5
d=len(All['County'].unique())
steps=4
N=13
sigma=1.5
h=torch.from_numpy(h).float()
C=torch.from_numpy(C).float()
gamma=30
wind,cost,demand,Y,rdemand,rcost=generate(N,M,y,people,sigma,d,gamma)
N=N-1

plt.rcParams['figure.dpi'] = 600
fig, axs = plt.subplots(nrows=3, ncols=3, figsize=(16, 8))

names = All['County'].unique()

for i, ax in enumerate(axs.flatten()):
    if i < len(names):
        ax.plot(y[:, i], label='Wind Intensity', color='blue')
        ax.set_ylabel('0 Wind Intensity (kt) 70', color='blue')
        ax.set_xticks(range(0, 13, 2))  # Take every second value for 24-hour interval
        ax.set_xticklabels(['Sep 8', 'Sep 9', 'Sep 10', 'Sep 11', 'Sep 12', 'Sep 13', 'Sep 14'], fontsize=10, ha='center')
        ax.tick_params(axis='x', which='both', length=0)
        ax.set_title(names[i])
        ax.axhline(y=0, color='black', linestyle='--', linewidth=0.5)  # add bottom line
        ax.axhline(y=70, color='black', linestyle='--', linewidth=0.5)  # add top line
        ax.set_ylim(0, 70)  # set y limit to only show values between 0 and 45
        ax.set_yticks([])  # set y-axis ticks to only show values 0 and 45
        ax.spines['left'].set_edgecolor('blue')
        ax.tick_params(axis='y', which='both', labelcolor='blue')

        # create a secondary y-axis for cost attribute
        ax2 = ax.twinx()
        ax2.plot(rcost[:, i,0]*rdemand[:,i,0], color='red', label='Cost')
        ax2.set_ylabel('0   Penalty Cost    7000', color='red')
        ax2.set_ylim(0, 7000)  # set y limit to show max value of 1.6
        ax2.set_yticks([])  # set y-axis ticks to show 0 and 1.6
        ax2.tick_params(axis='y', colors='red')
        ax2.spines['right'].set_edgecolor('red')
    else:
        fig.delaxes(ax)

fig.subplots_adjust(hspace=0.5, wspace=0.2)
plt.show()