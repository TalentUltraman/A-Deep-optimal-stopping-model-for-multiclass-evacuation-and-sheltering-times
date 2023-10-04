#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  4 13:46:05 2023

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
#%%
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
#%%
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

#simulator of wind, demand, and penalty cost
def generate(N,M,y,people,sigma,d,gamma):
    #x = np.linspace(0, T, N)
    wind=np.full((N,M,d),0.0)
    demand=np.full((N,M,d),0.0)
    cost=np.full((N,M,d),0.0)
    rcost=np.full((N,M,d),0.0)
    rdemand=np.full((N,M,d),0.0)
    #+ 2*np.random.randn(N)
    alpha=1
    #A=np.full((d,N),[0,1,2,2,2,2,2])
    #A=np.full((d,N),range(N))
    #A=np.full((d,N),[6,6,6,6,6,6,6]) 
    for epoch in range(M):
      #+ 2*np.random.randn(N)
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
        #plt.plot(demand[epoch],color='gray')
    return wind,cost,demand,y,rdemand,rcost

def generate2(N,M,y,people,sigma,d,gamma):
    #x = np.linspace(0, T, N)
    wind=np.full((N,M,d),0.0)
    demand=np.full((N,M,d),0.0)
    cost=np.full((N,M,d),0.0)
    rcost=np.full((N,M,d),0.0)
    rdemand=np.full((N,M,d),0.0)
    #+ 2*np.random.randn(N)
    alpha=1
    #A=np.full((d,N),[0,1,2,2,2,2,2])
    #A=np.full((d,N),range(N))
    #A=np.full((d,N),[6,6,6,6,6,6,6]) 
    for epoch in range(M):
      #+ 2*np.random.randn(N)
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
        #plt.plot(demand[epoch],color='gray')
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
    #optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    for epoch in range(epochs):
        F = model.forward(X[n])
        #print(F)
        criterion = loss(F,s,X,n,cost,step,J)
        optimizer.zero_grad()
        criterion.backward()
        optimizer.step()
        #print(criterion)

    return F,model      
#%%
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
#%%
#Figue 5
plt.rcParams['figure.dpi'] = 250
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
#%%
#Figue 6

M=5000
C=cap
h=cap*0.5
d=len(All['County'].unique())
steps=4
N=13
#L=3
#T=1.2
sigma=1.5
h=torch.from_numpy(h).float()
C=torch.from_numpy(C).float()
gamma=60#Set different gamma to get plot
wind,cost,demand,Y,rdemand,rcost=generate(N,M,y,people,sigma,d,gamma)
N=N-1
S=stock(3,100,C,h,0.05,N,M,d) 
X=demand
X=torch.from_numpy(X).float()
cost=torch.from_numpy(cost).float()
all_g=np.zeros((steps,N+1,M))

for z in range(steps):
    for i in range(S.M):
        for j in range(S.N+1):
            all_g[z,j,i]=S.g(j,i,X/100,cost,z)
    mean_values = all_g[z].mean(axis=1)
    min_index = np.argmin(mean_values)
    print(min_index)


plt.figure(dpi=250)
for z in range(steps):
    yhat=all_g[z].mean(axis=1)
    x = np.linspace(0, S.N, len(yhat))
    mins, _ =find_peaks(yhat*-1)
    plt.plot(x,yhat,label='C'+str(z+1))
    print(yhat)
plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
for z in range(steps):
    yhat=all_g[z].mean(axis=1)
    mins=yhat == min(yhat)
    plt.scatter(x[mins], yhat[mins])
plt.yticks([])
plt.xlabel('Time(n)')
plt.ylabel('Expected Open Cost')
#%%

#Table 2
#Change the Shelter_names to get different regions

outputs1=np.zeros((5,7))
jj=0

Shelter_names=['C1','C2','C3','C4']
#Shelter_names=['N1','N2','N3','N4']
#Shelter_names=['S1','S2','S3','S4']
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
    pop_str = pop_str.str.replace(',', '')  # remove commas from strings
    pop = pd.to_numeric(pop_str, errors='coerce')
    pop = pop.fillna(0)  # replace missing values with 0
    people[names==i]=pop*0.05
    
for gamma in [30,40,50,60]:
    M=5000
    sigma=1.5
    N=13
    wind,cost,demand,Y,rdemand,rcost=generate(N,M,y,people,sigma,d,gamma)
    N=N-1
    S=stock(3,100,C,h,0.05,N,M,d) 
    X=demand
    X=torch.from_numpy(X).float()
    cost=torch.from_numpy(cost).float()
    mods=[[None]*S.N for i in range(steps)]

    f_mat=np.zeros((steps,S.N+1,S.M))
    f_mat[:,S.N,:]=1


    tau_mat=np.zeros((steps,S.N+1,S.M))
    tau_mat[:,S.N,:]=S.N

    J = torch.zeros((steps+1,S.N+1,S.M))

    for j in range(S.M):
        for i in range(steps-1,-1,-1):
            J[i,S.N,j]=S.g(S.N,j,X/100,cost,i)+J[i+1,S.N,j]

    p_mat=np.zeros((steps,S.N+1,S.M))
    start_time = time.time()
    for n in range(S.N-1,-1,-1):
        for i in range(steps-1,-1,-1):
            probs, mod_temp=NN(n,X/100,S,cost,i,J)
            mods[i][n]=mod_temp
            np_probs=probs.detach().numpy().reshape(S.M)
            p_mat[i,n,:]=np_probs
            f_mat[i,n,:]=(np_probs > 0.5)*1.0
            tau_mat[i,n,:]=np.argmax(f_mat[i],axis=0)
            for j in range(S.M):
                    J[i,n,j]=(S.g(n,j,X/100,cost,i)+J[i+1,n,j])*np_probs[j]+J[i,n+1,j]*(1-np_probs[j])
    end_time=time.time()
    print("--- %s seconds ---" % (end_time - start_time))#Time
    
    N=13
    wind2,cost2,demand2,Y2,rdemand2,rcost2=generate(N,M,y,people,sigma,d,gamma)
    N=N-1
    X2=demand2
    X2=torch.from_numpy(X2).float()
    cost2=torch.from_numpy(cost2).float() 

    f_mat_test=np.zeros((steps,S.N+1,S.M))
    f_mat_test[:,S.N,:]=1


    tau_mat_test=np.zeros((steps,S.N+1,S.M))
    tau_mat_test[:,S.N,:]=S.N


    J_mat_test=np.zeros((steps+1,S.N+1,S.M))
    J_est_test=np.zeros(S.N+1)


    for j in range(S.M):
        for i in range(steps-1,-1,-1):
            J_mat_test[i,S.N,j]=S.g(S.N,j,X2/100,cost2,i)+J_mat_test[i+1,S.N,j]


            
    J_est_test[S.N]=np.mean(J_mat_test[0,S.N,:])


    for n in range(S.N-1,-1,-1):
        for i in range(steps-1,-1,-1):
            mod_curr=mods[i][n]
            probs=mod_curr(X2[n]/100)
            np_probs=probs.detach().numpy().reshape(S.M)
            f_mat_test[i,n,:]=(np_probs > 0.5)*1.0
            tau_mat_test[i,n,:]=np.argmax(f_mat_test[i,:,:], axis=0)
            for j in range(S.M):
                    J_mat_test[i,n,j]=(S.g(n,j,X2/100,cost2,i)+J_mat_test[i+1,n,j])*f_mat_test[i,n,j]+J_mat_test[i,n+1,j]*(1-f_mat_test[i,n,j])

    J_est_test=np.mean(J_mat_test[0,:,:], axis=1)
    J_std_test=np.std(J_mat_test[0,:,:], axis=1)
    J_se_test=J_std_test/(np.sqrt(S.M))

    z=scipy.stats.norm.ppf(0.975)
    lower=J_est_test[0] - z*J_se_test[0]
    upper=J_est_test[0] + z*J_se_test[0]
    outputs1[jj,0]=np.around(np.mean(J_mat_test[0,0,:]),decimals=2)#cost
    
    print(J_est_test[0])
    print(J_se_test[0])
    print(lower)#lower Bound
    print(upper)#Upper Bound
    all_solution2=np.zeros((steps,S.N+1))
    for z in range(steps):
        for i in range(S.N+1):
            all_solution2[z,i]=np.mean(f_mat_test[z,i,:])

    print(all_solution2)#solution

print(outputs1)#Table 2
#%%
#Table 3-5
#Change sigma, gamma and M will get three benchmark table
#generate use penalty function [o(X_n)]
#generate2 use penalty function [o(X_n)]^2

#outputs1=np.zeros((5,3))
outputs1=np.zeros((6,3))
jj=0

#for gamma in [30,40,50,60]:
#for sigma in [1.5,3,5,10,20]
for M in [10,50,100,500,1000,5000]:
    sigma=1
    gamma=40
    N=13
    wind,cost,demand,Y,rdemand,rcost=generate2(N,M,y,people,sigma,d,gamma)
    #wind,cost,demand,Y,rdemand,rcost=generate2(N,M,y,people,sigma,d,gamma)
    N=N-1
    S=stock(3,100,C,h,0.05,N,M,d) 
    X=demand
    X=torch.from_numpy(X).float()
    cost=torch.from_numpy(cost).float()
    mods=[[None]*S.N for i in range(steps)]

    f_mat=np.zeros((steps,S.N+1,S.M))
    f_mat[:,S.N,:]=1


    tau_mat=np.zeros((steps,S.N+1,S.M))
    tau_mat[:,S.N,:]=S.N

    J = torch.zeros((steps+1,S.N+1,S.M))

    for j in range(S.M):
        for i in range(steps-1,-1,-1):
            J[i,S.N,j]=S.g(S.N,j,X/100,cost,i)+J[i+1,S.N,j]

    p_mat=np.zeros((steps,S.N+1,S.M))
    start_time = time.time()
    for n in range(S.N-1,-1,-1):
        for i in range(steps-1,-1,-1):
            probs, mod_temp=NN(n,X/100,S,cost,i,J)
            mods[i][n]=mod_temp
            np_probs=probs.detach().numpy().reshape(S.M)
            p_mat[i,n,:]=np_probs
            f_mat[i,n,:]=(np_probs > 0.5)*1.0
            tau_mat[i,n,:]=np.argmax(f_mat[i],axis=0)
            for j in range(S.M):
                    J[i,n,j]=(S.g(n,j,X/100,cost,i)+J[i+1,n,j])*np_probs[j]+J[i,n+1,j]*(1-np_probs[j])

    end_time=time.time()
    print("--- %s seconds ---" % (end_time - start_time))
    L=10
    Trans_matrix=np.zeros((N,L,L))
    S_wind=np.mean(wind, axis=2)
    R_wind=S_wind//10
    for j in range(N):
        for i in range(M):
                Trans_matrix[j,int(R_wind[j,i]),int(R_wind[j+1,i])]+=1
             

    for j in range(N):
        for i in range(L):
            A=sum(Trans_matrix[j,i,:])
            if A==0:
                Trans_matrix[j,i,i]=1
            else:
                Trans_matrix[j,i,:]=Trans_matrix[j,i,:]/A
                
    P = np.array(people)
    states = np.arange(5, 100, 10)
    gamma1 = 1/gamma

    MDP1 = MDP_S(P, S.C, S.N+1, d, steps, states, gamma1, Trans_matrix)
    optimal_policy = MDP1.backward_induction()
    N=13
    wind2,cost2,demand2,Y2,rdemand2,rcost2=generate2(N,M,y,people,sigma,d,gamma)
    #wind2,cost2,demand2,Y2,rdemand2,rcost2=generate2(N,M,y,people,sigma,d,gamma)
    N=N-1
    X2=demand2
    X2=torch.from_numpy(X2).float()
    cost2=torch.from_numpy(cost2).float() 
    f_mat_test=np.zeros((steps,S.N+1,S.M))
    f_mat_test[:,S.N,:]=1
    tau_mat_test=np.zeros((steps,S.N+1,S.M))
    tau_mat_test[:,S.N,:]=S.N
    J_mat_test=np.zeros((steps+1,S.N+1,S.M))
    J_est_test=np.zeros(S.N+1)
    for j in range(S.M):
        for i in range(steps-1,-1,-1):
            J_mat_test[i,S.N,j]=S.g(S.N,j,X2/100,cost2,i)+J_mat_test[i+1,S.N,j]


            
    J_est_test[S.N]=np.mean(J_mat_test[0,S.N,:])


    for n in range(S.N-1,-1,-1):
        for i in range(steps-1,-1,-1):
            mod_curr=mods[i][n]
            probs=mod_curr(X2[n]/100)
            np_probs=probs.detach().numpy().reshape(S.M)
            f_mat_test[i,n,:]=(np_probs > 0.5)*1.0
            tau_mat_test[i,n,:]=np.argmax(f_mat_test[i,:,:], axis=0)
            for j in range(S.M):
                    J_mat_test[i,n,j]=(S.g(n,j,X2/100,cost2,i)+J_mat_test[i+1,n,j])*f_mat_test[i,n,j]+J_mat_test[i,n+1,j]*(1-f_mat_test[i,n,j])

    J_est_test=np.mean(J_mat_test[0,:,:], axis=1)
    J_std_test=np.std(J_mat_test[0,:,:], axis=1)
    J_se_test=J_std_test/(np.sqrt(S.M))

    z=scipy.stats.norm.ppf(0.975)
    lower=J_est_test[0] - z*J_se_test[0]
    upper=J_est_test[0] + z*J_se_test[0]
    outputs1[jj,1]=np.around(np.mean(J_mat_test[0,0,:]),decimals=2)#MCO-ESST COST
    
    
    J_mat_test2=np.zeros((steps+1,S.N+1,S.M))
    J_est_test2=np.zeros(S.N+1)
    S_wind2=np.mean(wind2, axis=2)
    R_wind2=S_wind2//10

    for j in range(S.M):
        for i in range(steps-1,-1,-1):
            J_mat_test2[i,S.N,j]=S.g(S.N,j,X2/100,cost2,i)+J_mat_test2[i+1,S.N,j]

    for j in range(S.M):
        for n in range(S.N-1,-1,-1):
            actions=optimal_policy[n,int(R_wind2[n,j])]
            for z in range(MDP1.steps-1,-1,-1):
                J_mat_test2[z,n,j]=(S.g(n,j,X2/100,cost2,z)+J_mat_test2[z+1,n,j])*actions[z]+J_mat_test2[z,n+1,j]*(1-actions[z])


    outputs1[jj,0]=np.around(np.mean(J_mat_test2[0,0,:]),decimals=2)#SAA COST
    outputs1[jj,2]=np.around((outputs1[jj,0]-outputs1[jj,1])/outputs1[jj,0],decimals=4)#improve proportion
    print(outputs1[jj,:])
    jj=jj+1

print(outputs1)#Final Table    