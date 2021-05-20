import numpy as np
from matplotlib import pyplot as plt
from base_model import Model
from RFMC_stacking import RFMC
from ising_model import Ising2D
import pickle
from utils import _ising_exact_energy_2D as exact
import os

temperatures = np.arange(1.5,2.5,0.05)
average_e = np.zeros(np.shape(temperatures))
average_s = np.zeros(np.shape(temperatures))
e_dif = np.zeros(np.shape(temperatures))

if not os.path.exists('plots'):
    os.makedirs('plots')

'''for T in temperatures:
    model = pickle.load(open(('runs/Ising128x128_T%1.2f'%T).replace('.','-')+'.p', 'rb'))

    plt.plot(np.cumsum(model.time_series), model.energy_series, 'o')
    plt.title('Ising128x128_T%1.2f'%T)
    plt.xlabel('steps')
    plt.ylabel('energy')
    plt.savefig(('plots/Ising128x128_T%1.2f'%T).replace('.','-')+'.png')
    plt.close()
'''
i = 0

for T in temperatures:
    '''if '%1.2f'%T == '1.75' or '%1.2f'%T == '1.90':
        i += 1
        continue'''
    t = pickle.load(open(('runs/1024x1024_%1.2f/time_9'%T).replace('.','-')+'.p', 'rb'))
    e = pickle.load(open(('runs/1024x1024_%1.2f/energy_9'%T).replace('.','-')+'.p', 'rb'))
    #for j in [36,37,38]:
#        e2 = pickle.load(open(('runs/1024x1024_%1.2f/energy_%d'%(T,j)).replace('.','-')+'.p', 'rb'))
#        t2 = pickle.load(open(('runs/1024x1024_%1.2f/time_%d'%(T,j)).replace('.','-')+'.p', 'rb'))
#        e = np.concatenate([e,e2])
#        t = np.concatenate([t,t2])
    average_e[i] = np.sum(t*e)/np.sum(t)/1024**2
    m2 = np.sum(t * e**2) / np.sum(t)/1024**4
    #print(m2)
    average_s[i] = (m2 - average_e[i]**2)/T**2 * 1024**2
    e_dif[i] = exact(T) - average_e[i]
    i += 1

print(average_s)
T = np.linspace(1.5,2.5,100)
E = np.zeros(np.shape(T))

for i in range(len(T)):
    E[i] = exact(T[i])

plt.plot(T,E)
plt.plot(temperatures, average_e,'o')
plt.title('1024x1024 RFMC Ising Compared to Analytical Solution')
plt.xlabel('Temperature')
plt.ylabel('Average Energy')

plt.savefig('plots/RFMC_Ising_Verification_Energy.png')

plt.close()

V = np.zeros(np.shape(T))

for i in range(len(T)):
    V[i] = -(exact(T[i]-0.0001) - exact(T[i]+0.0001))/0.0002

plt.plot(T,V)
plt.plot(temperatures, average_s,'o')
plt.title('1024x1024 RFMC Ising Compared to Analytical Solution')
plt.xlabel('Temperature')
plt.ylabel('Susceptibility')

plt.savefig('plots/RFMC_Ising_Verification_Susceptibility.png')

plt.close()

plt.plot(temperatures, e_dif,'o')
plt.title('1024x1024 RFMC Ising Compared to Analytical Solution')
plt.xlabel('Temperature')
plt.ylabel('E Dif')

plt.savefig('plots/RFMC_Ising_Verification_Dif.png')

plt.close()
