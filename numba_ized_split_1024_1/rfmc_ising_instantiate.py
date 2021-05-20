'''
This code is for running RFMC Ising 128x128 at various temperatures through
multiple submissions on MSI
'''
import numpy as np
from matplotlib import pyplot as plt
from base_model import Model
from RFMC_stacking import RFMC
from ising_model import Ising2D
import pickle
import os

temperatures = np.arange(1.5,2.5,0.05)

if not os.path.exists('runs'):
    os.makedirs('runs')

for T in temperatures:
    model = Ising2D(2**10,2**10,T)
    x = RFMC(model)
    pickle.dump(x, open(('runs/Ising1024x1024_T%1.2f'%T).replace('.','-')+'.p', 'wb'))
