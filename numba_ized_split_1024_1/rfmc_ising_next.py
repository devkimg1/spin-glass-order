'''
This code is for running RFMC Ising 1024x1024 at various temperatures through
multiple submissions on MSI
'''
import numpy as np
from matplotlib import pyplot as plt
from base_model import Model
from RFMC_stacking import RFMC
from ising_model import Ising2D
import pickle
import sys

T = float(sys.argv[1])/100.

x = pickle.load(open(('runs/Ising1024x1024_T%1.2f'%T).replace('.','-')+'.p', 'rb'))

for i in range(30):
    x.run(1e8)

pickle.dump(x, open(('runs/Ising1024x1024_T%1.2f'%T).replace('.','-')+'.p', 'wb'))
