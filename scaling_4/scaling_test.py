import numpy as np
from matplotlib import pyplot as plt
from base_model import Model
from RFMC_stacking import RFMC
from ising_model import Ising2D
import pickle
import os
from time import time

if not os.path.exists('plots'):
    os.makedirs('plots')

lengths = np.array([16,32,64,128,256,512,1024,1440,2048,2580,3250,4096,5160, 6502, 8192,10320, 13000, 16384,20643, 26000, 32768], dtype='int')
t = []
le = []

for l in lengths:
    glass = Ising2D(l,l,1.5)
    model = RFMC(glass)
    now = time()
    model.run(1e6)
    le.append(l)
    t.append(time() - now)

    pickle.dump(t,open('times.p','wb'))
    pickle.dump(le, open('lengths.p','wb'))
