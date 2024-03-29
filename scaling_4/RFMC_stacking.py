'''

'''

import numpy as np
import pickle
import os

class RFMC():
    model: None
    energy_series: None
    time_series: None
    steps = int(0)

    def __init__(self, model):
        self.model = model
        self.rng = np.random.default_rng()
        #self.energy_series = np.empty(1,dtype='int')
        #self.time_series = np.empty(1,dtype='float')
        self.steps = 0
        self.next_run_number = 0
        if not os.path.exists(('runs/%dx%d_%1.2f'%(self.model.n_x,self.model.n_y,self.model.T)).replace('.','-')):
            os.makedirs(('runs/%dx%d_%1.2f'%(self.model.n_x,self.model.n_y,self.model.T)).replace('.','-'))
        pass

    def run(self, steps):
        steps = int(steps)
        energy_series = np.zeros(steps)
        time_series = np.zeros(steps)
        # The iterations ignore the first state. This is an arbitrary choice
        for i in range(steps):
            self.steps += 1
            step = self.move_selection()

            self.model.glass_update(step)
            self.model.energy_update(step)
            self.model.probability_update(step)

            time_series[i] = self.time_calculation(self.model.rate())
            energy_series[i] = self.model._energy

        pickle.dump(energy_series,open(('runs/%dx%d_%1.2f/energy_%d'%(self.model.n_x,self.model.n_y,self.model.T,self.next_run_number)).replace('.','-')+'.p','wb'))
        pickle.dump(time_series,open(('runs/%dx%d_%1.2f/time_%d'%(self.model.n_x,self.model.n_y,self.model.T,self.next_run_number)).replace('.','-')+'.p','wb'))
        self.next_run_number += 1
        return

    def move_selection(self):
        return self.model.move_selection()

    def time_calculation(self, rate):
        return -np.log(self.rng.uniform())/rate

    def get_mean(self):
        return np.sum(np.dot(self.energy_series,self.time_series))/np.sum(self.time_series)

    def get_sus(self):
        mean = self.get_mean()
        m2 = np.sum(self.time_series * self.e2_series) / np.sum(self.time_series)
        return (m2 - mean**2)/  self.model.T**2
