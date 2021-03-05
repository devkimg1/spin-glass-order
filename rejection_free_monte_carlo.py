'''

'''

import numpy as np

class RFMC():
    model: None
    energy_series: None
    time_series: None
    steps = int(0)

    def __init__(self, model):
        self.model = model
        self._p = np.copy(self.model.get_probabilities())
        pass

    def run(self, steps):
        steps = int(steps)
        self.energy_series = np.zeros(steps)
        self.time_series = np.zeros(steps)
        # The iterations ignore the first state. This is an arbitrary choice
        for i in range(steps):
            self.steps += 1
            index = self.move_selection()
            step = self.model.convert_index(index)

            #print(self.model._glass)
            #print(self.model.energy_change_array)
            #print(self.model._probabilities)

            self.model.glass_update(step)
            self.model.energy_update(step)
            self.model.probability_update(step)

            self.time_series[i] = self.time_calculation(self.model.rate())
            self.energy_series[i] = self.model._energy
        return

    def move_selection(self):
        # as it is, this is inefficient
        np.copyto(self._p, self.model.get_probabilities())
        self._p /= np.sum(self._p)
        '''total = np.sum(_p)*np.random.uniform()
        max = np.size(_p)
        for i in range(max):
            total -= _p[i]
            if total < 0:
                return i
        return max - 1'''
        choice = np.random.choice(np.size(self._p), p=self._p)
        return choice

    def time_calculation(self, rate):
        return -np.log(np.random.uniform())/rate

    def get_mean(self):
        return np.sum(np.dot(self.energy_series,self.time_series))/np.sum(self.time_series)
