'''

'''

import numpy as np

class RFMC():
    model: None
    energy_series: None
    time_series: None
    steps = int(0)

    def __init__(self, model, steps):
        self.steps = int(steps)
        self.energy_series = np.array(np.zeros(self.steps))
        self.time_series = np.array(np.zeros(self.steps))
        self.model = model
        pass

    def run(self):
        # The iterations ignore the first state. This is an arbitrary choice
        for i in range(self.steps):
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
        _p = self.model.get_probabilities()
        return np.random.choice(np.array(list(range(np.size(self.model.get_probabilities())))), p=_p/np.sum(_p))

    def time_calculation(self, rate):
        return -np.log(np.random.uniform())/rate

    def get_mean(self):
        return np.sum(np.dot(self.energy_series,self.time_series))/np.sum(self.time_series)
