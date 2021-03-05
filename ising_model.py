import numpy as np
from base_model import Model

class Ising2D(Model):
    def __init__(self, x, y, T):
        self.T = T
        self.n_x = int(x)
        self.n_y = int(y)
        self._glass = np.random.randint(2, size = (self.n_x, self.n_y))*2 - 1
        self._generate_energy_change_array()
        self.total_energy()
        self.total_probability()

    def get_probabilities(self):
        x = np.ravel(self._probabilities)
        return x

    def convert_index(self, index):
        return [int(index/self.n_y), int(index%self.n_y)]

    def total_energy(self):
        e = 0
        for i in range(self.n_x):
            for j in range(self.n_y):
                e += self._glass[i,j] * (self._glass[(i+1)%self.n_x,j] + self._glass[i,(j+1)%self.n_y])
        self._energy = int(-e)
        return None

    def _generate_energy_change_array(self):
        self.energy_change_array = np.zeros((self.n_x,self.n_y))
        for i in range(self.n_x):
            for j in range(self.n_y):
                self.energy_change_array[i,j] = self._energy_change(i,j)

    def energy_update(self, step):
        x = step[0]
        y = step[1]

        self._energy += self.energy_change_array[x,y]

        # recompute energies around x and y
        a = np.array([-1, 0, 1], dtype = "int")
        for i in (x + a)%self.n_x:
            for j in (y + a)%self.n_y:
                self.energy_change_array[i, j] = self._energy_change(i, j)
        return

    def _energy_change(self, i, j):
        latt = self._glass
        s = latt[(i+1)%self.n_x,j] + latt[(i-1)%self.n_x, j] + latt[i,(j+1)%self.n_y] + latt[i,(j-1)%self.n_y]

        return int(2*s*latt[i,j])

    def total_probability(self):
        self._probabilities = np.exp(-self.energy_change_array/self.T)
        self._probabilities[self._probabilities > 1] = 1
        return None

    def probability_update(self, step):
        # recompute probabilities around x and y
        x = step[0]
        y = step[1]
        a = np.array([-1, 0, 1], dtype = "int")
        for i in (x + a)%self.n_x:
            for j in (y + a)%self.n_y:
                self._probabilities[i,j] = np.min([1,np.exp(-self.energy_change_array[i, j]/self.T)])
        return

    def glass_update(self, step):
        x = step[0]
        y = step[1]
        self._glass[x,y] = -self._glass[x,y]
        return

    def rate(self):
        return np.sum(self._probabilities)

    def get_glass(self):
        return self._glass
