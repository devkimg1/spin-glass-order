import numpy as np
from base_model import Model
from numba import njit

@njit
def _energy_update(step, n_x, n_y, energy_change_array, glass):
    x = step[0]
    y = step[1]

    # recompute energies around x and y
    a = np.array([-1, 0, 1])
    for i in (x + a)%n_x:
        for j in (y + a)%n_y:
            energy_change_array[i, j] = energy_change(i, j, glass, n_x, n_y)
    return

@njit
def energy_change(i,j,glass, n_x, n_y):
    s = glass[(i+1)%n_x,j] + glass[(i-1)%n_x, j] + glass[i,(j+1)%n_y] + glass[i,(j-1)%n_y]
    return int(2*s*glass[i,j])

@njit
def _probability_update(step, n_x,n_y,T,probabilities, energy_change_array, p_row, p_t):
    # recompute probabilities around x and y
    x = step[0]
    y = step[1]
    a = np.array([-1, 0, 1])
    '''for i in (x + a)%n_x:
        for j in (y + a)%n_y:
            probabilities[i,j] = np.min(np.array([1.,np.exp(-float(energy_change_array[i, j])/T)]))
    p_row[(x + a)%n_x] = np.sum(probabilities[(x + a)%n_x, :], axis = 1)
    p_t = np.sum(p_row)'''
    p_t -= np.sum(p_row[(x + a)%n_x])
    for i in (x + a)%n_x:
        for j in (y + a)%n_y:
            p_row[i] -= probabilities[i,j]
            probabilities[i,j] = np.min(np.array([1.,np.exp(-float(energy_change_array[i, j])/T)]))
            p_row[i] += probabilities[i,j]
    p_t += np.sum(p_row[(x + a)%n_x])
    return p_t

@njit
def _choice(p, r):
    i = int(0)
    c = 0.
    for k in p:
        c += k
        if c > r:
            return i
        i += 1
    return i-1

class Ising2D(Model):
    def __init__(self, x, y, T):
        self.T = T
        self.n_x = int(x)
        self.n_y = int(y)
        self._glass = np.random.randint(2, size = (self.n_x, self.n_y))*2 - 1
        self.rng = np.random.default_rng()
        self._generate_energy_change_array()
        self.total_energy()
        self.total_probability()
        self._p_t = np.sum(self._probabilities)
        self._p_row = np.sum(self._probabilities, axis = 1)

    def move_selection(self):
        r = self.rng.uniform()
        i = _choice(self._p_row, self._p_t * r)
        r = self.rng.uniform()
        j = _choice(self._probabilities[i], self._p_row[i] * r)
        return np.array([i,j], dtype = 'int')

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
        _energy_update(step, self.n_x, self.n_y, self.energy_change_array, self._glass)
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
        self._p_t = _probability_update(step, self.n_x, self.n_y, self.T, self._probabilities, self.energy_change_array, self._p_row, self._p_t)

    def glass_update(self, step):
        x = step[0]
        y = step[1]
        self._glass[x,y] = -self._glass[x,y]
        return

    def rate(self):
        return self._p_t

    def get_glass(self):
        return self._glass
