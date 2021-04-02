'''
Outlining the structure of classes required to use the kMC algorithm
This is designed so that in the kMC algorithm, the glass, energy, and
probabilities are synchronized. Therefore, the iterations are record current
energy, calculate time spent in current state, change state, update energy
and probabilities based on the changed state. Repeat
'''

import abc

class Model(abc.ABC):
    _glass: None
    _energy: None
    _probabilities: None

    @abc.abstractmethod
    def __init__(self):
        '''
        make function calls and what not to initialize
        _glass
        _energy
        _probabilities
        '''
        pass

    @abc.abstractmethod
    def total_energy(self):
        '''
        Calculate the energy of the system in the current state of glass.

        :rtype: None
        '''
        pass

    @abc.abstractmethod
    def energy_update(self, step):
        '''
        Provide a method of updating energy_change_array based on the MC step
        that is taken. When this is calculated, the state of glass is the
        CHANGED state.

        Optionally, this could just call total_energy().

        I recommend using the same data
        structure as _probabilities to record an _energy_change_array. This
        allows for making this step a memory call to _energy_change_array and
        updating the elements of _energy_change_array to fit the new state.

        :rtype: None
        '''
        pass

    @abc.abstractmethod
    def total_probability(self):
        '''
        Provide a method of calculating the probability of all possible moves
        for the model.

        I recommend having a separate function which implements Metropolis on to
        a specific possible step. Then total_probability_array will iterate that
        function through all possible steps while filling out _probabilities.

        :rtype: None
        '''
        pass

    @abc.abstractmethod
    def probability_update(self, step):
        '''
        Provide a method of updating _probabilities based on the MC step
        that is taken.

        Optionally, this could just call total_probability().

        :rtype: None
        '''
        pass

    @abc.abstractmethod
    def glass_update(self, step):
        '''
        Provide a method of updating _glass based on the MC step that is taken

        :rtype: None
        '''
        pass

    @abc.abstractmethod
    def rate(self):
        '''
        I don't fully understand this concept either. It is the sum of the
        probability of possible moves. Therefore, the returned value will be
        less than the number of possible moves and should be a float of sorts.

        :rtype: double
        '''
        pass

    @abc.abstractmethod
    def move_selection(self):
        '''
        Return the selected move
        '''
        pass
