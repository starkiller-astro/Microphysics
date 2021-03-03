import StarKillerMicrophysics as SKM
from StarKiller.interfaces import EosType
import numpy as np

class BurnType(object):
    def __init__(self):
        self.BurnTypeModule = SKM.Burn_Type_Module()
        self.neqs = self.BurnTypeModule.neqs
        self.state = self.BurnTypeModule.burn_t()

        self.ydot = np.zeros(self.neqs)
        self.jac = np.zeros((self.neqs, self.neqs))

        self.ydot = np.asfortranarray(self.ydot)
        self.jac = np.asfortranarray(self.jac)

    def to_eos_type(self):
        # Return an EosType object from this BurnType
        eos_state = EosType()
        self.BurnTypeModule.burn_to_eos(self.state, eos_state.state)
        return eos_state
        
    def from_eos_type(self, eos_state):
        # Given an EosType object, set this BurnType data
        self.BurnTypeModule.eos_to_burn(eos_state.state, self.state)

    def copy(self):
        # Return a deep copy of this object
        state_copy = BurnType()

        # Copy the Fortran derived type data
        self.BurnTypeModule.copy_burn_t(state_copy.state, self.state)

        # Copy the python object data to fill ydot, jac
        state_copy.ydot[:] = self.ydot[:]
        state_copy.jac[:,:] = self.jac[:,:]

        return state_copy

