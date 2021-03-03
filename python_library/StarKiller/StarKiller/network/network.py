import StarKillerMicrophysics as SKM
from StarKiller.interfaces import BurnType, EosType
from StarKiller.eos import Eos

class Network(object):
    def __init__(self):
        self.NetworkModule = SKM.Network()
        self.ActualNetworkModule = SKM.Actual_Network()
        self.NetworkPropertiesModule = SKM.Network_Properties()
        self.RHSModule = SKM.actual_rhs_module
        self.eos = Eos()

        self.name = self.NetworkModule.get_network_name().decode("ASCII").strip().lower()

        self.nspec = self.NetworkPropertiesModule.nspec

        self.zion = self.NetworkPropertiesModule.zion
        self.aion = self.NetworkPropertiesModule.aion
        self.aion_inv = self.NetworkPropertiesModule.aion_inv

        # These are python zero based indexes
        self.net_itemp = self.nspec
        self.net_ienuc = self.nspec + 1

        self.short_species_names = [self.NetworkModule.get_network_short_species_name(i+1).decode("ASCII").strip().lower() for i in range(self.nspec)]
        self.species_names = [self.NetworkModule.get_network_species_name(i+1).decode("ASCII").strip().lower() for i in range(self.nspec)]
        self.species_map = dict([(name, i) for i, name in enumerate(self.short_species_names)])
        self.long_short_species_map = dict([(long_name, short_name) for long_name, short_name in zip(self.species_names, self.short_species_names)])
        self.short_long_species_map = dict([(short_name, long_name) for long_name, short_name in zip(self.species_names, self.short_species_names)])

    def shorten_species(self, long_species_name):
        return self.long_short_species_map[long_species_name]

    def lengthen_species(self, short_species_name):
        return self.short_long_species_map[short_species_name]

    def rhs_to_x(self, ydot):
        xdot = ydot[:]
        xdot[:self.nspec] *= self.aion
        return xdot

    def jac_to_x(self, yjac):
        xjac = yjac[:,:]
        for j in range(self.nspec):
            for i in range(self.net_ienuc):
                xjac[j,i] *= self.aion[j]
                xjac[i,j] *= self.aion_inv[j]
        return xjac

    def rhs(self, burn_state):
        n_rhs = burn_state.state.n_rhs
        n_jac = burn_state.state.n_jac

        eos_state = burn_state.to_eos_type()
        self.eos.evaluate(eos_state.eos_input_rt, eos_state)
        burn_state.from_eos_type(eos_state)

        self.RHSModule.actual_rhs(burn_state.state, burn_state.ydot)

        # Restore n_rhs, n_jac without incrementing
        # since we want them to be valid for the
        # statistics of a preceding burn.
        burn_state.state.n_rhs = n_rhs
        burn_state.state.n_jac = n_jac


    def jacobian(self, burn_state):
        n_rhs = burn_state.state.n_rhs
        n_jac = burn_state.state.n_jac

        eos_state = burn_state.to_eos_type()
        self.eos.evaluate(eos_state.eos_input_rt, eos_state)
        burn_state.from_eos_type(eos_state)

        self.RHSModule.actual_jac(burn_state.state, burn_state.jac)

        # Restore n_rhs, n_jac without incrementing
        # since we want them to be valid for the
        # statistics of a preceding burn.
        burn_state.state.n_rhs = n_rhs
        burn_state.state.n_jac = n_jac

    @staticmethod
    def _initialize_safe():
        return True
