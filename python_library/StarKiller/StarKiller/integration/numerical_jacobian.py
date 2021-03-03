import StarKillerMicrophysics as SKM

class NumericalJacobian(object):
    def __init__(self):
        self.NumericalJacModule = SKM.Numerical_Jac_Module()

    def jacobian(self, burn_state):
        # state is a BurnType object
        # evaluate the numerical jacobian
        self.NumericalJacModule.numerical_jac(burn_state.state, burn_state.jac)
