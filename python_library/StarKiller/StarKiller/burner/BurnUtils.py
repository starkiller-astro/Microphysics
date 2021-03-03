from __future__ import print_function
import numpy as np
from cycler import cycler
import matplotlib.pyplot as plt
from matplotlib import rc

from StarKiller.initialization import starkiller_initialize
from StarKiller.interfaces import EosType, BurnType
from StarKiller.network import Network
from StarKiller.eos import Eos
from StarKiller.integration import Integrator

rc('text', usetex=True)

class BurnerDriver(object):
    def __init__(self, probin_file):
        starkiller_initialize(probin_file)

        self.eos = Eos()
        self.network = Network()
        self.integrator = Integrator()

        self.history = BurnHistory()
        self.plotting = BurnPlotting()

        self.initial_burn = BurnType()
        self.initial_burn.state.rho = 0.0
        self.initial_burn.state.t = 0.0
        self.initial_burn.state.xn = np.zeros(self.nspec, dtype=np.float64)

        self.end_time = 0.0
        self.num_steps = 0
        self.dt = 0.0

    def list_species(self):
        print("Species in network:\n")
        for sname, short_sname in zip(self.network.species_names, self.network.short_species_names):
            print("{} ({})".format(sname, short_sname))

    def set_initial_density(self, dens):
        self.initial_burn.state.rho = dens

    def set_initial_temperature(self, temp):
        self.initial_burn.state.t = temp

    def set_initial_massfractions(self, xn):
        self.initial_burn.state.xn = xn[:]

    def set_initial_species(self, species_name, xspec):
        sname = species_name.lower()

        try:
            idx = self.network.species_map[sname]
        except ValueError:
            try:
                idx = self.network.species_map[self.network.long_short_species_map[sname]]
            except ValueError:
                print("ERROR: species {} is not in this network.".format(species_name))
                self.list_species()
                return

        self.initial_burn.xn[idx] = xspec

    def get_initial_state(self):
        return self.initial_burn

    def burn(self, end_time, num_steps):
        self.history = BurnHistory()
        self.end_time = end_time
        self.num_steps = num_steps
        self.dt = self.end_time/self.num_steps

        current_time = 0.0

        state_in = self.initial_burn.copy()
        
        for istep in range(self.num_steps):
            state_out = self.integrator.integrate(state_in, self.dt)
            current_time = current_time + self.dt
            self.history.store(state_out, current_time, self.dt, istep+1)
            state_in = state_out.copy()

        self.plotting.plot_burn_history(self.history)

    def eos(self, input, burn_state):
        eos_state = burn_state.to_eos_type()
        self.eos.evaluate(input, eos_state)
        burn_state.from_eos_type(eos_state)

    def rhs(self, burn_state):
        # Call the EOS in (r,t) mode and then evaluate the rhs
        self.eos(self.eos_type_module.eos_input_rt, burn_state)
        self.rhs_module.actual_rhs(burn_state)

    def jac(self, burn_state):
        # Call the EOS in (r,t) mode and then evaluate the jacobian
        self.eos(self.eos_type_module.eos_input_rt, burn_state)
        self.rhs_module.actual_jac(burn_state)

    def save(self, file_name):
        self.history.save(self.species_names, self.initial_burn, file_name)

    def get_temp_dot(self, burn_state):
        return burn_state.ydot[-2]

    def get_enuc_dot(self, burn_state):
        return burn_state.ydot[-1]

class BurnHistory(object):
    def __init__(self):
        self.nspec = SKM.Actual_Network().nspec
        self.xn = [[] for i in range(self.nspec)]
        self.t = []
        self.edot = []
        self.time = []
        self.step = []
        
    def append_xn(self, xn):
        for i, xi in enumerate(xn):
            self.xn[i].append(xn[i])

    def store(self, state, time, dt, step):
        self.append_xn(state.xn)
        self.t.append(state.t)
        self.edot.append(state.e/dt)
        self.time.append(time)
        self.step.append(step)

    def get_save_string(self, step, time, temp, enuc, xn):
        saveline = "   ".join([str(step), str(time), str(temp), str(enuc)] + [str(xi) for xi in xn])
        saveline = saveline + "\n"
        return saveline

    def get_species_vector(self, ixn):
        xvec = np.array([xi[ixn] for xi in self.xn])
        return xvec

    def save(self, spec_names, initial_state, filename):
        f = open("{}.dat".format(filename), "w")
        spec_names_string = "   ".join(spec_names)
        f.write("step   time   temperature   enucdot   " + spec_names_string + "\n")
        f.write(self.get_save_string(0, 0.0, initial_state.t, 0.0, initial_state.xn))
        for ii, (istep, itime, itemp, ienuc) in enumerate(zip(self.step, self.time, self.t, self.edot)):
            xvec = self.get_species_vector(ii)
            f.write(self.get_save_string(istep, itime, itemp, ienuc, xvec))
        f.close()


class BurnPlotting(object):
    def __init__(self):
        self.nspec = SKM.Actual_Network().nspec
        self.short_species_names = [SKM.Network().get_network_short_species_name(i+1).decode("ASCII").strip() for i in range(self.nspec)]
    
    def rgba_to_hex(self, rgba):
        r = int(rgba[0]*255.0)
        g = int(rgba[1]*255.0)
        b = int(rgba[2]*255.0)
        return '#{:02X}{:02X}{:02X}'.format(r,g,b)

    def plot_burn_history(self, history, logtime=True):
        plt.clf()
        
        if logtime:
            xlabel = '$\mathrm{Log_{10}~Time~(s)}$'
            xvec = np.log10(history.time)
            xlim = [np.log10(history.time[0]), np.log10(history.time[-1])]        
        else:
            xlabel = '$\mathrm{Time~(s)}$'
            xvec = history.time
            xlim = [history.time[0], history.time[-1]]

        # Get set of colors to use for abundances
        cm = plt.get_cmap('nipy_spectral')
        clist = [cm(1.0*i/self.nspec) for i in range(self.nspec)]
        hexclist = [self.rgba_to_hex(ci) for ci in clist]

        # Get the figure
        fig = plt.figure()
        fig.set_figheight(10.0)
        fig.set_figwidth(5.0)

        # Plot X vs. time    
        ax = fig.add_subplot(211)
        ax.set_prop_cycle(cycler('color', hexclist))
        for i in range(self.nspec):    
            ax.plot(xvec, np.log10(history.xn[i]), label=self.short_species_names[i])
        lgd = ax.legend(bbox_to_anchor=(1.15, 1.0), loc=2, borderaxespad=0.0)
        ax.set_xlim(xlim)
        plt.setp(ax.get_xticklabels(), visible=False)
        # plt.xlabel(xlabel)
        plt.ylabel('$\mathrm{Log_{10}~X}$')

        # Plot T, edot vs. time
        # Find where edot = 0
        def y_where_x_zero(y, x):
            yzero = []
            xiszero = False
            ylo = 0.0
            for yi, xi in zip(y, x):
                if xi == 0.0 and not xiszero:
                    xiszero = True
                    ylo = yi
                if xi != 0.0 and xiszero:
                    xiszero = False
                    yzero.append([ylo, yi])
            if xiszero:
                yzero.append([ylo, y[-1]])
            return yzero

        edotzero = y_where_x_zero(history.time, history.edot)
        axe = fig.add_subplot(212)
        axe.plot(xvec, np.log10(history.t), label='Temperature', color='red')
        lgd1 = axe.legend(bbox_to_anchor=(0.3, 1.02, 0.7, 1.02), loc='lower left',
                          ncol=1, mode='expand', borderaxespad=0.0)
        axe.set_xlabel(xlabel)
        axe.set_ylabel('$\mathrm{Log_{10}~T~(K)}$')
        axe.set_xlim(xlim)
        ax2 = axe.twinx()
        ax2.plot(xvec, np.log10(history.edot), label='E Gen Rate', color='blue')
        ax2.set_ylabel('$\mathrm{Log_{10}~\\dot{e}~(erg/g/s)}$')
        ax2.set_xlim(xlim)
        lgd2 = ax2.legend(bbox_to_anchor=(0.7, 1.02, 1.0, 1.02), loc='lower left',
                          ncol=1, borderaxespad=0.0)
        # hatch where edot=0
        for edz in edotzero:
            plt.axvspan(np.log10(edz[0]), np.log10(edz[1]), color='blue', fill=False,
                        linewidth=0, hatch='/', alpha=0.2)

        plt.savefig('burn.eps',
                    bbox_extra_artists=(lgd, lgd1, lgd2,), bbox_inches='tight')
        plt.savefig('burn.png', dpi=300,
                    bbox_extra_artists=(lgd, lgd1, lgd2,), bbox_inches='tight')

        plt.show()        
