import StarKillerMicrophysics as SKM
from StarKiller.interfaces import EosType
import os

class Eos(object):
    def __init__(self):
        self.EosModule = SKM.Eos_Module()
        self.name = self.EosModule.get_eos_name().decode("ASCII").strip().lower()

    def evaluate(self, input_mode, input_state, use_raw_inputs=False):
        self.EosModule.eos(input_mode, input_state.state, use_raw_inputs)
        input_state.update_composition()

    @staticmethod
    def _initialize_safe():
        eos = Eos()
        if (eos.name == "helmholtz"):
            # check to see if the Helmholtz table is in the
            # current working directory
            try:
                assert(os.path.isfile("helm_table.dat"))
            except AssertionError:
                # if we cannot find the table, attempt to symlink it in
                # by reading the MICROPHYSICS_HOME environment variable
                print("helm_table.dat file or symlink is missing from the working directory")
                try:
                    microphysics_home = os.environ.get("MICROPHYSICS_HOME")
                    helm_source = os.path.join(microphysics_home, "EOS/helmholtz/helm_table.dat")
                    helm_dest   = os.path.join(os.getcwd(), "helm_table.dat")
                    os.symlink(helm_source, helm_dest)
                except:
                    print("unable to link helm_table.dat from MICROPHYSICS_HOME")
                    raise
                else:
                    print("linked helm_table.dat from MICROPHYSICS_HOME={}".format(microphysics_home))
                    pass
            except:
                print("could not verify if helm_table.dat is present")
                raise
        return True
