import StarKillerMicrophysics as SKM

class EosType(object):
    def __init__(self):
        self.EosTypeModule = SKM.Eos_Type_Module()
        self.EosCompositionModule = SKM.Eos_Composition_Module()

        self.eos_input_rt = self.EosTypeModule.eos_input_rt
        self.eos_input_rh = self.EosTypeModule.eos_input_rh
        self.eos_input_tp = self.EosTypeModule.eos_input_tp
        self.eos_input_rp = self.EosTypeModule.eos_input_rp
        self.eos_input_re = self.EosTypeModule.eos_input_re
        self.eos_input_ps = self.EosTypeModule.eos_input_ps
        self.eos_input_ph = self.EosTypeModule.eos_input_ph
        self.eos_input_th = self.EosTypeModule.eos_input_th

        self.state = self.EosTypeModule.eos_t()

        self.xderivs = self.EosCompositionModule.eos_xderivs_t()

    def update_composition(self):
        self.EosCompositionModule.composition(self.state)
        self.xderivs = self.EosCompositionModule.composition_derivatives(self.state)
