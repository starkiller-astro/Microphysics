#include <eos_composition.H>

AMREX_GPU_HOST_DEVICE
eos_xderivs_t composition_derivatives(eos_t& state) {

  eos_xderivs_t state_xderivs;

#ifdef EXTRA_THERMO
  for (int n = 0; n < NumSpec; n++) {
    state_xderivs.dpdX[n] =
      state.dpdA * (state.abar / aion[n]) * (aion[n] - state.abar) +
      state.dpdZ * (state.abar / aion[n]) * (zion[n] - state.zbar);

    state_xderivs.dedX[n] =
      state.dedA * (state.abar / aion[n]) * (aion[n] - state.abar) +
      state.dedZ * (state.abar / aion[n]) * (zion[n] - state.zbar);

    if (state.dpdr != 0.0) {

      state_xderivs.dhdX[n] = state_xderivs.dedX[n]
            + (state.p / (state.rho*state.rho) - state.dedr)
            * state_xderivs.dpdX[n] / state.dpdr;
    }
  }

#endif

  return state_xderivs;
}
