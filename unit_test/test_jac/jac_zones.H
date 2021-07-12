#ifndef _react_zones_H_
#define _react_zones_H_

#include <variables.H>
#include <network.H>
#include <eos_type.H>
#include <eos.H>
#include <burn_type.H>
#include <burner.H>
#include <extern_parameters.H>
#include <ArrayUtilities.H>
#include <variables.H>
#include <numerical_jacobian.H>

#ifndef NETWORK_HAS_CXX_IMPLEMENTATION
#include <fortran_to_cxx_actual_rhs.H>
#endif

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
bool do_jac (int i, int j, int k, Array4<Real> const& state, const plot_t p)
{

    burn_t burn_state;

    burn_state.rho = state(i, j, k, p.irho);
    burn_state.T = state(i, j, k, p.itemp);
    for (int n = 0; n < NumSpec; ++n) {
        burn_state.xn[n] = state(i, j, k, p.ispec_old + n);
    }

    normalize_abundances_burn(burn_state);

    eos_t eos_state;

    burn_to_eos(burn_state, eos_state);

    eos(eos_input_rt, eos_state);

    eos_to_burn(eos_state, burn_state);

    // the integrator doesn't actually care about the initial internal
    // energy.
    burn_state.e = 0.0_rt;

    burn_state.self_heat = true;

    JacNetArray2D jac_analytic;
    JacNetArray2D jac_numerical;

    actual_jac(burn_state, jac_analytic);

    // we need to convert this to be in terms of X instead of Y

    for (int j = 1; j <= NumSpec; ++j) {
        for (int i = 1; i <= neqs; ++i) {
            jac_analytic(j, i) *= aion[j-1];
            jac_analytic(i, j) *= aion_inv[j-1];
        }
    }

    jac_info_t jac_info;
    jac_info.h = 1.e-5_rt;  // timestep really doesn't make sense here

    numerical_jac(burn_state, jac_info, jac_numerical);

    int n = 0;
    for (int jj = 1; jj <= neqs; ++jj) {
        for (int ii = 1; ii <= neqs; ++ii) {
            state(i, j, k, p.ijac + n) = std::abs(jac_numerical(ii, jj) - jac_analytic(ii, jj));
            if (jac_analytic(ii, jj) != 0.0_rt) {
                state(i, j, k, p.ijac + n) /= std::abs(jac_analytic(ii, jj));
            }
            n++;
        }
    }

    return true;

}

#endif
