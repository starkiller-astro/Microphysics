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

#ifdef NETWORK_SOLVER
#include <actual_matrix.H>
#endif

#ifndef NETWORK_HAS_CXX_IMPLEMENTATION
#include <fortran_to_cxx_actual_rhs.H>
#endif

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
bool do_rhs (int i, int j, int k, Array4<Real> const& state)
{

    burn_t burn_state;

    burn_state.rho = state(i, j, k, p_irho);
    burn_state.T = state(i, j, k, p_itemp);
    for (int n = 0; n < NumSpec; ++n) {
        burn_state.xn[n] = state(i, j, k, p_ispec_old + n);
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

    Array1D<Real, 1, neqs> ydot;

#ifdef NETWORK_SOLVER
    SparseMatrix jac;
#else
    ArrayUtil::MathArray2D<1, neqs, 1, neqs> jac;
#endif

    actual_rhs(burn_state, ydot);
    actual_jac(burn_state, jac);

    for (int n = 1; n <= NumSpec; ++n) {
        state(i, j, k, p_ispec + n - 1) = ydot(n);
        state(i, j, k, p_itemp_dot) = ydot(net_itemp);
        state(i, j, k, p_ienuc_dot) = ydot(net_ienuc);
    }

    int n = 0;
    for (int jj = 1; jj <= neqs; ++jj) {
        for (int ii = 1; ii <= neqs; ++ii) {
            state(i, j, k, p_ijac + n) = jac.get(ii, jj);
            n++;
        }
    }

    return true;

}

#endif
