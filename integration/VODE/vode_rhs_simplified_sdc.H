#ifndef VODE_RHS_SIMPLIFIED_SDC_H
#define VODE_RHS_SIMPLIFIED_SDC_H

#include <network.H>
#include <burn_type.H>
#include <extern_parameters.H>

#include <vode_type_simplified_sdc.H>
#include <vode_jac_util_simplified_sdc.H>
#ifdef NETWORK_HAS_CXX_IMPLEMENTATION
#include <actual_network.H>
#include <actual_rhs.H>
#else
#include <fortran_to_cxx_actual_rhs.H>
#endif
#include <numerical_jacobian.H>

// The f_rhs routine provides the right-hand-side for the DVODE solver.
// This is a generic interface that calls the specific RHS routine in the
// network you're actually using.

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
void rhs(const Real time, burn_t& state, dvode_t& vode_state, RArray1D& ydot)
{

    // update rho, rho*u, etc.

    fill_unevolved_variables(time, state, vode_state);

    // ensure that the mass fractions are valid -- only vode_state is
    // updated here

    clean_state(time, state, vode_state);

    // convert to the burn_t -- this does an EOS call to get T
    // and populates the (burn_t) state

    vode_to_burn(time, vode_state, state);

    // make sure that the temperature is valid

    if (state.T <= EOSData::mintemp || state.T >= MAX_TEMP) {

        for (int n = 1; n <= VODE_NEQS; ++n) {
            ydot(n) = 0.0_rt;
        }

        return;

    }

    // call the specific network to get the RHS

    YdotNetArray1D ydot_react = {0};

    actual_rhs(state, ydot_react);

    // apply fudge factor:

    if (react_boost > 0.0_rt) {
        // the correct size here is the size of the reactive state RHS,
        // not the SDC state RHS (neqs, not VODE_NEQS)
        for (int n = 1; n <= neqs; ++n) {
            ydot_react(n) *= react_boost;
        }
    }

    // convert back to the vode type -- this will add the advective terms

    rhs_to_vode(time, state, ydot_react, vode_state, ydot);

}



// Analytical Jacobian

template<class MatrixType>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
void jac (burn_t& state, dvode_t& vode_state, MatrixType& pd)
{
    // NOTE: the time at which to evaluate the Jacobian is not
    // explicitly passed. VODE always evaluates the analytic
    // Jacobian at vode_state.tn, and we pass vode_state.


    // update rho, rho*u, ... in the burn_t state

    fill_unevolved_variables(vode_state.tn, state, vode_state);

    // convert to the burn_t -- this does an EOS call to get T
    // and populates the (burn_t) state

    vode_to_burn(vode_state.tn, vode_state, state);


    if (state.T <= EOSData::mintemp || state.T >= MAX_TEMP) {

        for (int j = 1; j <= VODE_NEQS; ++j) {
            for (int i = 1; i <= VODE_NEQS; ++i) {
                pd(i,j) = 0.0_rt;
            }
        }

        return;

    }

    // Call the specific network routine to get the Jacobian.

    JacNetArray2D jac_react;

    if (jacobian == 3) {
        numerical_jac(state, jac_react);
    } else {
        actual_jac(state, jac_react);

        // The Jacobian from the nets is in terms of dYdot/dY, but we want
        // it was dXdot/dX, so convert here.
        for (int n = 1; n <= NumSpec; n++) {
            for (int m = 1; m <= neqs; m++) {
                jac_react(n,m) = jac_react(n,m) * aion[n-1];
            }
        }

        for (int m = 1; m <= neqs; m++) {
            for (int n = 1; n <= NumSpec; n++) {
                jac_react(m,n) = jac_react(m,n) * aion_inv[n-1];
            }
        }

    }

    // apply fudge factor:

    if (react_boost > 0.0_rt) {
        for (int j = 1; j <= neqs; ++j) {
            for (int i = 1; i <= neqs; ++i) {
                jac_react.mul(i, j, react_boost);
            }
        }
    }

    // convert to the system we are using

    jac_to_vode(vode_state.tn, state, jac_react, vode_state, pd);

}
#endif
