#ifndef NUMERICAL_JACOBIAN_H
#define NUMERICAL_JACOBIAN_H

#include <network.H>
#include <burn_type.H>
#include <extern_parameters.H>
#include <actual_rhs.H>

///
/// Compute the numerical Jacobian of the reactive system, dydot/dy,
/// where ydot = {dX/dt, denuc/dt}, and y = {X, e}.
///
/// Note the analytic Jacobian usually works in terms of molar
/// fractions, Y, instead of mass fractions, X, but our integrators
/// always work in terms of X, so the numerical Jacobian is expressed
/// in terms of X.
///
AMREX_GPU_HOST_DEVICE
void numerical_jac(burn_t& state, JacNetArray2D& jac)
{


    // the choice of eps should be ~ sqrt(eps), where eps is machine epsilon.
    // this balances truncation vs. roundoff error in the differencing

    constexpr Real eps = 1.e-8_rt;

    for (int i = 1; i <= neqs; i++) {
        for (int j = 1; j <= neqs; j++) {
            jac(i,j) = 0.0_rt;
        }
    }

    YdotNetArray1D ydotp;
    YdotNetArray1D ydotm;

    if (centered_diff_jac) {
        burn_t state_delp = state;
        burn_t state_delm = state;

        // species derivatives

        for (int n = 1; n <= NumSpec; n++) {
            // perturb species

            for (int q = 0; q < NumSpec; q++) {
                state_delp.xn[q] = state.xn[q];
            }

            // xn is 0-based
            state_delp.xn[n-1] *= (1.0_rt + eps);

            actual_rhs(state_delp, ydotp);

            // We integrate X, so convert from the Y we got back from the RHS

            for (int q = 1; q <= NumSpec; q++) {
                // note ydotp is 1-based, but aion is 0-based
                ydotp(q) *= aion[q-1];
            }

            for (int q = 0; q < NumSpec; q++) {
                state_delm.xn[q] = state.xn[q];
            }

            // xn is 0-based
            state_delm.xn[n-1] *= (1.0_rt - eps);

            actual_rhs(state_delm, ydotm);

            for (int q = 1; q <= NumSpec; q++) {
                // note ydotm is 1-based, but aion is 0-based
                ydotm(q) *= aion[q-1];
            }

            for (int m = 1; m <= neqs; m++) {
                jac(m,n) = 0.5_rt * (ydotp(m) - ydotm(m)) / (eps * state.xn[n-1]);
            }
        }


        // e derivative

        for (int q = 0; q < NumSpec; q++) {
            state_delp.xn[q] = state.xn[q];
        }
        state_delp.e *= (1.0_rt + eps);

        eos(eos_input_re, state_delp);

        actual_rhs(state_delp, ydotp);

        for (int q = 1; q <= NumSpec; q++) {
            ydotp(q) *= aion[q-1];
        }

        for (int q = 0; q < NumSpec; q++) {
            state_delm.xn[q] = state.xn[q];
        }

        state_delm.e *= (1.0_rt - eps);

        eos(eos_input_re, state_delm);

        actual_rhs(state_delm, ydotm);

        for (int q = 1; q <= NumSpec; q++) {
            ydotm(q) *= aion[q-1];
        }

        for (int m = 1; m <= neqs; m++) {
            jac(m, net_ienuc) = 0.5_rt * (ydotp(m) - ydotm(m)) / (eps * state.e);
        }


    } else {
        burn_t state_delp = state;

        // default

        actual_rhs(state, ydotm);

        for (int q = 1; q <= NumSpec; q++) {
            ydotm(q) *= aion[q-1];
        }

        // species derivatives

        for (int n = 1; n <= NumSpec; n++) {
            // perturb species -- we send in X, but ydot is in terms
            // of dY/dt, not dX/dt

            for (int q = 0; q < NumSpec; q++) {
                state_delp.xn[q] = state.xn[q];
            }

            Real h = eps * std::abs(state.xn[n-1]);
            if (h == 0) {
                h = eps;
            }

            state_delp.xn[n-1] += h;

            actual_rhs(state_delp, ydotp);

            // We integrate X, so convert from the Y we got back from the RHS

            for (int q = 1; q <= NumSpec; q++) {
                ydotp(q) *= aion[q-1];
            }

            for (int m = 1; m <= neqs; m++) {
                jac(m, n) = (ydotp(m) - ydotm(m)) / h;
            }
        }

        // e derivative

        for (int q = 0; q < NumSpec; q++) {
            state_delp.xn[q] = state.xn[q];
        }

        Real h = eps * std::abs(state.e);
        if (h == 0) {
            h = eps;
        }

        state_delp.e += h;

        eos(eos_input_re, state_delp);

        actual_rhs(state_delp, ydotp);

        for (int q = 1; q <= NumSpec; q++) {
            ydotp(q) *= aion[q-1];
        }

        for (int m = 1; m <= neqs; m++) {
            jac(m, net_ienuc) = (ydotp(m) - ydotm(m)) / h;
        }

    }
}
#endif
