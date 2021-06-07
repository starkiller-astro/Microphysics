#ifndef actual_integrator_H
#define actual_integrator_H

#include <be_type.H>
#include <network.H>
#include <actual_network.H>
#include <actual_rhs.H>
#include <burn_type.H>
#include <eos_type.H>
#include <eos.H>
#include <extern_parameters.H>
#include <linpack.H>
#include <numerical_jacobian.H>

#define VODELIKE_ERROR 1

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
void clean_state (burn_t& state)
{
    // Renormalize the abundances.

    normalize_abundances_burn(state);

    // Evaluate the EOS to get T from e.

    if (call_eos_in_rhs) {
        eos(eos_input_re, state);
    }

    // Ensure that the temperature always stays within reasonable limits.

    state.T = amrex::min(MAX_TEMP, amrex::max(state.T, EOSData::mintemp));

}


AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
void evaluate_rhs(burn_t state, Array1D<Real, 1, BE_NEQS>& ydot)
{

    actual_rhs(state, ydot);

    // convert species to be in terms of mass fractions

    for (int ii = 1; ii <= NumSpec; ++ii) {
        ydot(ii) *= aion[ii-1];
    }

    // if we are not integrating e, then zero this out

    if (!integrate_energy) {
        ydot(net_ienuc) = 0.0_rt;
    }
}


AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
Real initial_dt (burn_t& state, const Real tend, Array1D<Real, 1, BE_NEQS>& ydot)
{
    // this is a version of the initial timestep estimator from VODE

    Array1D<Real, 1, BE_NEQS> y;

    for (int n = 1; n <= NumSpec; n++) {
        y(n) = state.xn[n-1];
    }
    y(net_ienuc) = state.e;

    // initial lower and upper bounds on the timestep

    Real hL = 100.0_rt * std::numeric_limits<Real>::epsilon() * tend;
    Real hU = 0.1_rt * tend;

    // initial guess for the itteraction

    Real h = std::sqrt(hL * hU);
    Real h_old = 10.0_rt * h;

    // Iterate on ddydtt = (RHS(t + h, y + h * dydt) - dydt) / h

    Array1D<Real, 1, BE_NEQS> ewt;
    Array1D<Real, 1, BE_NEQS> y_temp;
    Array1D<Real, 1, BE_NEQS> ydot_temp;
    Array1D<Real, 1, BE_NEQS> ddydtt;

    for (int n = 1; n <= 4; n++) {

       h_old = h;

       // Get the error weighting -- this is similar to VODE's dewset
       // routine

       for (int ii = 1; ii <= NumSpec; ii++) {
           ewt(ii) = rtol_spec * std::abs(y(ii)) + atol_spec;
       }
       ewt(net_ienuc) = rtol_enuc * std::abs(y(net_ienuc)) + atol_enuc;

       // Construct the trial point.

       for (int ii = 1; ii <= BE_NEQS; ii++) {
           y_temp(ii) = y(ii) + h * ydot(ii);
       }

       // Call the RHS, then estimate the finite difference.

       burn_t state_temp = state;
       for (int ii = 1; ii <= NumSpec; ii++) {
           state_temp.xn[ii-1] = y_temp(ii);
       }
       state_temp.e = y_temp(net_ienuc);

       clean_state(state_temp);

       evaluate_rhs(state_temp, ydot_temp);

       for (int ii = 1; ii <= BE_NEQS; ii++) {
           ddydtt(ii) = (ydot_temp(ii) - ydot(ii)) / h;
       }

       Real yddnorm = 0.0_rt;
       for (int ii = 1; ii <= BE_NEQS; ii++) {
           yddnorm += std::pow(ddydtt(ii) * ewt(ii), 2);
       }
       yddnorm = std::sqrt(yddnorm / BE_NEQS);

       if (yddnorm*hU*hU > 2.0_rt) {
          h = std::sqrt(2.0_rt / yddnorm);
       } else {
          h = std::sqrt(h * hU);
       }

       if (h_old < 2.0_rt * h && h_old > 0.5_rt * h) {
           break;
       }

    }

    // Save the final timestep, with a bias factor.

    Real dt = h / 2.0_rt;
    dt = amrex::min(amrex::max(h, hL), hU);

    dt = amrex::min(dt, ode_max_dt);

    return dt;
}


AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
void initialize_state (burn_t& state)
{
    // We assume that (rho, T) coming in are valid, do an EOS call
    // to fill the rest of the thermodynamic variables.

    eos(eos_input_rt, state);

    state.self_heat = true;

    state.success = true;

    state.n_rhs = 0;
}


///
/// update state.xn[] and state.e through a timestep dt
/// state is updated in place -- if we are unsuccessful, we reset it
/// we return an error code
///
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
int single_step (burn_t& state, const Real dt)
{

    clean_state(state);

    // pack the initial state into a vector -- we need this for the
    // RHS of the linear system and to reset the state if we are
    // unsuccessful here

    Array1D<Real, 1, BE_NEQS> y;
    for (int n = 1; n <= NumSpec; n++) {
        y(n) = state.xn[n-1];
    }
    y(net_ienuc) = state.e;

    int ierr = BE_SUCCESS;
    bool converged = false;

    // create our current guess for the solution -- just as a first
    // order explicit prediction

    Array1D<Real, 1, neqs> ydot;

    evaluate_rhs(state, ydot);

    state.n_rhs += 1;

    Array1D<Real, 1, BE_NEQS> y_np1;

    for (int n = 1; n <= BE_NEQS; n++) {
        y_np1(n) = y(n) + dt * ydot(n);
    }

    // Newton loop

    for (int iter = 1; iter <= max_iter; iter++) {

        // work with the current guess

        for (int n = 0; n < NumSpec; n++) {
            state.xn[n] = y_np1(n+1);
        }
        state.e = y_np1(net_ienuc);

        clean_state(state);

        // construct the Jacobian

        JacNetArray2D jac;

        if (jacobian == 1) {
            actual_jac(state, jac);

            // We integrate X, not Y
            for (int j = 1; j <= NumSpec; ++j) {
                for (int i = 1; i <= neqs; ++i) {
                    jac(j, i) *= aion[j-1];
                    jac(i, j) *= aion_inv[j-1];
                }
            }

        } else {
            numerical_jac(state, jac);
        }

        if (!integrate_energy) {
            for (int j = 1; j <= BE_NEQS; j++) {
                jac(net_ienuc, j) = 0.0_rt;
            }
        }

        // construct the matrix for the linear system
        // (I - dt J) dy^{n+1} = rhs

        for (int m  = 1; m <= BE_NEQS; m++) {
            for (int n = 1; n <= BE_NEQS; n++) {
                jac(m, n) *= -dt;
                if (m == n) {
                    jac(m, n) = 1.0_rt + jac(m, n);
                }
            }
        }

        // get the ydots for our current guess of y

        evaluate_rhs(state, ydot);

        state.n_rhs += 1;

        // construct the RHS of our linear system

        Array1D<Real, 1, BE_NEQS> b;
        for (int n = 1; n <= BE_NEQS; n++) {
            b(n) = y(n) - y_np1(n) + dt * ydot(n);
        }

        // solve the linear system

        int ierr_linpack;
        IArray1D pivot;
        dgefa<BE_NEQS>(jac, pivot, ierr_linpack);

        if (ierr_linpack != 0) {
            ierr = BE_LU_DECOMPOSITION_ERROR;
            break;
        }

        dgesl<BE_NEQS>(jac, pivot, b);

        // update our current guess for the solution

        for (int n = 1; n <= BE_NEQS; n++) {
            y_np1(n) += b(n);
        }

        // check to see if we converged
        // we compute the norms

        Real y_norm = 0.0;
        Real b_norm = 0.0;
        for (int n = 1; n <= BE_NEQS; n++) {
            y_norm += y_np1(n) * y_np1(n);
            b_norm += b(n) * b(n);
        }
        y_norm = std::sqrt(y_norm / BE_NEQS);
        b_norm = std::sqrt(b_norm / BE_NEQS);

        if (b_norm < tol * y_norm) {
            converged = true;
            break;
        }

    }

    // we are done iterating -- did we converge?

    if (! converged) {

        if (ierr == BE_SUCCESS) {

            // if we didn't set another error, then we probably ran
            // out of iterations, so set nonconvergence

            ierr = BE_NONCONVERGENCE;
        }

        // reset the state

        for (int n = 0; n < NumSpec; n++) {
            state.xn[n] = y(n+1);
        }
        state.e = y(net_ienuc);

    } else {

        // update the state to the latest values

        for (int n = 0; n < NumSpec; n++) {
            state.xn[n] = y_np1(n+1);
        }
        state.e = y_np1(net_ienuc);

    }

    return ierr;

}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
void actual_integrator (burn_t& state, const Real dt)
{
    // We assume that (rho, T) coming in are valid, do an EOS call
    // to fill the rest of the thermodynamic variables.

    initialize_state(state);

    Real T_in = state.T;
    Real e_in = state.e;
    Real xn_in[NumSpec];
    for (int n = 0; n < NumSpec; ++n) {
        xn_in[n] = state.xn[n];
    }

    state.success = true;

    state.n_rhs = 0;

    Real t = 0.0;

    // When checking the integration time to see if we're done,
    // be careful with roundoff issues.

    const Real timestep_safety_factor = 1.0e-12_rt;

    // pack the solution into a vector

    int num_timesteps = 0;

    // estimate the timestep

    Array1D<Real, 1, neqs> ydot;
    evaluate_rhs(state, ydot);

    state.n_rhs += 1;

    Real dt_sub = initial_dt(state, dt, ydot);

    // main timestepping loop

    while (t < (1.0_rt - timestep_safety_factor) * dt &&
           num_timesteps < ode_max_steps) {

        // don't go too far

        if (t + dt_sub > dt) {
            dt_sub = dt - t;
        }

        // our strategy is to take 2 steps at dt/2 and one at dt and
        // to compute the error from those


        // try to take a step dt

        burn_t state_coarse_dt = state;

        // first do 2 (fine) dt/2 steps

        int ierr = single_step(state, dt_sub/2);
        if (ierr == BE_SUCCESS) {
            ierr = single_step(state, dt_sub/2);
        }

        // now that single (coarse) dt step

        if (ierr == BE_SUCCESS) {
            ierr = single_step(state_coarse_dt, dt_sub);
        }

#ifdef VODELIKE_ERROR
        // define a weight for each variable to use in checking the error

        Array1D<Real, 1, BE_NEQS> w;
        for (int n = 1; n <= NumSpec; n++) {
            w(n) = 1.0_rt / (rtol_spec * std::abs(state.xn[n-1]) + atol_spec);
        }
        w(net_ienuc) = 1.0_rt / (rtol_enuc * std::abs(state.e) + atol_enuc);

        // now look for w |y_fine - y_coarse| < 1

        Real rel_error = 0.0_rt;
        for (int n = 1; n <= NumSpec; n++) {
            rel_error = amrex::max(rel_error, w(n) * std::abs(state.xn[n-1] - state_coarse_dt.xn[n-1]));
        }
        rel_error = amrex::max(rel_error, w(net_ienuc) * std::abs(state.e - state_coarse_dt.e));

        bool step_success = false;
        if (rel_error < 1.0_rt) {
            step_success = true;
        }
#else

        // compute a relative error only for the species that are more
        // abundant that atol_spec

        Real rel_error = 0.0_rt;
        for (int n = 1; n <= NumSpec; n++) {
            if (state.xn[n-1] > atol_spec) {
                rel_error = amrex::max(rel_error,
                                       std::abs((state.xn[n-1] - state_coarse_dt.xn[n-1]) /
                                                (state.xn[n-1] + atol_spec)));
            }
        }
        rel_error = amrex::max(rel_error,
                               std::abs((state.e - state_coarse_dt.e) / state.e));

        bool step_success = false;
        if (rel_error < rtol_spec) {
            step_success = true;
        }

#endif


        if (ierr == BE_SUCCESS) {

            // there were no problems with the mechanics of the step
            // (linear algebra, etc.).  Now check if we met our error
            // goals.

            // state has the current best solution

            if (step_success) {
                t += dt_sub;

                // can we potentially increase the timestep?
                // backward-Euler has a local truncation error of dt**2

#ifdef VODELIKE_ERROR
                Real dt_new = dt_sub * std::pow(1.0_rt / rel_error, 0.5_rt);
#else
                Real dt_new = dt_sub * std::pow(rtol_spec / rel_error, 0.5_rt);
#endif
                dt_sub = amrex::min(amrex::max(dt_new, dt_sub / 2.0), 2.0 * dt_sub);

            } else {

                // we did not make our error goals, so just cut the timestep and leave the old solution in place

                dt_sub /= 2;

            }
        } else {

            // adjust the timestep and try again
            dt_sub /= 2;

        }

        ++num_timesteps;
        ++num_timesteps;

    }

    // do a final clean_state on the solution

    clean_state(state);

    // Subtract off the initial energy (the application codes expect
    // to get back only the generated energy during the burn).

    state.e -= e_in;

    if (num_timesteps >= ode_max_steps) {
        state.success = false;
    }


#ifndef AMREX_USE_CUDA
    if (burner_verbose) {
        // Print out some integration statistics, if desired.
        std::cout <<  "integration summary: " << std::endl;
        std::cout <<  "dens: " << state.rho << " temp: " << state.T << std::endl;
        std::cout <<  "energy released: " << state.e << std::endl;
        std::cout <<  "number of steps taken: " << num_timesteps << std::endl;
        std::cout <<  "number of f evaluations: " << state.n_rhs << std::endl;
    }
#endif

    // If we failed, print out the current state of the integration.

    if (!state.success) {
#ifndef AMREX_USE_CUDA
        std::cout << "ERROR: integration failed in net" << std::endl;
        std::cout << "time = " << t << std::endl;
        std::cout << "dens = " << state.rho << std::endl;
        std::cout << "temp start = " << T_in << std::endl;
        std::cout << "xn start = ";
        for (int n = 0; n < NumSpec; ++n) {
            std::cout << xn_in[n] << " ";
        }
        std::cout << std::endl;
        std::cout << "temp current = " << state.T << std::endl;
        std::cout << "xn current = ";
        for (int n = 0; n < NumSpec; ++n) {
            std::cout << state.xn[n] << " ";
        }
        std::cout << std::endl;
        std::cout << "energy generated = " << state.e << std::endl;
#endif
    }

}

#endif
