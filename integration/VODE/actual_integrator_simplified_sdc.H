#ifndef actual_integrator_H
#define actual_integrator_H

// Common variables and routines for burners
// that use VODE for their integration.

#include <network.H>
#include <burn_type.H>
#include <eos_type.H>
#include <eos.H>
#include <extern_parameters.H>
#include <vode_type.H>
#include <vode_dvode.H>

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
void actual_integrator (burn_t& state, Real dt)
{

    dvode_t vode_state;

    // Start off by assuming a successful burn.

    state.success = true;

    // Initialize the integration time.

    vode_state.t = 0.0_rt;
    vode_state.tout = dt;

    // Set the (inverse of the) timestep limiter.

    vode_state.HMXI = 1.0_rt / ode_max_dt;

    // We assume that (rho, T) coming in are valid, do an EOS call
    // to fill the rest of the thermodynamic variables.

    eos_re_t eos_state;

    // initialize the EOS state using the conserved state that was
    // passed in

    sdc_burn_to_eos(state, eos_state);

    eos(eos_input_rt, eos_state);

    eos_to_burn(eos_state, state);

    // Fill in the initial integration state.

    burn_to_vode(state, vode_state);


    // Set the tolerances.

    Real sdc_tol_fac = std::pow(sdc_burn_tol_factor, state.num_sdc_iters - state.sdc_iter - 1);

    // we use 1-based indexing inside of VODE, so we need to shift the
    // indices SRHO, SFS, etc by 1

#if defined(SDC_EVOLVE_ENERGY)
    Real sdc_min_density = amrex::min(state.rho, state.rho + state.ydot_a[SRHO] * dt);

    vode_state.atol(SEDEN+1) = sdc_min_density * atol_enuc * sdc_tol_fac; 
    vode_state.atol(SEINT+1) = sdc_min_density * atol_enuc * sdc_tol_fac;

    vode_state.rtol(SEDEN+1) = rtol_enuc * sdc_tol_fac; 
    vode_state.rtol(SEINT+1) = rtol_enuc * sdc_tol_fac;

#elif defined(SDC_EVOLVE_ENTHALPY)
    Real sdc_min_density = state.rho;
    for (int n = 0; n < NumSpec; ++n) {
      sdc_min_density += dt * state.ydot_a[SFS+n];
    }
    sdc_min_density = amrex::min(state.rho, sdc_min_density);

    vode_state.atol(SENTH+1) = sdc_min_density * atol_enuc * sdc_tol_fac; 
    vode_state.rtol(SENTH+1) = rtol_enuc * sdc_tol_fac; 

#endif

    // Note: we define the input atol for species to refer only to the
    // mass fraction part, and we multiply by a representative density
    // so that atol becomes an absolutely tolerance on (rho X)

    for (int n = 1; n <= NumSpec; ++n) {
        vode_state.atol(SFS+n) = sdc_min_density * atol_spec * sdc_tol_fac;
    }
    for (int n = 1; n <= NumSpec; ++n) {
        vode_state.rtol(SFS+n) = rtol_spec * sdc_tol_fac;
    }


    // Call the integration routine.

    int istate = dvode(state, vode_state);

    // Get the number of RHS and Jacobian evaluations.

    state.n_rhs = vode_state.NFE;
    state.n_jac = vode_state.NJE;

    // Copy the integration data back to the burn state.
    // This will also update the aux state from X if we are using NSE

    vode_to_burn(vode_state.t, vode_state, state);

    // normalize the abundances on exit.  We'll assume that the driver
    // calling this is making use of the conserved state (state.y[]),
    // so that is what will be normalized.

    renormalize_species(state);

    // VODE does not always fail even though it can lead to unphysical states.
    // Add some checks that indicate a burn fail even if VODE thinks the
    // integration was successful.

    if (istate < 0) {
        state.success = false;
    }

#if defined(SDC_EVOLVE_ENERGY)
    if (vode_state.y(SEINT+1) < 0.0_rt || vode_state.y(SEDEN+1) < 0.0_rt) {
      state.success = false;
    }

    for (int n = 1; n <= NumSpec; ++n) {
        if (vode_state.y(SFS+n) / state.y[SRHO] < -vode_failure_tolerance) {
            state.success = false;
        }

        if (vode_state.y(SFS+n) / state.y[SRHO] > 1.0_rt + vode_failure_tolerance) {
            state.success = false;
        }
    }

#elif defined(SDC_EVOLVE_ENTHALPY)
    for (int n = 1; n <= NumSpec; ++n) {
        if (vode_state.y(SFS+n) / state.rho < -vode_failure_tolerance) {
            state.success = false;
        }

        if (vode_state.y(SFS+n) / state.rho > 1.0_rt + vode_failure_tolerance) {
            state.success = false;
        }
    }

#endif


#ifndef AMREX_USE_CUDA
    if (burner_verbose) {
        // Print out some integration statistics, if desired.
        std::cout <<  "integration summary: " << std::endl;
        std::cout <<  "dens: " << state.rho << " temp: " << state.T << std::endl;
        std::cout << " energy released: " << state.e << std::endl;
        std::cout <<  "number of steps taken: " << vode_state.NST << std::endl;
        std::cout <<  "number of f evaluations: " << vode_state.NFE << std::endl;
    }
#endif

    // If we failed, print out the current state of the integration.

    if (!state.success) {
        if (istate != -100) {
#ifndef AMREX_USE_GPU
            std::cout << "ERROR: integration failed in net" << std::endl;
            std::cout << "istate = " << istate << std::endl;
            std::cout << "time = " << vode_state.t << std::endl;
            std::cout << "dens = " << state.rho << std::endl;
            std::cout << "temp start = " << eos_state.T << std::endl;
            std::cout << "xn start = ";
            for (int n = 0; n < NumSpec; ++n) {
                std::cout << eos_state.xn[n] << " ";
            }
            std::cout << std::endl;
            std::cout << "temp current = " << state.T << std::endl;
            std::cout << "xn current = ";
            for (int n = 0; n < NumSpec; ++n) {
                std::cout << state.xn[n] << " ";
            }
            std::cout << std::endl;
#endif
        } else {
#ifndef AMREX_USE_GPU
            std::cout << "burn entered NSE during integration" << std::endl;
#endif
        }
    }

}

#endif
