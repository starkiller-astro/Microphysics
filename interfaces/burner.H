#ifndef _burner_H_
#define _burner_H_

#include <burn_type.H>
#ifdef SIMPLIFIED_SDC
#include <integrator_sdc.H>
#endif
#ifdef STRANG
#include <integrator.H>
#endif

#include <ArrayUtilities.H>

#ifdef NSE_TABLE
#include <nse.H>
#endif

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
void burner (burn_t& state, Real dt)
{

#ifndef TRUE_SDC

#ifdef NSE_TABLE

    bool nse_check = in_nse(state);

    if (nse_check) {

        // call the table
#ifdef SIMPLIFIED_SDC
        sdc_nse_burn(state, dt);
#else
        nse_burn(state, dt);
#endif

    } else {
        // burn as usual
        integrator(state, dt);

#ifndef SIMPLIFIED_SDC
        // update the aux from the new X's this is not needed with the
        // SDC method, since we took care of that internally.
        set_nse_aux_from_X(state);
#endif

        // if we failed our burn, it may have been because we tried to
        // enter NSE and required too many steps.  At this point, the
        // aux data, T, and e represent the result from the incomplete
        // burn.

        // we use a relaxed NSE criteria now to catch states that are
        // right on the edge of being in NSE

        if (in_nse(state, true) && state.success == false) {
            // replace dt with just the remaining integration time
            // left after the failure
            Real dt_remaining = amrex::max(dt - state.time, 0.0_rt);
            // This will append to state.e the amount additional
            // energy released from adjusting to the new NSE state
#ifdef SIMPLIFIED_SDC
            sdc_nse_burn(state, dt_remaining);
#else
            nse_burn(state, dt_remaining);
#endif
        } else if (state.success == false && abort_on_failure) {
#ifndef AMREX_USE_GPU
            std::cout << "failure \n";
            std::cout << state << std::endl;
            amrex::Error("integration failed in burner");
#endif
        }
    }

#else
    // we did not compile with NSE
    integrator(state, dt);
#endif

#endif

}

#endif
