#ifndef VODE_TYPE_SIMPLIFIED_SDC_H
#define VODE_TYPE_SIMPLIFIED_SDC_H

#include <eos.H>
#include <eos_composition.H>
#include <burn_type.H>
#ifdef NETWORK_HAS_CXX_IMPLEMENTATION
#include <actual_rhs.H>
#else
#include <fortran_to_cxx_actual_rhs.H>
#endif
#ifdef NSE_THERMO
#include <nse.H>
#endif
#if defined(SDC_EVOLVE_ENTHALPY)
#include <maestro_params.H>
#endif


AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
void fill_unevolved_variables(const Real time, burn_t& state, dvode_t& vode_state)
{

    // some quantities are only advected.  Here we update those state
    // variables in burn_t's y[] array due only to advection.
    //
    // Note: we don't update the aux data here based on the X's -- that
    // needs to be done separately.
    //
    // Also note: vode_state is only used as input for the case where
    // we need to construct rho from (rho X).

#if defined(SDC_EVOLVE_ENERGY)

    // we are always integrating from t = 0, so there is no offset
    // time needed here.  The indexing of ydot_a is based on
    // the indices in burn_t and is 0-based
    state.y[SRHO] = amrex::max(state.rho_orig + state.ydot_a[SRHO] * time, EOSData::mindens);

    // for consistency
    state.rho = state.y[SRHO];

    state.y[SMX] = state.umx_orig + state.ydot_a[SMX] * time;
    state.y[SMY] = state.umy_orig + state.ydot_a[SMY] * time;
    state.y[SMZ] = state.umz_orig + state.ydot_a[SMZ] * time;

#elif defined(SDC_EVOLVE_ENTHALPY)

    // Keep density consistent with the partial densities.
    state.rho = 0.0_rt;
    for (int n = 1; n <= NumSpec; n++) {
        // use 1-based indexing into vode_state.y
        state.rho += vode_state.y(SFS+n);
    }
#endif

}


template <typename T>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
void sdc_burn_to_eos (const burn_t& state, T& eos_state)
{
#if defined(SDC_EVOLVE_ENERGY)
    eos_state.rho = state.y[SRHO];

#elif defined(SDC_EVOLVE_ENTHALPY)
    eos_state.rho = 0.0_rt;
    for (int n = 0; n < NumSpec; n++) {
        eos_state.rho += state.y[SFS+n];
    }
#endif

    eos_state.T = state.T;

    // species

    for (int n = 0; n < NumSpec; n++) {
        eos_state.xn[n] = state.y[SFS+n] / eos_state.rho;
    }

#if NAUX_NET > 0
    // aux
    for (int n = 0; n < NumAux; n++) {
        eos_state.aux[n] = state.y[SFX+n] / eos_state.rho;
    }
    //set_nse_aux_from_X(eos_state);
#endif

    // we don't bother filling the other fields, since the EOS call
    // will do that for us
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
void renormalize_species(const Real time, burn_t& state, dvode_t& vode_state)
{

    // Constrain the partial densities in vode_state to sum to the
    // density.  This does not update the burn_t state.
    //
    // We only renormalize species when evolving energy because
    // when we evolve enthalpy, we define the density as
    // the sum of the partial densities rho*X for each species.

#ifdef SDC_EVOLVE_ENERGY

    // update rho, rho*u, etc.

    fill_unevolved_variables(time, state, vode_state);

    Real nspec_sum = 0.0_rt;
    for (int n = 1; n <= NumSpec; n++) {
        // use 1-based indexing
        nspec_sum += vode_state.y(SFS+n);
    }
    nspec_sum /= state.y[SRHO];

    for (int n = 1; n <= NumSpec; n++) {
        vode_state.y(SFS+n) /= nspec_sum;
    }

    // note: we don't carry the auxiliary data in vode_state, so there
    // is no update to them done here

#endif

}


AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
void renormalize_species(burn_t& state)
{

    // Constrain the partial densities in burn_t state to sum to the 
    // density.
    //
    // This is meant to be used upon exit, and we assume that
    // vode_to_burn was already called

#ifdef SDC_EVOLVE_ENERGY

    // we have state.y[SRHO] to constrain to

    Real rhoX_sum = 0.0_rt;
    for (int n = 0; n < NumSpec; n++) {
        state.y[SFS+n] = amrex::max(amrex::min(state.y[SFS+n], state.y[SRHO]), 0.0_rt);
        rhoX_sum += state.y[SFS+n];
    }
    rhoX_sum /= state.y[SRHO];

    for (int n = 0; n < NumSpec; n++) {
        state.y[SFS+n] /= rhoX_sum;
    }

#ifdef NSE_THERMO
    // make the aux data consistent with the vode_state X's

    eos_re_t eos_state;
    for (int n = 0; n < NumSpec; n++) {
        eos_state.xn[n] = state.y[SFS+n] / state.y[SRHO];
    }

    set_nse_aux_from_X(eos_state);

    // also store it in the burn_t state

    for (int n = 0; n < NumAux; n++) {
        state.y[SFX+n] = state.y[SRHO] * eos_state.aux[n];
    }
#endif

#endif

}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
void clean_state(const Real time, burn_t& state, dvode_t& vode_state)
{

    // Ensure that mass fractions always stay positive.

    for (int n = 1; n <= NumSpec; ++n) {
        // we use 1-based indexing, so we need to offset SFS
        vode_state.y(SFS+n) = amrex::max(amrex::min(vode_state.y(SFS+n), state.rho),
                                         state.rho * 1.e-200_rt);
    }


    // renormalize abundances as necessary

    if (renormalize_abundances) {
        renormalize_species(time, state, vode_state);
    }

}


// Given a burn state, fill the rpar and integration state data.
// this is intended only to be called once -- at the very start of the integration

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
void burn_to_vode(burn_t& state, dvode_t& vode_state)
{

    for (int n = 0; n < SVAR_EVOLVE; n++) {
        // vode_state uses 1-based indexing
        vode_state.y(n+1) = state.y[n];
    }

    // store the original rho and rho U
#if defined(SDC_EVOLVE_ENERGY)

    state.rho_orig = state.y[SRHO];

    state.umx_orig = state.y[SMX];
    state.umy_orig = state.y[SMY];
    state.umz_orig = state.y[SMZ];

#else

    state.rho_orig = 0.0_rt;
    for (int n = 0; n < NumSpec; n++) {
        state.rho_orig += state.y[SFS+n];
    }

#endif

}


AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
void vode_to_burn(const Real time, dvode_t& vode_state, burn_t& state)
{

    // this makes burn_t represent the current integration state.  The
    // main thing this really does is compute the current temperature

    state.time = time;

    for (int n = 0; n < SVAR_EVOLVE; n++) {
        // note vode_state uses 1-based indexing
        state.y[n] = vode_state.y(n+1);
    }

    // update rho, rho*u, ... in the burn_t state
    // this may be redundant, but better to be safe

    fill_unevolved_variables(time, state, vode_state);

    Real rhoInv = 1.0_rt / state.rho;

#if defined(SDC_EVOLVE_ENERGY)
    eos_re_t eos_state;
#elif defined(SDC_EVOLVE_ENTHALPY)
    eos_t eos_state;
#endif
    eos_state.rho = state.rho;
    for (int n = 0; n < NumSpec; n++) {
        // vode_state uses 1-based indexing
        eos_state.xn[n] = vode_state.y(SFS+1+n) * rhoInv;
    }
#ifdef NSE_THERMO
    // make the aux data consistent with the vode_state X's
    set_nse_aux_from_X(eos_state);

    // also store it in the burn_t state
    for (int n = 0; n < NumAux; n++) {
        state.y[SFX+n] = state.rho * eos_state.aux[n];
    }
#endif


#if defined(SDC_EVOLVE_ENERGY)

    if (state.T_from_eden) {
        eos_state.e = (vode_state.y(SEDEN+1) -
                       0.5_rt * rhoInv * (state.y[SMX] * state.y[SMX] +
                                          state.y[SMY] * state.y[SMY] +
                                          state.y[SMZ] * state.y[SMZ])) * rhoInv;
    } else {
        eos_state.e = vode_state.y(SEINT+1) * rhoInv;
    }

#elif defined(SDC_EVOLVE_ENTHALPY)

    if (maestro::use_tfromp) {
        // under this assumption, we are assuming that there is no base state evolution
        eos_state.p = state.p0;
    } else {
        eos_state.h = vode_state.y(SENTH+1) * rhoInv;
    }

#endif

    eos_state.T = state.T;  // initial guess

#if defined(SDC_EVOLVE_ENERGY)
    eos(eos_input_re, eos_state);

#elif defined(SDC_EVOLVE_ENTHALPY)

    if (maestro::use_tfromp) {
        // NOT SURE IF THIS IS VALID
        // used to be an Abort statement
        eos(eos_input_rp, eos_state);
    } else {
        eos(eos_input_rh, eos_state);
    }

#endif



    // fill the rest of the burn_t state

    eos_to_burn(eos_state, state);

}


AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
void rhs_to_vode(const Real time, burn_t& state,
                 YdotNetArray1D& ydot_react, dvode_t& vode_state,
                 RArray1D& ydot)
{


    // ydot_react has just the contribution to the RHS from the
    // reaction network.  Note that these are in terms of dY/dt

    // start with the contribution from the non-reacting sources

    // Note: both ydot and ydot_react are 1-based
    for (int n = 0; n < SVAR_EVOLVE; n++) {
        ydot(n+1) = state.ydot_a[n];
    }

    // add in the reacting terms -- here we convert from dY/dt to dX/dt
    // The species derivatives are the first NumSpec components of
    // ydot_react

    for (int n = 1; n <= NumSpec; n++) {
        ydot(SFS+n) += state.rho * aion[n-1] * ydot_react(n);
    }

#if defined(SDC_EVOLVE_ENERGY)

    ydot(SEINT+1) += state.rho * ydot_react(net_ienuc);
    ydot(SEDEN+1) += state.rho * ydot_react(net_ienuc);

#elif defined(SDC_EVOLVE_ENTHALPY)

    ydot(SENTH+1) += state.rho * ydot_react(net_ienuc);

#endif

}

#endif
