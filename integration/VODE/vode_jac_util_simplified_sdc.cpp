#include <AMReX_Gpu.H>
#include <AMReX_REAL.H>
#include <eos.H>
#include <burn_type.H>
#include <vode_type.H>

AMREX_GPU_HOST_DEVICE
void jac_to_vode(const Real time, burn_t& state,
                 JacNetArray2D& jac_react, dvode_t& vode_state,
                 RArray2D& jac)
{

    // this is only used with an analytic Jacobian.  At the moment, we
    // only support a dense Jacobian.


    // we come in with jac_react being the Jacobian of the reacting system
    // but we need to convert it to the SDC system


    // SVAR_EVOLVE doesn't include rho, but we will include it here in
    // the intermediate this affects both the Castro
    // (SDC_EVOLVE_ENERGY) and MAESTROeX (SDC_EVOLVE_ENTHALPY) systems.

#if defined(SDC_EVOLVE_ENERGY)
    constexpr int iwrho = 1;
    constexpr int iwfs = 2;
    constexpr int iwK = iwfs+NumSpec;
    constexpr int iwe = iwK+1;
    constexpr int iwvar = 3+NumSpec;
#else
    constexpr int iwrho = 1;
    constexpr int iwfs=2;
    constexpr int iwe = iwfs+NumSpec;
    constexpr int iwvar = 2+NumSpec;
#endif

    amrex::Array2D<Real, 1, SVAR_EVOLVE+1, 1, iwvar> dRdw = {0.0_rt};
    amrex::Array2D<Real, 1, iwvar, 1, SVAR_EVOLVE+1> dwdU = {0.0_rt};

    constexpr Real eps = 1.e-8_rt;

    // this is 0-based to be consistent with SFS, SEDEN, ...
    constexpr int SRHO_EXTRA = SVAR_EVOLVE;


    // jac_react has the derivatives with respect to the native
    // network variables, X, T. e.  It does not have derivatives with
    // respect to density, so we'll have to compute those ourselves.

    // also fill the ydot
    YdotNetArray1D ydot;
    vode_to_burn(time, vode_state, state);
    actual_rhs(state, ydot);

    // at this point, our Jacobian should be entirely in terms of X,
    // not Y.  Let's now fix the rhs terms themselves to be in terms of
    // dX/dt and not dY/dt.
    for (int n = 1; n <= NumSpec; n++) {
        ydot(n) = ydot(n) * aion[n-1];
    }

    // now perturb density and call the RHS to compute the derivative wrt rho
    // species rates come back in terms of molar fractions
    burn_t state_pert = state;
    state_pert.rho = state.rho * (1.0_rt + eps);

    YdotNetArray1D ydot_pert;
    actual_rhs(state_pert, ydot_pert);

    // make the rates dX/dt and not dY/dt
    for (int n = 1; n <= NumSpec; n++) {
        ydot_pert(n) = ydot_pert(n) * aion[n-1];
    }

#if defined(SDC_EVOLVE_ENERGY)

    // The system we integrate has the form (rho X_k, rho E, rho e), but we will temporarily augment
    // This with rho, giving U = (rho, rho X_k, rho E, rho e).
    //
    // The intermediate state, w, has the form w = (rho, X_k, K, T), where K is 1/2 |U|^2

    // First compute dR/dw using the Jacobian that comes from the
    // network.  Note: this doesn't include density derivatives, so we
    // compute those via differencing.

    // dR/dw has the form:
    //
    //  SFS         / d(rho X1dot)/drho  d(rho X1dot)/dX1   d(rho X1dit)/dX2   ...  0   d(rho X1dot)/dT \
    //              | d(rho X2dot)/drho  d(rho X2dot)/dX1   d(rho X2dot)/dX2   ...  0   d(rho X2dot)/dT |
    //  SFS-1+nspec |   ...                                                         0                   |
    //  SEINT       | d(rho Edot)/drho   d(rho Edot)/dX1    d(rho Edot)/dX2    ...  0   d(rho Edot)/dT  |
    //  SEDEN       | d(rho Edot)/drho   d(rho Edot)/dX1    d(rho Edot)/dX2    ...  0   d(rho Edot)/dT  |
    //  SRHO_EXTRA  \        0                  0                  0                0          0       /
    //
    //                                                                              ^
    //                                                                              K derivatives


    // fill the column of dRdw corresponding to the derivative
    // with respect to rho

    // keep in mind here that that we are using 1-based indexing but SFS, ... are 0-based.

    for (int m = 1; m <= NumSpec; m++) {
        // d( d(rho X_m)/dt)/drho
        dRdw(SFS+m, iwrho) = ydot(m) + state.rho * (ydot_pert(m) - ydot(m))/(eps * state.rho);
    }

    // d( d(rho e)/dt)/drho
    dRdw(SEINT+1, iwrho) = ydot(net_ienuc) +
        state.rho * (ydot_pert(net_ienuc) - ydot(net_ienuc))/(eps * state.rho);

    // d( d(rho E)/dt)/drho
    dRdw(SEDEN+1, iwrho) = dRdw(SEINT+1, iwrho);

    // fill the columns of dRdw corresponding to each derivative
    // with respect to species mass fraction
    for (int n = 1; n <= NumSpec; n++) {
        for (int m = 1; m <= NumSpec; m++) {
            // d( d(rho X_m)/dt)/dX_n
            dRdw(SFS+m, iwfs-1+n) = state.rho * jac_react(m, n);
        }

        // d( d(rho e)/dt)/dX_n
        dRdw(SEINT+1, iwfs-1+n) = state.rho * jac_react(net_ienuc, n);

        // d( d(rho E)/dt)/dX_n
        dRdw(SEDEN+1, iwfs-1+n) = state.rho * jac_react(net_ienuc, n);
    }

    // now fill the column corresponding to derivatives with respect to
    // internal energy -- this column is iwe

    // d( d(rho X_m)/dt)/de
    for (int m = 1; m <= NumSpec; m++) {
        dRdw(SFS+m, iwe) = state.rho * jac_react(m, net_ienuc);
    }

    // d( d(rho e)/dt)/de
    dRdw(SEINT+1, iwe) = state.rho * jac_react(net_ienuc, net_ienuc);

    // d( d(rho E)/dt)/de
    dRdw(SEDEN+1, iwe) = dRdw(SEINT+1, iwe);

    // for the K derivatives, dRdw(:, iwK), and the rho sources,
    // dRdw(SRHO_EXTRA, :), we don't need to do anything, because these
    // are already zeroed out

    // that completes dRdw


    // construct dwdU

    // kinetic energy, K = 1/2 |U|^2
    Real kineng = 0.5_rt * (state.y[SMX] * state.y[SMX] +
                            state.y[SMY] * state.y[SMY] +
                            state.y[SMZ] * state.y[SMZ]) /
        (state.rho * state.rho);

    // density row (iwrho)
    dwdU(iwrho, SRHO_EXTRA+1) = 1.0_rt;

    // species rows
    for (int m = 1; m <= NumSpec; m++) {
        dwdU(iwfs-1+m, SFS+m) = 1.0_rt / state.rho;
        dwdU(iwfs-1+m, SRHO_EXTRA+1) = -state.xn[m-1] / state.rho;
    }

    // K row
    dwdU(iwK, SRHO_EXTRA+1) = -kineng / state.rho;
    dwdU(iwK, SEINT+1) = -1.0_rt / state.rho;
    dwdU(iwK, SEDEN+1) = 1.0_rt / state.rho;

    // e row
    eos_re_t eos_state;
    eos_state.rho = state.rho;
    eos_state.T = 1.e4_rt;   // initial guess
    for (int n = 0; n < NumSpec; n++) {
        eos_state.xn[n] = vode_state.y(SFS+1+n) / state.rho;
    }
#ifdef NSE_THERMO
    set_nse_aux_from_X(eos_state);
#endif

    eos_state.e = vode_state.y(SEINT+1) / state.rho;

    eos(eos_input_re, eos_state);

    const eos_xderivs_t eos_xderivs = composition_derivatives(eos_state);

    // internal energy row
    for (int n = 1; n <= NumSpec; n++) {
        dwdU(iwe, SFS+n) = -eos_xderivs.dedX[n-1] / state.rho;
    }
    dwdU(iwe, SEINT+1) = 1.0_rt / state.rho;
    dwdU(iwe, SEDEN+1) = 0.0_rt;
    Real X_dedX_sum = 0.0_rt;
    for (int n = 0; n < NumSpec; n++) {
        X_dedX_sum += eos_state.xn[n] * eos_xderivs.dedX[n];
    }
    dwdU(iwe, SRHO_EXTRA+1) =
        (X_dedX_sum - eos_state.e) / state.rho;

#elif defined(SDC_EVOLVE_ENTHALPY)

    // Our R source has components for species and enthalpy only.  But
    // we will extend it here to include the mass density too to ensure
    // that we have a square matrix in dU/dw that we can take the
    // inverse of to use below.  When we compute the final Jacobian, we will
    // discard the density row.

    // Our jacobian, dR/dw has the form:
    //
    //  SFS         / d(rho X1dot)/drho  d(rho X1dot)/dX1   d(rho X1dit)/dX2   ...  d(rho X1dot)/dT \
    //              | d(rho X2dot)/drho  d(rho X2dot)/dX1   d(rho X2dot)/dX2   ...  d(rho X2dot)/dT |
    //  SFS-1+nspec |   ...                                                                         |
    //  SENTH       | d(rho h)/drho      d(rho h)/dX1       d(rho h)/dX2       ...  d(rho h)/dT     |
    //  SRHO_EXTRA  \ 0                  0                  0                       0               /


    // fill the column of dRdw corresponding to the derivative
    // with respect to rho
    for (int m = 1; m <= NumSpec; m++) {
        // d( d(rho X_m)/dt)/drho
        dRdw(SFS+m, iwrho) = ydot(m) + state.rho * (ydot_pert(m) - ydot(m))/(eps * state.rho);
    }

    // d( d(rho h)/dt)/drho
    dRdw(SENTH+1, iwrho) = ydot(net_ienuc) +
        state.rho * (ydot_pert(net_ienuc) - ydot(net_ienuc))/(eps * state.rho);

    // d( d(rho)/dt)/drho
    dRdw(SRHO_EXTRA+1, iwrho) = 0.0_rt;

    // fill the columns of dRdw corresponding to each derivative
    // with respect to species mass fraction
    for (int n = 1; n <= NumSpec; n++) {
        for (int m = 1; m <= NumSpec; m++) {
            // d( d(rho X_m)/dt)/dX_n
            dRdw(SFS+m, iwfs-1+n) = state.rho * jac_react(m, n);
        }

        // d( d(rho h)/dt)/dX_n
        dRdw(SENTH+1, iwfs-1+n) = state.rho * jac_react(net_ienuc, n);

        // d( d(rho)/dt)/dX_n
        dRdw(SRHO_EXTRA+1, iwfs-1+n) = 0.0_rt;
    }

    // now fill the column corresponding to derivatives with respect to
    // internal energy -- this column is iwe

    // d( d(rho X_m)/dt)/de
    for (int m = 1; m <= NumSpec; m++) {
        dRdw(SFS+m, iwe) = state.rho * jac_react(m, net_ienuc);
    }

    // d( d(rho h)/dt)/de
    dRdw(SENTH+1, iwe) = state.rho * jac_react(net_ienuc, net_ienuc);

    // d( d(rho)/dt)/de
    dRdw(SRHO_EXTRA+1, iwe) = 0.0_rt;

    // that completes dRdw

    // construct dwdU.  Here we take U = (rho X, rho h, rho)^T

    // density row (iwrho)
    dwdU(iwrho, SRHO_EXTRA+1) = 1.0_rt;

    // species rows
    for (int m = 1; m <= NumSpec; m++) {
        dwdU(iwfs-1+m, SFS+m) = 1.0_rt / state.rho;
        dwdU(iwfs-1+m, SRHO_EXTRA+1) = -state.xn[m-1] / state.rho;
    }

    // h row
    eos_t eos_state;
    eos_state.rho = state.rho;
    eos_state.T = 1.e4_rt;   // initial guess
    for (int n = 0; n < NumSpec; n++) {
        eos_state.xn[n] = vode_state.y(SFS+1+n) / state.rho;
    }
#ifdef NSE_THERMO
    set_nse_aux_from_X(eos_state);
#endif

    eos_state.h = vode_state.y(SENTH+1) / state.rho;

    eos(eos_input_rh, eos_state);

    const eos_xderivs_t eos_xderivs = composition_derivatives(eos_state);

    // internal energy row
    for (int n = 1; n <= NumSpec; n++) {  
        dwdU(iwe, SFS+n) = -eos_xderivs.dhdX[n-1] / (state.rho * eos_state.dedT);
    }
    dwdU(iwe, SENTH+1) = 1.0_rt / (state.rho * eos_state.dhdT);
    Real X_dhdX_sum = 0.0;
    for (int n = 0; n < NumSpec; n++) {
        X_dhdX_sum += eos_state.xn[n] * eos_xderivs.dhdX[n];
    }
    dwdU(iwe, SRHO_EXTRA+1) =
        (X_dhdX_sum - state.rho * eos_state.dhdr - eos_state.h) /
        (state.rho * eos_state.dhdT);

#endif


    // compute J = dR/dw dw/dU

    // J is SVAR_EVOLVE x SVAR_EVOLVE, which will call m x n
    //
    // J = dR/dw dw/dU
    //
    //   dR/dw is SVAR_EVOLVE+1 x iwvar, which we call m x k
    //   dw/dU is iwvar x SVAR_EVOLVE+1, which we call k x n
    //

    // we need to cut out the density (SRHO_EXTRA) row and column of
    // the Jacobian, since that is not in our full SVAR_EVOLVE state
    for (int n = 1; n <= SVAR_EVOLVE; n++) {
        if (n == SRHO_EXTRA+1) continue;
        for (int m = 1; m <= SVAR_EVOLVE; m++) {
            if (m == SRHO_EXTRA+1) continue;

            jac(m, n) = 0.0_rt;
            for (int k = 1; k <= iwvar; k++) {
                jac(m, n) += dRdw(m, k) * dwdU(k, n);
            }
        }
    }

}

AMREX_GPU_HOST_DEVICE 
void rhs_to_vode(const Real time, burn_t& state,
                 YdotNetArray1D& ydot_react, dvode_t& vode_state,
                 RArray1D& ydot)
{


    fill_unevolved_variables(time, state, vode_state);

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
