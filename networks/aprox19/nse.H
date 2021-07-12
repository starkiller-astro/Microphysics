#ifndef NSE_H
#define NSE_H

#include <iostream>
#include <fstream>
#include <actual_network.H>

#include <AMReX.H>
#include <AMReX_Print.H>
#include <AMReX_Algorithm.H>
#include <AMReX_Array.H>
#include <AMReX_REAL.H>

#include <extern_parameters.H>

#include <burn_type.H>
#include <eos.H>

using namespace amrex;

AMREX_INLINE
void init_nse() {

  // set table parameters

  // read in table
  std::ifstream nse_table;

  amrex::Print() << "reading the NSE table (C++) ..." << std::endl;

  nse_table.open("nse19.tbl", std::ios::in);

  Real the, tsi, tfe;

  for (int irho = 1; irho <= table::nden; irho++) {
    for (int it9 = 1; it9 <= table::ntemp; it9++) {
      for (int iye = 1; iye <= table::nye; iye++) {
        int j = (irho-1)*table::ntemp*table::nye + (it9-1)*table::nye + iye;

        nse_table >> table::ttlog(j) >> table::ddlog(j) >> table::yetab(j);
        nse_table >> the >> tsi >> tfe;
        nse_table >> table::abartab(j) >> table::ebtab(j) >> table::wratetab(j);
        for (int n = 1; n <= NumSpec; n++) {
          nse_table >> table::massfractab(n, j);
        }
      }
    }
  }

}


AMREX_GPU_HOST_DEVICE AMREX_INLINE
bool in_nse(const eos_t& state, const bool relax = false) {

  using namespace Species;

  Real f = relax ? nse_relax_factor : 1.0_rt;

  bool nse_check = false;

  if (state.rho > f * rho_nse && state.T > f * T_nse) {

    // Ma checks on Fe-group (for our composition, this means Cr48, Fe52, Fe54, Ni56)
    // and C-group (for us, that is C12, N14)
    // and He-group (for us, that is H1, He3, He4)
    // also make sure there is not a lot of O16 (O16 burning will dominate then) or Si28

    Real Fe_group = state.xn[Cr48-1] + state.xn[Fe52-1] +
                    state.xn[Fe54-1] + state.xn[Ni56-1];
    Real C_group = state.xn[C12-1] + state.xn[N14-1];
    Real He_group = state.xn[H1-1] + state.xn[He3-1] + state.xn[He4-1];
    Real O_group = state.xn[O16-1];
    Real Si_group = state.xn[Si28-1];

    if (Fe_group + He_group > He_Fe_nse &&
        C_group < C_nse &&
        O_group < O_nse &&
        Si_group < Si_nse / f) {
      nse_check = true;
    }
  }

  return nse_check;

}


AMREX_GPU_HOST_DEVICE AMREX_INLINE
bool in_nse(const burn_t& state, const bool relax = false) {

    using namespace Species;

    Real f = relax ? nse_relax_factor : 1.0_rt;

    bool nse_check = false;

    if (state.rho > f * rho_nse && state.T > f * T_nse) {

        // Ma checks on Fe-group (for our composition, this means Cr48, Fe52, Fe54, Ni56)
        // and C-group (for us, that is C12, N14)
        // and He-group (for us, that is H1, He3, He4)
        // also make sure there is not a lot of O16 (O16 burning will dominate then) or Si28

#ifdef STRANG
        Real Fe_group = state.xn[Cr48-1] + state.xn[Fe52-1] +
                        state.xn[Fe54-1] + state.xn[Ni56-1];
        Real C_group = state.xn[C12-1] + state.xn[N14-1];
        Real He_group = state.xn[H1-1] + state.xn[He3-1] + state.xn[He4-1];
        Real O_group = state.xn[O16-1];
        Real Si_group = state.xn[Si28-1];
#else
        // we need to get the mass fractions from the conserved state passed in

        Real Fe_group = (state.y[SFS+Cr48-1] + state.y[SFS+Fe52-1] +
                         state.y[SFS+Fe54-1] + state.y[SFS+Ni56-1]) / state.rho;
        Real C_group = (state.y[SFS+C12-1] + state.y[SFS+N14-1]) / state.rho;
        Real He_group = (state.y[SFS+H1-1] + state.y[SFS+He3-1] + state.y[SFS+He4-1]) / state.rho;
        Real O_group = state.y[SFS+O16-1] / state.rho;
        Real Si_group = state.y[SFS+Si28-1] / state.rho;
#endif

        if (Fe_group + He_group > He_Fe_nse &&
            C_group < C_nse &&
            O_group < O_nse &&
            Si_group < Si_nse / f) {
            nse_check = true;
        }
    }

    return nse_check;

}

///
/// After doing a burn via integrating the network, update the composition
/// aux variables to be consistent with the new X's resulting from the burn.
///
template <class state_t> 
AMREX_GPU_HOST_DEVICE AMREX_INLINE
void set_nse_aux_from_X(state_t& state) {

  using namespace AuxZero;

  state.aux[iye] = 0.0_rt;
  state.aux[iabar] = 0.0_rt;
  state.aux[ibea] = 0.0_rt;

  for (int n = 0; n < NumSpec; n++) {
    state.aux[iye] += state.xn[n] * zion[n] * aion_inv[n];
    state.aux[iabar] += state.xn[n] * aion_inv[n];
    state.aux[ibea] += state.xn[n] * network::bion(n+1) * aion_inv[n];
  }
  state.aux[iabar] = 1.0_rt/state.aux[iabar];

}



AMREX_GPU_HOST_DEVICE AMREX_INLINE
void nse_interp(const Real T, const Real rho, const Real ye,
                Real& abar, Real& dq, Real& dyedt, Real* X) {

  using namespace table;
  using namespace AuxZero;

  Real tlog = std::log10(T);
  Real rholog = std::log10(rho);
  Real yet = ye;

  if (tlog < 9.0_rt) {
    tlog = 9.0_rt;
  }

  if (tlog > 10.4_rt) {
    tlog = 10.4_rt;
  }

  int it1 = static_cast<int>((tlog - 9.0_rt)*50.0_rt - 1.e-6_rt);
  it1 += 1;
  int it2 = it1 + 1;

  if (rholog < 7.0_rt) {
    rholog = 7.0_rt;
  }

  if (rholog > 10.0_rt) {
    rholog = 10.0_rt;
  }

  int ir1 = static_cast<int>((rholog - 7.0_rt)*10.0_rt - 1.e-6_rt);
  ir1 += 1;
  int ir2 = ir1 + 1;

  if (yet < 0.40_rt) {
    yet = 0.40_rt;
  }

  if (yet > 0.50_rt) {
    yet = 0.50_rt;
  }

  int ic1 = static_cast<int>((0.50_rt - yet)/0.005_rt - 1.0e-6_rt);
  ic1 += 1;
  int ic2 = ic1 + 1;

  // find the eight interpolation points in the 1D arrays

  int it1r1c1 = (ir1-1)*71*21 + (it1-1)*21 + ic1;
  int it1r1c2 = (ir1-1)*71*21 + (it1-1)*21 + ic2;
  int it1r2c1 = (ir2-1)*71*21 + (it1-1)*21 + ic1;
  int it1r2c2 = (ir2-1)*71*21 + (it1-1)*21 + ic2;
  int it2r1c1 = (ir1-1)*71*21 + (it2-1)*21 + ic1;
  int it2r1c2 = (ir1-1)*71*21 + (it2-1)*21 + ic2;
  int it2r2c1 = (ir2-1)*71*21 + (it2-1)*21 + ic1;
  int it2r2c2 = (ir2-1)*71*21 + (it2-1)*21 + ic2;

  Real t0 = 9.0_rt + static_cast<Real>(it1-1)*0.02_rt;
  Real r0 = 7.0_rt + static_cast<Real>(ir1-1)*0.10_rt;
  Real x0 = 0.50_rt - static_cast<Real>(ic1-1)*0.005_rt;

  Real td = (tlog - t0)/0.02_rt;
  Real rd = (rholog - r0)/0.10_rt;
  Real xd = (x0-yet)/0.005_rt;
  xd = amrex::max(0.0_rt, xd);

  Real omtd = 1.0_rt - td;
  Real omrd = 1.0_rt - rd;
  Real omxd = 1.0_rt - xd;

  abar =
    abartab(it1r1c1)*omtd*omrd*omxd +
    abartab(it1r1c2)*omtd*omrd*xd +
    abartab(it1r2c1)*omtd*rd*omxd +
    abartab(it1r2c2)*omtd*rd*xd +
    abartab(it2r1c1)*td*omrd*omxd +
    abartab(it2r1c2)*td*omrd*xd +
    abartab(it2r2c1)*td*rd*omxd +
    abartab(it2r2c2)*td*rd*xd;

  dq =
    ebtab(it1r1c1)*omtd*omrd*omxd +
    ebtab(it1r1c2)*omtd*omrd*xd +
    ebtab(it1r2c1)*omtd*rd*omxd +
    ebtab(it1r2c2)*omtd*rd*xd +
    ebtab(it2r1c1)*td*omrd*omxd +
    ebtab(it2r1c2)*td*omrd*xd +
    ebtab(it2r2c1)*td*rd*omxd +
    ebtab(it2r2c2)*td*rd*xd;

  dyedt =
    wratetab(it1r1c1)*omtd*omrd*omxd +
    wratetab(it1r1c2)*omtd*omrd*xd +
    wratetab(it1r2c1)*omtd*rd*omxd +
    wratetab(it1r2c2)*omtd*rd*xd +
    wratetab(it2r1c1)*td*omrd*omxd +
    wratetab(it2r1c2)*td*omrd*xd +
    wratetab(it2r2c1)*td*rd*omxd +
    wratetab(it2r2c2)*td*rd*xd;

  // this is actually the sum of all e- capture and e+ decay, so if
  // e- capture dominates, this quantity is positive, but Ye should
  // decrease, so we swap the sign here.
  dyedt = -dyedt;

  for (int n = 1; n <= NumSpec; n++) {
    X[n-1] =
      massfractab(n, it1r1c1)*omtd*omrd*omxd +
      massfractab(n, it1r1c2)*omtd*omrd*xd +
      massfractab(n, it1r2c1)*omtd*rd*omxd +
      massfractab(n, it1r2c2)*omtd*rd*xd +
      massfractab(n, it2r1c1)*td*omrd*omxd +
      massfractab(n, it2r1c2)*td*omrd*xd +
      massfractab(n, it2r2c1)*td*rd*omxd +
      massfractab(n, it2r2c2)*td*rd*xd;
  }

}


#ifdef SIMPLIFIED_SDC
///
/// update the state due to NSE changes for the simplified-SDC case
///
AMREX_GPU_HOST_DEVICE AMREX_INLINE
void sdc_nse_burn(burn_t& state, const Real dt) {

    using namespace AuxZero;

    state.success = true;
    state.n_rhs = 0;
    state.n_jac = 0;

    // call the NSE table to get (dYe/dt)^n
    Real abar_out;
    Real dq_out;
    Real dyedt;
    Real X[NumSpec];

    Real ye_in = state.y[SFX+iye] / state.rho;
    nse_interp(state.T, state.rho, ye_in,
               abar_out, dq_out, dyedt, X);

    Real dyedt_old = dyedt;

    // density and momentum have no reactive sources
    Real rho_old = state.y[SRHO];
    Real rho_bea_old = state.y[SFX+ibea];

    state.y[SRHO] += dt * state.ydot_a[SRHO];
    state.y[SMX] += dt * state.ydot_a[SMX];
    state.y[SMY] += dt * state.ydot_a[SMY];
    state.y[SMZ] += dt * state.ydot_a[SMZ];

    // predict the U^{n+1,*} state with only estimates of the aux
    // reaction terms and advection to dt

    eos_re_t eos_state;
    eos_state.T = 1.e6_rt; // initial guess

    // initial aux_sources
    Real aux_source[NumAux] = {0.0_rt};

    aux_source[iye] = rho_old * dyedt;

    Real rhoe_new;
    Real rho_aux_new[NumAux];
    Real dq;
    Real enucdot = 0.0_rt;

    Real rho_half = 0.5_rt * (rho_old + state.y[SRHO]);

    for (int iter = 0; iter < 3; iter++) {

        rhoe_new = state.y[SEINT] + dt * state.ydot_a[SEINT] + dt * rho_half * enucdot;

        for (int n = 0; n < NumAux; n++) {
            rho_aux_new[n] = state.y[SFX+n] + dt * state.ydot_a[SFX+n] + dt * aux_source[n];
        }

        // call the EOS to get the updated T*

        eos_state.rho = state.y[SRHO];
        eos_state.e = rhoe_new / state.y[SRHO];
        for (int n = 0; n < NumAux; n++) {
            eos_state.aux[n] = rho_aux_new[n] / state.y[SRHO];
        }

        eos(eos_input_re, eos_state);

        // call the NSE table using the * state to get the t^{n+1} source estimates

        nse_interp(eos_state.T, eos_state.rho, eos_state.aux[iye],
                   abar_out, dq_out, dyedt, X);

        // compute the energy release

        // this is MeV / nucleon * g / cm**3
        dq = dq_out - rho_bea_old/rho_old;

        // convert the energy to erg / cm**3
        enucdot  = dq * C::MeV2eV * C::ev2erg * C::n_A / dt;

        // update the aux source terms

        aux_source[iye] = 0.5_rt * rho_half * (dyedt_old + dyedt);
        aux_source[iabar] = rho_half * (abar_out - state.y[SFX+iabar] / rho_old) / dt;
        aux_source[ibea] = 0.0;   // this is actually not used in the EOS

    }

    // now update the aux quantities

    // the new mass fractions are just those that come from the table

    for (int n = 0; n < NumSpec; n++) {
        state.y[SFS+n] = state.y[SRHO] * X[n];
    }

    // aux data comes from the table (except Ye, which we predicted)

    state.y[SFX+iye] = eos_state.rho * eos_state.aux[iye];
    state.y[SFX+iabar] = eos_state.rho * abar_out;
    state.y[SFX+ibea] = eos_state.rho * dq_out;

    // density and momenta have already been updated

    // update the total and internal energy now

    state.y[SEINT] += dt * state.ydot_a[SEINT] + dt * rho_half * enucdot;
    state.y[SEDEN] += dt * state.ydot_a[SEDEN] + dt * rho_half * enucdot;

}

#else
///
/// update the state due to NSE changes for the Strang-split case
///
AMREX_GPU_HOST_DEVICE AMREX_INLINE
void nse_burn(burn_t& state, const Real dt) {

  using namespace AuxZero;

  // use the NSE table
  Real abar_out;
  Real dq_out;
  Real dyedt;
  Real X[NumSpec];

  // use the NSE table to get an estimate of dYe/dt

  nse_interp(state.T, state.rho, state.aux[iye],
             abar_out, dq_out, dyedt, X);

  // update Ye

  state.aux[iye] += dt * dyedt;

  // now get the composition from the table using the upated Ye

  nse_interp(state.T, state.rho, state.aux[iye],
             abar_out, dq_out, dyedt, X);

  state.success = true;
  state.n_rhs = 0;
  state.n_jac = 0;

  // this is MeV / nucleon -- here aux has not yet been updated, so we
  // access the old binding energy
  Real deltaq = dq_out - state.aux[ibea];

  // under-relaxation / inertia (see Ma et el. 2013)
  deltaq = eta * deltaq;

  state.aux[ibea] += deltaq;

  // convert the energy to erg / g
  Real enuc = deltaq * C::MeV2eV * C::ev2erg * C::n_A;

  state.e = enuc + state.e;

  state.aux[iabar] = abar_out;

  // store the new composition
  for (int n = 0; n < NumSpec; n++) {
    state.xn[n] = X[n];
  }

}
#endif

#endif
