#ifndef _actual_rhs_H_
#define _actual_rhs_H_

#include <AMReX.H>
#include <AMReX_Print.H>
#include <AMReX_Algorithm.H>
#include <AMReX_Array.H>
#include <AMReX_REAL.H>

#include <extern_parameters.H>
#include <microphysics_math.H>
#include <actual_network.H>
#include <burn_type.H>
#include <tfactors.H>
#include <rate_type.H>
#include <screen.H>
#include <sneut5.H>
#include <aprox_rates.H>
#include <ArrayUtilities.H>
#include <rhs_utilities.H>
#include <jacobian_utilities.H>

using namespace amrex;
using namespace ArrayUtil;

void actual_rhs_init();

namespace RateTable
{
    constexpr Real tab_tlo = 6.0e0_rt;
    constexpr Real tab_thi = 10.0e0_rt;
    constexpr int tab_per_decade = 2000;
    constexpr int nrattab = static_cast<int>(tab_thi - tab_tlo) * tab_per_decade + 1;
    constexpr int tab_imax = static_cast<int>(tab_thi - tab_tlo) * tab_per_decade + 1;
    constexpr Real tab_tstp = (tab_thi - tab_tlo) / static_cast<Real>(tab_imax - 1);

    extern AMREX_GPU_MANAGED Array2D<Real, 1, Rates::NumRates, 1, nrattab> rattab;
    extern AMREX_GPU_MANAGED Array2D<Real, 1, Rates::NumRates, 1, nrattab> drattabdt;
    extern AMREX_GPU_MANAGED Array1D<Real, 1, nrattab> ttab;
}


AMREX_GPU_HOST_DEVICE AMREX_INLINE
void iso7tab(const Real btemp, const Real bden,
             Array1D<Real, 1, Rates::NumRates>& rate,
             Array1D<Real, 1, Rates::NumRates>& dratedt)
{
    using namespace RateTable;
    using namespace Rates;

    constexpr int mp = 4;

    int iat;
    Real x, x1, x2, x3, x4;
    Real a, b, c, d, e, f, g, h, p, q;
    Real alfa, beta, gama, delt;

    // hash locate
    iat = static_cast<int>((std::log10(btemp) - tab_tlo)/tab_tstp) + 1;
    iat = amrex::max(1, amrex::min(iat - 1, tab_imax - mp + 1));

    // setup the lagrange interpolation coefficients for a cubic
    x  = btemp;
    x1 = ttab(iat);
    x2 = ttab(iat+1);
    x3 = ttab(iat+2);
    x4 = ttab(iat+3);
    a  = x - x1;
    b  = x - x2;
    c  = x - x3;
    d  = x - x4;
    e  = x1 - x2;
    f  = x1 - x3;
    g  = x1 - x4;
    h  = x2 - x3;
    p  = x2 - x4;
    q  = x3 - x4;
    alfa =  b*c*d/(e*f*g);
    beta = -a*c*d/(e*h*p);
    gama =  a*b*d/(f*h*q);
    delt = -a*b*c/(g*p*q);

    // crank off the raw reaction rates
    for (int rateindex = 1; rateindex <= Rates::NumRates; ++rateindex) {

        Real dtab = 0.0_rt;

        // Set the density dependence

        switch (rateindex) {

        case (C12_He4_to_O16_forward):
            dtab = bden;
            break;

        case (C12_He4_to_O16_reverse):
            dtab = 1.0e0_rt;
            break;

        case (He4_He4_He4_to_C12_forward):
            dtab = bden * bden;
            break;

        case (He4_He4_He4_to_C12_reverse):
            dtab = 1.0e0_rt;
            break;

        case (C12_C12_to_Ne20_He4_forward):
            dtab = bden;
            break;

        case (C12_C12_to_Ne20_He4_reverse):
            dtab = bden; // rate is zero in this net
            break;

        case (C12_O16_to_Mg24_He4_forward):
            dtab = bden;
            break;

        case (C12_O16_to_Mg24_He4_reverse):
            dtab = bden; // rate is zero in this net
            break;

        case (C12_O16_to_Si28_forward):
            dtab = bden;
            break;

        case (C12_O16_to_Si28_reverse):
            dtab = 1.0_rt; // rate is zero in this net
            break;

        case (O16_O16_to_Si28_He4_forward):
            dtab = bden;
            break;

        case (O16_O16_to_Si28_He4_reverse):
            dtab = bden; // rate is zero in this net
            break;

        case (O16_He4_to_Ne20_forward):
            dtab = bden;
            break;

        case (O16_He4_to_Ne20_reverse):
            dtab = 1.0e0_rt;
            break;

        case (Ne20_He4_to_Mg24_forward):
            dtab = bden;
            break;

        case (Ne20_He4_to_Mg24_reverse):
            dtab = 1.0e0_rt;
            break;

        case (Mg24_He4_to_Si28_forward):
            dtab = bden;
            break;

        case (Mg24_He4_to_Si28_reverse):
            dtab = 1.0e0_rt;
            break;

        case (Ca40_He4_to_Ti44_forward):
            dtab = bden;
            break;

        case (Ca40_He4_to_Ti44_reverse):
            dtab = 1.0e0_rt;
            break;

        case (Si28_7He4_to_Ni56_forward):
            dtab = 0.0e0_rt;
            break;

        case (Si28_7He4_to_Ni56_reverse):
            dtab = 0.0e0_rt;
            break;

        }

        rate(rateindex) = (alfa * rattab(rateindex, iat  ) +
                           beta * rattab(rateindex, iat+1) +
                           gama * rattab(rateindex, iat+2) +
                           delt * rattab(rateindex, iat+3)) * dtab;

        dratedt(rateindex) = (alfa * drattabdt(rateindex, iat  ) +
                              beta * drattabdt(rateindex, iat+1) +
                              gama * drattabdt(rateindex, iat+2) +
                              delt * drattabdt(rateindex, iat+3)) * dtab;

    }
}


AMREX_GPU_HOST_DEVICE AMREX_INLINE
void iso7rat(const Real btemp, const Real bden,
             Array1D<Real, 1, Rates::NumRates>& rate,
             Array1D<Real, 1, Rates::NumRates>& dratedt)
{
    using namespace Rates;

    // this routine generates unscreened
    // nuclear reaction rates for the iso7 network.

    Real rrate,drratedt;

    for (int i = 1; i <= Rates::NumRates; ++i) {
       rate(i)    = 0.0_rt;
       dratedt(i) = 0.0_rt;
    }

    if (btemp < 1.0e6_rt) return;


    // get the temperature factors
    tf_t tf = get_tfactors(btemp);

    // c12(a,g)o16
    rate_c12ag(tf,bden,
               rate(C12_He4_to_O16_forward),dratedt(C12_He4_to_O16_forward),
               rate(C12_He4_to_O16_reverse),dratedt(C12_He4_to_O16_reverse));

    // triple alpha to c12
    rate_triplealf(tf,bden,
                   rate(He4_He4_He4_to_C12_forward),dratedt(He4_He4_He4_to_C12_forward),
                   rate(He4_He4_He4_to_C12_reverse),dratedt(He4_He4_He4_to_C12_reverse));

    // c12 + c12
    rate_c12c12(tf,bden,
                rate(C12_C12_to_Ne20_He4_forward),dratedt(C12_C12_to_Ne20_He4_forward),
                rrate,drratedt);

    // c12 + o16
    rate_c12o16(tf,bden,
                rate(C12_O16_to_Mg24_He4_forward),dratedt(C12_O16_to_Mg24_He4_forward),
                rrate,drratedt);

    // 16o + 16o
    rate_o16o16(tf,bden,
                rate(O16_O16_to_Si28_He4_forward),dratedt(O16_O16_to_Si28_He4_forward),
                rrate,drratedt);

    // o16(a,g)ne20
    rate_o16ag(tf,bden,
               rate(O16_He4_to_Ne20_forward),dratedt(O16_He4_to_Ne20_forward),
               rate(O16_He4_to_Ne20_reverse),dratedt(O16_He4_to_Ne20_reverse));

    // ne20(a,g)mg24
    rate_ne20ag(tf,bden,
                rate(Ne20_He4_to_Mg24_forward),dratedt(Ne20_He4_to_Mg24_forward),
                rate(Ne20_He4_to_Mg24_reverse),dratedt(Ne20_He4_to_Mg24_reverse));

    // mg24(a,g)si28
    rate_mg24ag(tf,bden,
                rate(Mg24_He4_to_Si28_forward),dratedt(Mg24_He4_to_Si28_forward),
                rate(Mg24_He4_to_Si28_reverse),dratedt(Mg24_He4_to_Si28_reverse));

    // ca40(a,g)ti44
    rate_ca40ag(tf,bden,
                rate(Ca40_He4_to_Ti44_forward),dratedt(Ca40_He4_to_Ti44_forward),
                rate(Ca40_He4_to_Ti44_reverse),dratedt(Ca40_He4_to_Ti44_reverse));

}


AMREX_GPU_HOST_DEVICE AMREX_INLINE
void set_iso7rat()
{
    using namespace RateTable;

    Real btemp;
    Real bden = 1.0e0_rt;
    Array1D<Real, 1, Rates::NumRates> rate;
    Array1D<Real, 1, Rates::NumRates> dratedt;

    for (int i = 1; i <= tab_imax; ++i) {

       btemp = tab_tlo + static_cast<Real>(i-1) * tab_tstp;
       btemp = std::pow(10.0e0_rt, btemp);

       iso7rat(btemp, bden, rate, dratedt);

       ttab(i) = btemp;

       for (int j = 1; j <= Rates::NumRates; ++j) {

          rattab(j,i)    = rate(j);
          drattabdt(j,i) = dratedt(j);

       }
    }
}


AMREX_GPU_HOST_DEVICE AMREX_INLINE
void screen_iso7(const Real btemp, const Real bden,
                 Array1D<Real, 1, NumSpec> const& y,
                 Array1D<Real, 1, Rates::NumRates>& rate,
                 Array1D<Real, 1, Rates::NumRates>& dratedt,
                 Array2D<Real, 1, NumSpec, 1, Rates::NumRates>& dratedY)
{
    using namespace Species;
    using namespace Rates;

    /*
    this routine computes the screening factors
    and applies them to the raw reaction rates,
    producing the final reaction rates used by the
    right hand sides and jacobian matrix elements
    */

    int jscr;
    Real sc1a,sc1adt,sc1add,sc2a,sc2adt,sc2add;
    Real sc3a,sc3adt,sc3add,abar,zbar,ye,z2bar;
    Real t992,t9i92,yeff_ca40,yeff_ca40dt,yeff_ti44,yeff_ti44dt;
    Real denom,denomdt,denomdd,xx,zz;

    plasma_state_t pstate;

    // initialize
    for (int j = 1; j <= NumSpec; ++j) {
        for (int i = 1; i <= Rates::NumRates; ++i) {
            dratedY(j, i) = 0.0_rt;
        }
    }

    // get the temperature factors
    tf_t tf = get_tfactors(btemp);

    // Set up the state data, which is the same for all screening factors.

    fill_plasma_state(pstate, btemp, bden, y);

    // first the always fun triple alpha and its inverse
    jscr = 0;
    screen5(pstate,jscr,
            zion[He4-1], aion[He4-1], zion[He4-1], aion[He4-1],
            sc1a,sc1adt,sc1add);

    jscr++;
    screen5(pstate,jscr,
            zion[He4-1], aion[He4-1], 4.0_rt, 8.0_rt,
            sc2a,sc2adt,sc2add);

    sc3a   = sc1a * sc2a;
    sc3adt = sc1adt*sc2a + sc1a*sc2adt;

    dratedt(He4_He4_He4_to_C12_forward) = dratedt(He4_He4_He4_to_C12_forward) * sc3a + rate(He4_He4_He4_to_C12_forward) * sc3adt;
    rate(He4_He4_He4_to_C12_forward)    = rate(He4_He4_He4_to_C12_forward) * sc3a;

    // c12 to o16
    jscr++;
    screen5(pstate,jscr,
            zion[C12-1], aion[C12-1], zion[He4-1], aion[He4-1],
            sc1a,sc1adt,sc1add);

    dratedt(C12_He4_to_O16_forward)  = dratedt(C12_He4_to_O16_forward) * sc1a + rate(C12_He4_to_O16_forward) * sc1adt;
    rate(C12_He4_to_O16_forward)     = rate(C12_He4_to_O16_forward) * sc1a;

    // c12 + c12
    jscr++;
    screen5(pstate,jscr,
            zion[C12-1], aion[C12-1], zion[C12-1], aion[C12-1],
            sc1a,sc1adt,sc1add);

    dratedt(C12_C12_to_Ne20_He4_forward) = dratedt(C12_C12_to_Ne20_He4_forward) * sc1a + rate(C12_C12_to_Ne20_He4_forward) * sc1adt;
    rate(C12_C12_to_Ne20_He4_forward)    = rate(C12_C12_to_Ne20_He4_forward) * sc1a;

    // c12 + o16
    jscr++;
    screen5(pstate,jscr,
            zion[C12-1], aion[C12-1], zion[O16-1], aion[O16-1],
            sc1a,sc1adt,sc1add);

    dratedt(C12_O16_to_Mg24_He4_forward) = dratedt(C12_O16_to_Mg24_He4_forward) * sc1a + rate(C12_O16_to_Mg24_He4_forward) * sc1adt;
    rate(C12_O16_to_Mg24_He4_forward)    = rate(C12_O16_to_Mg24_He4_forward) * sc1a;

    // Equal probability branching ratio between (Mg24 + He4) and (Si28) endpoints

    rate(C12_O16_to_Mg24_He4_forward) *= 0.5;
    dratedt(C12_O16_to_Mg24_He4_forward) *= 0.5;

    rate(C12_O16_to_Si28_forward) = rate(C12_O16_to_Mg24_He4_forward);
    dratedt(C12_O16_to_Si28_forward) = dratedt(C12_O16_to_Mg24_He4_forward);

    // o16 + o16
    jscr++;
    screen5(pstate,jscr,
            zion[O16-1], aion[O16-1], zion[O16-1], aion[O16-1],
            sc1a,sc1adt,sc1add);

    dratedt(O16_O16_to_Si28_He4_forward) = dratedt(O16_O16_to_Si28_He4_forward) * sc1a + rate(O16_O16_to_Si28_He4_forward) * sc1adt;
    rate(O16_O16_to_Si28_He4_forward)    = rate(O16_O16_to_Si28_He4_forward) * sc1a;

    // o16 to ne20
    jscr++;
    screen5(pstate,jscr,
            zion[O16-1], aion[O16-1], zion[He4-1], aion[He4-1],
            sc1a,sc1adt,sc1add);

    dratedt(O16_He4_to_Ne20_forward) = dratedt(O16_He4_to_Ne20_forward) * sc1a + rate(O16_He4_to_Ne20_forward) * sc1adt;
    rate(O16_He4_to_Ne20_forward)    = rate(O16_He4_to_Ne20_forward) * sc1a;

    // ne20 to mg24
    jscr++;
    screen5(pstate,jscr,
            zion[Ne20-1], aion[Ne20-1], zion[He4-1], aion[He4-1],
            sc1a,sc1adt,sc1add);

    dratedt(Ne20_He4_to_Mg24_forward) = dratedt(Ne20_He4_to_Mg24_forward) * sc1a + rate(Ne20_He4_to_Mg24_forward) * sc1adt;
    rate(Ne20_He4_to_Mg24_forward)    = rate(Ne20_He4_to_Mg24_forward) * sc1a;

    // mg24 to si28
    jscr++;
    screen5(pstate,jscr,
            zion[Mg24-1], aion[Mg24-1], zion[He4-1], aion[He4-1],
            sc1a,sc1adt,sc1add);

    dratedt(Mg24_He4_to_Si28_forward) = dratedt(Mg24_He4_to_Si28_forward) * sc1a + rate(Mg24_He4_to_Si28_forward) * sc1adt;
    rate(Mg24_He4_to_Si28_forward)    = rate(Mg24_He4_to_Si28_forward) * sc1a;

    // ca40 to ti44
    jscr++;
    screen5(pstate,jscr,
            20.0_rt, 40.0_rt, zion[He4-1], aion[He4-1],
            sc1a,sc1adt,sc1add);

    dratedt(Ca40_He4_to_Ti44_forward) = dratedt(Ca40_He4_to_Ti44_forward) * sc1a + rate(Ca40_He4_to_Ti44_forward) * sc1adt;
    rate(Ca40_He4_to_Ti44_forward)    = rate(Ca40_He4_to_Ti44_forward) * sc1a;

    /*
    the publication, timmes, woosley & hoffman apjs, 129, 377
    has a typo on page 393, where its says "y(C12)+y(O16) .gt. 0.004"
    it should be less than or equal to, since the idea is this piece
    gets activated during silicon buring, after all the c + o from
    oxygen burning is gone.
    */

    if (tf.t9 > 2.5_rt && y(C12) + y(O16) <= 4.0e-3_rt) {

       t992  = tf.t972 * tf.t9;
       t9i92 = 1.0e0_rt/t992;

       yeff_ca40   = t9i92 * std::exp(239.42_rt*tf.t9i - 74.741_rt);
       yeff_ca40dt = -yeff_ca40*(239.42_rt*tf.t9i2 + 4.5e0_rt*tf.t9i);

       yeff_ti44   = t992  * std::exp(-274.12_rt*tf.t9i + 74.914_rt);
       yeff_ti44dt = yeff_ti44*(274.12_rt*tf.t9i2 + 4.5e0_rt*tf.t9i);

       denom     = std::pow(bden * y(He4), 3.0e0_rt);

       rate(Si28_7He4_to_Ni56_forward)     = yeff_ca40*denom*rate(Ca40_He4_to_Ti44_forward);
       dratedt(Si28_7He4_to_Ni56_forward)  = (yeff_ca40dt*rate(Ca40_He4_to_Ti44_forward)
                                              + yeff_ca40*dratedt(Ca40_He4_to_Ti44_forward))*denom*1.0e-9_rt;

       if (denom != 0.0_rt) {

          zz     = 1.0e0_rt/denom;
          rate(Si28_7He4_to_Ni56_reverse) = amrex::min(1.0e10_rt,yeff_ti44*rate(Ca40_He4_to_Ti44_reverse)*zz);

          if (rate(Si28_7He4_to_Ni56_reverse) == 1.0e10_rt) {
             dratedt(Si28_7He4_to_Ni56_reverse)  = 0.0e0_rt;
          } else {
             dratedt(Si28_7He4_to_Ni56_reverse)  = (yeff_ti44dt*rate(Ca40_He4_to_Ti44_reverse)
                                                    + yeff_ti44*dratedt(Ca40_He4_to_Ti44_reverse))*zz*1.0e-9_rt;
          }

       }

    }
}



// RHS version (single rate group)
AMREX_GPU_HOST_DEVICE AMREX_INLINE
void evaluate_rates(burn_t const& state, rate_t& rr)
{
    Real rho, temp;
    Array1D<Real, 1, NumSpec> y;

    // Temporary arrays for storing unused derivatives.
    // A useful cleanup would be to rewrite iso7tab, iso7rat,
    // and screen_iso7 so that there are versions that do not
    // evaluate the derivatives.
    Array1D<Real, 1, Rates::NumRates> dratedt;
    Array2D<Real, 1, NumSpec, 1, Rates::NumRates> dratedY;

    // Get the data from the state
    rho  = state.rho;
    temp = state.T;

    for (int i = 1; i <= NumSpec; ++i) {
        y(i) = state.xn[i-1] * aion_inv[i-1];
    }

    // Get the raw reaction rates
    if (use_tables)
        iso7tab(temp, rho, rr.rates, dratedt);
    else
        iso7rat(temp, rho, rr.rates, dratedt);

    // Do the screening (depends on composition)
    screen_iso7(temp, rho, y, rr.rates, dratedt, dratedY);
}


AMREX_GPU_HOST_DEVICE AMREX_INLINE
void get_rates(burn_t const& state, Array1D<rate_t, 1, Rates::NumGroups>& rr)
{
    Real rho, temp;
    Array1D<Real, 1, NumSpec> y;
    Array1D<Real, 1, Rates::NumRates> rate, dratedt;
    Array2D<Real, 1, NumSpec, 1, Rates::NumRates> dratedY;

    // Get the data from the state
    rho  = state.rho;
    temp = state.T;

    for (int i = 1; i <= NumSpec; ++i)
        y(i) = state.xn[i-1] * aion_inv[i-1];

    // Get the raw reaction rates
    if (use_tables)
        iso7tab(temp, rho, rate, dratedt);
    else
        iso7rat(temp, rho, rate, dratedt);

    // Do the screening here because the corrections depend on the composition
    screen_iso7(temp, rho, y, rate, dratedt, dratedY);

    // Save the rate data, for the Jacobian later if we need it.
    for (int i = 1; i <= Rates::NumRates; ++i) {
        rr(1).rates(i) = rate(i);
        rr(2).rates(i) = dratedt(i);
    }

    for (int j = 1; j <= NumSpec; ++j) {
        for (int i = 1; i <= Rates::NumRates; ++i) {
            rr(j+2).rates(i) = dratedY(j, i);
        }
    }
}


template<class MatrixType>
AMREX_GPU_HOST_DEVICE AMREX_INLINE
void dfdy_isotopes_iso7(Array1D<Real, 1, NumSpec> const& y,
                        burn_t const& state, Array1D<rate_t, 1, Rates::NumGroups> const& rr,
                        MatrixType& jac)
{

    // this routine sets up the dense iso7 jacobian for the isotopes

    using namespace Species;
    using namespace Rates;

    auto spec_jac = RHS::species_jac(state, rr(1));

    for (int spec1 = 1; spec1 <= NumSpec; ++spec1) {
        for (int spec2 = 1; spec2 <= NumSpec; ++spec2) {
            jac(spec1, spec2) = spec_jac(spec1, spec2);
        }
    }

}

template<class T>
AMREX_GPU_HOST_DEVICE AMREX_INLINE
void ener_gener_rate(T const& dydt, Real& enuc)
{

    using namespace network;

    // Computes the instantaneous energy generation rate

    Real Xdot = 0.0_rt;

    // Sum the mass fraction time derivatives
    for (int i = 1; i <= NumSpec; ++i) {
        Xdot += dydt(i) * mion(i);
    }

    // This is basically e = m c**2
    enuc = Xdot * C::Legacy::enuc_conv2;

}


AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
void actual_rhs(burn_t& state, Array1D<Real, 1, neqs>& ydot)
{
    // Evaluate the rates

    rate_t rr;
    evaluate_rates(state, rr);

    // Call the RHS to get dydt

    Array1D<Real, 1, NumSpec> spec_rhs = RHS::species_rhs(state, rr);

    for (int n = 1; n <= NumSpec; ++n) {
        ydot(n) = spec_rhs(n);
    }

    // Instantaneous energy generation rate

    ydot(net_ienuc) = RHS::ener_rhs(state, spec_rhs);
}


// Analytical Jacobian
template<class MatrixType>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
void actual_jac(burn_t& state, MatrixType& jac)
{

    Array1D<rate_t, 1, Rates::NumGroups> rr;

    bool deriva;

    Real b1, sneut, dsneutdt, dsneutdd, snuda, snudz;

    Real rho, temp, abar, zbar;
    Array1D<Real, 1, NumSpec> y;

    jac.zero();

    get_rates(state, rr);

    // Get the data from the state

    rho  = state.rho;
    temp = state.T;
    abar = state.abar;
    zbar = state.zbar;

    for (int i = 1; i <= NumSpec; ++i)
        y(i) = state.xn[i-1] * aion_inv[i-1];

    // Species Jacobian elements with respect to other species

    dfdy_isotopes_iso7(y, state, rr, jac);

    // Energy generation rate Jacobian elements with respect to species

    for (int j = 1; j <= NumSpec; ++j) {
        auto jac_slice_2 = [&](int i) -> Real { return jac.get(i, j); };
        ener_gener_rate(jac_slice_2, jac(net_ienuc,j));
    }

    // Account for the thermal neutrino losses

    sneut5(temp, rho, abar, zbar, sneut, dsneutdt, dsneutdd, snuda, snudz);

    for (int j = 1; j <= NumSpec; ++j) {
       b1 = (-abar * abar * snuda + (zion[j-1] - zbar) * abar * snudz);
       jac.add(net_ienuc, j, -b1);
    }

    // Evaluate the Jacobian elements with respect to energy by
    // calling the RHS using d(rate) / dT and then transform them
    // to our energy integration variable.

    const int index_rate = 2;

    Array1D<Real, 1, NumSpec> yderivs = RHS::species_rhs(state, rr(index_rate));

    for (int i = 1; i <= NumSpec; ++i) {
        jac(i, net_ienuc) = temperature_to_energy_jacobian(state, yderivs(i));
    }

    Real jac_e_T;
    ener_gener_rate(yderivs, jac_e_T);
    jac_e_T -= dsneutdt;
    jac(net_ienuc, net_ienuc) = temperature_to_energy_jacobian(state, jac_e_T);

}


AMREX_INLINE
void set_up_screening_factors()
{
    // Compute and store the more expensive screening factors

    using namespace Species;

    // note: we need to set these up in the same order that we evaluate the
    // rates in actual_rhs.H (yes, it's ugly)
    int jscr = 0;
    add_screening_factor(jscr++, zion[He4-1], aion[He4-1], zion[He4-1], aion[He4-1]);
    add_screening_factor(jscr++, zion[He4-1], aion[He4-1],  4.0e0_rt,  8.0e0_rt);
    add_screening_factor(jscr++, zion[C12-1], aion[C12-1], zion[He4-1], aion[He4-1]);
    add_screening_factor(jscr++, zion[C12-1], aion[C12-1], zion[C12-1], aion[C12-1]);
    add_screening_factor(jscr++, zion[C12-1], aion[C12-1], zion[O16-1], aion[O16-1]);
    add_screening_factor(jscr++, zion[O16-1], aion[O16-1], zion[O16-1], aion[O16-1]);
    add_screening_factor(jscr++, zion[O16-1], aion[O16-1], zion[He4-1], aion[He4-1]);
    add_screening_factor(jscr++, zion[Ne20-1], aion[Ne20-1], zion[He4-1], aion[He4-1]);
    add_screening_factor(jscr++, zion[Mg24-1], aion[Mg24-1], zion[He4-1], aion[He4-1]);
    add_screening_factor(jscr++,  20.0e0_rt,  40.0e0_rt, zion[He4-1], aion[He4-1]);

}

#endif
