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
#include <rhs_utilities.H>
#include <jacobian_utilities.H>

using namespace amrex;
using namespace ArrayUtil;

void actual_rhs_init();

namespace RateTable
{
    constexpr Real tab_tlo = 6.0e0_rt;
    constexpr Real tab_thi = 10.0e0_rt;
    constexpr int tab_per_decade = 500;
    constexpr int nrattab = static_cast<int>(tab_thi - tab_tlo) * tab_per_decade + 1;
    constexpr int tab_imax = static_cast<int>(tab_thi - tab_tlo) * tab_per_decade + 1;
    constexpr Real tab_tstp = (tab_thi - tab_tlo) / static_cast<Real>(tab_imax - 1);

    extern AMREX_GPU_MANAGED Array2D<Real, 1, Rates::NumRates, 1, nrattab> rattab;
    extern AMREX_GPU_MANAGED Array2D<Real, 1, Rates::NumRates, 1, nrattab> drattabdt;
    extern AMREX_GPU_MANAGED Array1D<Real, 1, nrattab> ttab;
}


AMREX_GPU_HOST_DEVICE AMREX_INLINE
void aprox13tab(const Real btemp, const Real bden, Array1D<rate_t, 1, Rates::NumGroups>& rr)
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
    for (int rate = 1; rate <= Rates::NumRates; ++rate) {

        Real dtab = 0.0_rt;

        // Set the density dependence
        switch (rate) {

        case (He4_He4_He4_to_C12_forward):
            dtab = bden*bden;
            break;

        case (He4_He4_He4_to_C12_reverse):
            dtab = 1.0e0_rt;
            break;

        case (C12_He4_to_O16_forward):
            dtab = bden;
            break;

        case (C12_He4_to_O16_reverse):
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

        case (O16_O16_to_S32_forward):
            dtab = bden;
            break;

        case (O16_O16_to_S32_reverse):
            dtab = 1.0_rt; // rate is zero in this net
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

        case (irmgap):
            dtab = bden;
            break;

        case (iralpa):
            dtab = bden;
            break;

        case (iralpg):
            dtab = bden;
            break;

        case (irsigp):
            dtab = 1.0e0_rt;
            break;

        case (Si28_He4_to_S32_forward):
            dtab = bden;
            break;

        case (Si28_He4_to_S32_reverse):
            dtab = 1.0e0_rt;
            break;

        case (irppa):
            dtab = bden;
            break;

        case (irsiap):
            dtab = bden;
            break;

        case (irppg):
            dtab = bden;
            break;

        case (irsgp):
            dtab = 1.0e0_rt;
            break;

        case (S32_He4_to_Ar36_forward):
            dtab = bden;
            break;

        case (S32_He4_to_Ar36_reverse):
            dtab = 1.0e0_rt;
            break;

        case (irsap):
            dtab = bden;
            break;

        case (irclpa):
            dtab = bden;
            break;

        case (irclpg):
            dtab = bden;
            break;

        case (irargp):
            dtab = 1.0e0_rt;
            break;

        case (Ar36_He4_to_Ca40_forward):
            dtab = bden;
            break;

        case (Ar36_He4_to_Ca40_reverse):
            dtab = 1.0e0_rt;
            break;

        case (irarap):
            dtab = bden;
            break;

        case (irkpa):
            dtab = bden;
            break;

        case (irkpg):
            dtab = bden;
            break;

        case (ircagp):
            dtab = 1.0e0_rt;
            break;

        case (Ca40_He4_to_Ti44_forward):
            dtab = bden;
            break;

        case (Ca40_He4_to_Ti44_reverse):
            dtab = 1.0e0_rt;
            break;

        case (ircaap):
            dtab = bden;
            break;

        case (irscpa):
            dtab = bden;
            break;

        case (irscpg):
            dtab = bden;
            break;

        case (irtigp):
            dtab = 1.0e0_rt;
            break;

        case (Ti44_He4_to_Cr48_forward):
            dtab = bden;
            break;

        case (Ti44_He4_to_Cr48_reverse):
            dtab = 1.0e0_rt;
            break;

        case (irtiap):
            dtab = bden;
            break;

        case (irvpa):
            dtab = bden;
            break;

        case (irvpg):
            dtab = bden;
            break;

        case (ircrgp):
            dtab = 1.0e0_rt;
            break;

        case (Cr48_He4_to_Fe52_forward):
            dtab = bden;
            break;

        case (Cr48_He4_to_Fe52_reverse):
            dtab = 1.0e0_rt;
            break;

        case (ircrap):
            dtab = bden;
            break;

        case (irmnpa):
            dtab = bden;
            break;

        case (irmnpg):
            dtab = bden;
            break;

        case (irfegp):
            dtab = 1.0e0_rt;
            break;

        case (Fe52_He4_to_Ni56_forward):
            dtab = bden;
            break;

        case (Fe52_He4_to_Ni56_reverse):
            dtab = 1.0e0_rt;
            break;

        case (irfeap):
            dtab = bden;
            break;

        case (ircopa):
            dtab = bden;
            break;

        case (ircopg):
            dtab = bden;
            break;

        case (irnigp):
            dtab = 1.0e0_rt;
            break;
        }

        rr(1).rates(rate) = (alfa * rattab(rate, iat  ) +
                             beta * rattab(rate, iat+1) +
                             gama * rattab(rate, iat+2) +
                             delt * rattab(rate, iat+3)) * dtab;

        rr(2).rates(rate) = (alfa * drattabdt(rate, iat  ) +
                             beta * drattabdt(rate, iat+1) +
                             gama * drattabdt(rate, iat+2) +
                             delt * drattabdt(rate, iat+3)) * dtab;

    }
}


AMREX_GPU_HOST_DEVICE AMREX_INLINE
void aprox13rat(const Real btemp, const Real bden, Array1D<rate_t, 1, Rates::NumGroups>& rr)
{
    using namespace Rates;

    // this routine generates unscreened
    // nuclear reaction rates for the aprox13 network.

    Real rrate,drratedt;

    for (int i = 1; i <= Rates::NumRates; ++i) {
       rr(1).rates(i) = 0.0_rt;
       rr(2).rates(i) = 0.0_rt;
    }

    if (btemp < 1.0e6_rt) return;


    // get the temperature factors
    tf_t tf = get_tfactors(btemp);

    // c12(a,g)o16
    rate_c12ag(tf,bden,
               rr(1).rates(C12_He4_to_O16_forward),rr(2).rates(C12_He4_to_O16_forward),
               rr(1).rates(C12_He4_to_O16_reverse),rr(2).rates(C12_He4_to_O16_reverse));

    // triple alpha to c12
    rate_triplealf(tf,bden,
                   rr(1).rates(He4_He4_He4_to_C12_forward),rr(2).rates(He4_He4_He4_to_C12_forward),
                   rr(1).rates(He4_He4_He4_to_C12_reverse),rr(2).rates(He4_He4_He4_to_C12_reverse));

    // c12 + c12
    rate_c12c12(tf,bden,
                rr(1).rates(C12_C12_to_Ne20_He4_forward),rr(2).rates(C12_C12_to_Ne20_He4_forward),
                rrate,drratedt);

    // c12 + o16
    rate_c12o16(tf,bden,
                rr(1).rates(C12_O16_to_Mg24_He4_forward),rr(2).rates(C12_O16_to_Mg24_He4_forward),
                rrate,drratedt);

    // o16 + o16
    rate_o16o16(tf,bden,
                rr(1).rates(O16_O16_to_Si28_He4_forward),rr(2).rates(O16_O16_to_Si28_He4_forward),
                rrate,drratedt);

    // o16(a,g)ne20
    rate_o16ag(tf,bden,
               rr(1).rates(O16_He4_to_Ne20_forward),rr(2).rates(O16_He4_to_Ne20_forward),
               rr(1).rates(O16_He4_to_Ne20_reverse),rr(2).rates(O16_He4_to_Ne20_reverse));

    // ne20(a,g)mg24
    rate_ne20ag(tf,bden,
                rr(1).rates(Ne20_He4_to_Mg24_forward),rr(2).rates(Ne20_He4_to_Mg24_forward),
                rr(1).rates(Ne20_He4_to_Mg24_reverse),rr(2).rates(Ne20_He4_to_Mg24_reverse));

    // mg24(a,g)si28
    rate_mg24ag(tf,bden,
                rr(1).rates(Mg24_He4_to_Si28_forward),rr(2).rates(Mg24_He4_to_Si28_forward),
                rr(1).rates(Mg24_He4_to_Si28_reverse),rr(2).rates(Mg24_He4_to_Si28_reverse));

    // mg24(a,p)al27
    rate_mg24ap(tf,bden,
                rr(1).rates(irmgap),rr(2).rates(irmgap),
                rr(1).rates(iralpa),rr(2).rates(iralpa));

    // al27(p,g)si28
    rate_al27pg(tf,bden,
                rr(1).rates(iralpg),rr(2).rates(iralpg),
                rr(1).rates(irsigp),rr(2).rates(irsigp));

    // si28(a,g)s32
    rate_si28ag(tf,bden,
                rr(1).rates(Si28_He4_to_S32_forward),rr(2).rates(Si28_He4_to_S32_forward),
                rr(1).rates(Si28_He4_to_S32_reverse),rr(2).rates(Si28_He4_to_S32_reverse));

    // si28(a,p)p31
    rate_si28ap(tf,bden,
                rr(1).rates(irsiap),rr(2).rates(irsiap),
                rr(1).rates(irppa),rr(2).rates(irppa));

    // p31(p,g)s32
    rate_p31pg(tf,bden,
               rr(1).rates(irppg),rr(2).rates(irppg),
               rr(1).rates(irsgp),rr(2).rates(irsgp));

    // s32(a,g)ar36
    rate_s32ag(tf,bden,
               rr(1).rates(S32_He4_to_Ar36_forward),rr(2).rates(S32_He4_to_Ar36_forward),
               rr(1).rates(S32_He4_to_Ar36_reverse),rr(2).rates(S32_He4_to_Ar36_reverse));

    // s32(a,p)cl35
    rate_s32ap(tf,bden,
               rr(1).rates(irsap),rr(2).rates(irsap),
               rr(1).rates(irclpa),rr(2).rates(irclpa));

    // cl35(p,g)ar36
    rate_cl35pg(tf,bden,
                rr(1).rates(irclpg),rr(2).rates(irclpg),
                rr(1).rates(irargp),rr(2).rates(irargp));

    // ar36(a,g)ca40
    rate_ar36ag(tf,bden,
                rr(1).rates(Ar36_He4_to_Ca40_forward),rr(2).rates(Ar36_He4_to_Ca40_forward),
                rr(1).rates(Ar36_He4_to_Ca40_reverse),rr(2).rates(Ar36_He4_to_Ca40_reverse));

    // ar36(a,p)k39
    rate_ar36ap(tf,bden,
                rr(1).rates(irarap),rr(2).rates(irarap),
                rr(1).rates(irkpa),rr(2).rates(irkpa));

    // k39(p,g)ca40
    rate_k39pg(tf,bden,
               rr(1).rates(irkpg),rr(2).rates(irkpg),
               rr(1).rates(ircagp),rr(2).rates(ircagp));

    // ca40(a,g)ti44
    rate_ca40ag(tf,bden,
                rr(1).rates(Ca40_He4_to_Ti44_forward),rr(2).rates(Ca40_He4_to_Ti44_forward),
                rr(1).rates(Ca40_He4_to_Ti44_reverse),rr(2).rates(Ca40_He4_to_Ti44_reverse));

    // ca40(a,p)sc43
    rate_ca40ap(tf,bden,
                rr(1).rates(ircaap),rr(2).rates(ircaap),
                rr(1).rates(irscpa),rr(2).rates(irscpa));

    // sc43(p,g)ti44
    rate_sc43pg(tf,bden,
                rr(1).rates(irscpg),rr(2).rates(irscpg),
                rr(1).rates(irtigp),rr(2).rates(irtigp));

    // ti44(a,g)cr48
    rate_ti44ag(tf,bden,
                rr(1).rates(Ti44_He4_to_Cr48_forward),rr(2).rates(Ti44_He4_to_Cr48_forward),
                rr(1).rates(Ti44_He4_to_Cr48_reverse),rr(2).rates(Ti44_He4_to_Cr48_reverse));

    // ti44(a,p)v47
    rate_ti44ap(tf,bden,
                rr(1).rates(irtiap),rr(2).rates(irtiap), 
                rr(1).rates(irvpa),rr(2).rates(irvpa));

    // v47(p,g)cr48
    rate_v47pg(tf,bden,
               rr(1).rates(irvpg),rr(2).rates(irvpg), 
               rr(1).rates(ircrgp),rr(2).rates(ircrgp));

    // cr48(a,g)fe52
    rate_cr48ag(tf,bden,
                rr(1).rates(Cr48_He4_to_Fe52_forward),rr(2).rates(Cr48_He4_to_Fe52_forward),
                rr(1).rates(Cr48_He4_to_Fe52_reverse),rr(2).rates(Cr48_He4_to_Fe52_reverse));

    // cr48(a,p)mn51
    rate_cr48ap(tf,bden,
                rr(1).rates(ircrap),rr(2).rates(ircrap),
                rr(1).rates(irmnpa),rr(2).rates(irmnpa));

    // mn51(p,g)fe52
    rate_mn51pg(tf,bden,
                rr(1).rates(irmnpg),rr(2).rates(irmnpg),
                rr(1).rates(irfegp),rr(2).rates(irfegp));

    // fe52(a,g)ni56
    rate_fe52ag(tf,bden,
                rr(1).rates(Fe52_He4_to_Ni56_forward),rr(2).rates(Fe52_He4_to_Ni56_forward),
                rr(1).rates(Fe52_He4_to_Ni56_reverse),rr(2).rates(Fe52_He4_to_Ni56_reverse));

    // fe52(a,p)co55
    rate_fe52ap(tf,bden,
                rr(1).rates(irfeap),rr(2).rates(irfeap),
                rr(1).rates(ircopa),rr(2).rates(ircopa));

    // co55(p,g)ni56
    rate_co55pg(tf,bden,
                rr(1).rates(ircopg),rr(2).rates(ircopg),
                rr(1).rates(irnigp),rr(2).rates(irnigp));
}


AMREX_GPU_HOST_DEVICE AMREX_INLINE
void set_aprox13rat()
{
    using namespace RateTable;

    Real btemp;
    Real bden = 1.0e0_rt;
    Array1D<rate_t, 1, Rates::NumGroups> rr;

    for (int i = 1; i <= tab_imax; ++i) {

       btemp = tab_tlo + static_cast<Real>(i-1) * tab_tstp;
       btemp = std::pow(10.0e0_rt, btemp);

       aprox13rat(btemp, bden, rr);

       ttab(i) = btemp;

       for (int j = 1; j <= Rates::NumRates; ++j) {

          rattab(j,i)    = rr(1).rates(j);
          drattabdt(j,i) = rr(2).rates(j);

       }
    }
}


AMREX_GPU_HOST_DEVICE AMREX_INLINE
void screen_aprox13(const Real btemp, const Real bden,
                    Array1D<Real, 1, NumSpec> const& y,
                    Array1D<rate_t, 1, Rates::NumGroups>& rr)
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
    Real sc1a,sc1adt,sc2a,sc2adt,sc3a,sc3adt;
    Real sc1add,sc2add;
    Real denom,denomdt,zz;
    Real ratraw;
    plasma_state_t state;

    // Set up the state data, which is the same for all screening factors.

    fill_plasma_state(state, btemp, bden, y);

    // first the always fun triple alpha and its inverse
    jscr = 0;
    screen5(state,jscr,
            zion[He4-1], aion[He4-1], zion[He4-1], aion[He4-1],
            sc1a,sc1adt,sc1add);

    jscr++;
    screen5(state,jscr,
            zion[He4-1], aion[He4-1], 4.0_rt, 8.0_rt,
            sc2a,sc2adt,sc2add);

    sc3a   = sc1a * sc2a;
    sc3adt = sc1adt*sc2a + sc1a*sc2adt;

    ratraw = rr(1).rates(He4_He4_He4_to_C12_forward);
    rr(1).rates(He4_He4_He4_to_C12_forward) = ratraw * sc3a;
    rr(2).rates(He4_He4_He4_to_C12_forward) = rr(2).rates(He4_He4_He4_to_C12_forward)*sc3a + ratraw*sc3adt;

    ratraw = rr(1).rates(He4_He4_He4_to_C12_reverse);
    rr(1).rates(He4_He4_He4_to_C12_reverse) = ratraw * sc3a;
    rr(2).rates(He4_He4_He4_to_C12_reverse) = rr(2).rates(He4_He4_He4_to_C12_reverse)*sc3a + ratraw*sc3adt;

    // c12 to o16
    // c12(a,g)o16
    jscr++;
    screen5(state,jscr,
            zion[C12-1], aion[C12-1], zion[He4-1], aion[He4-1],
            sc1a,sc1adt,sc1add);

    ratraw = rr(1).rates(C12_He4_to_O16_forward);
    rr(1).rates(C12_He4_to_O16_forward)  = ratraw * sc1a;
    rr(2).rates(C12_He4_to_O16_forward)  = rr(2).rates(C12_He4_to_O16_forward)*sc1a + ratraw*sc1adt;

    ratraw = rr(1).rates(C12_He4_to_O16_reverse);
    rr(1).rates(C12_He4_to_O16_reverse)  = ratraw * sc1a;
    rr(2).rates(C12_He4_to_O16_reverse)  = rr(2).rates(C12_He4_to_O16_reverse)*sc1a + ratraw*sc1adt;

    // c12 + c12
    jscr++;
    screen5(state,jscr,
            zion[C12-1], aion[C12-1], zion[C12-1], aion[C12-1],
            sc1a,sc1adt,sc1add);

    ratraw = rr(1).rates(C12_C12_to_Ne20_He4_forward);
    rr(1).rates(C12_C12_to_Ne20_He4_forward) = ratraw * sc1a;
    rr(2).rates(C12_C12_to_Ne20_He4_forward) = rr(2).rates(C12_C12_to_Ne20_He4_forward)*sc1a + ratraw*sc1adt;

    // c12 + o16
    jscr++;
    screen5(state,jscr,
            zion[C12-1], aion[C12-1], zion[O16-1], aion[O16-1], 
            sc1a,sc1adt,sc1add);

    ratraw = rr(1).rates(C12_O16_to_Mg24_He4_forward);
    rr(1).rates(C12_O16_to_Mg24_He4_forward) = ratraw * sc1a;
    rr(2).rates(C12_O16_to_Mg24_He4_forward) = rr(2).rates(C12_O16_to_Mg24_He4_forward)*sc1a + ratraw*sc1adt;

    // Equal probability branching ratio between (Mg24 + He4) and (Si28) endpoints

    rr(1).rates(C12_O16_to_Mg24_He4_forward) *= 0.5;
    rr(2).rates(C12_O16_to_Mg24_He4_forward) *= 0.5;

    rr(1).rates(C12_O16_to_Si28_forward) = rr(1).rates(C12_O16_to_Mg24_He4_forward);
    rr(2).rates(C12_O16_to_Si28_forward) = rr(2).rates(C12_O16_to_Mg24_He4_forward);

    // o16 + o16
    jscr++;
    screen5(state,jscr,
            zion[O16-1], aion[O16-1], zion[O16-1], aion[O16-1],
            sc1a,sc1adt,sc1add);

    ratraw = rr(1).rates(O16_O16_to_Si28_He4_forward);
    rr(1).rates(O16_O16_to_Si28_He4_forward) = ratraw * sc1a;
    rr(2).rates(O16_O16_to_Si28_He4_forward) = rr(2).rates(O16_O16_to_Si28_He4_forward)*sc1a + ratraw*sc1adt;

    Real O16_O16_rate = rr(1).rates(O16_O16_to_Si28_He4_forward);
    Real O16_O16_dratedt = rr(2).rates(O16_O16_to_Si28_He4_forward);

    // o16 to ne20
    jscr++;
    screen5(state,jscr,
            zion[O16-1], aion[O16-1], zion[He4-1], aion[He4-1],
            sc1a,sc1adt,sc1add);


    // o16(a,g)ne20
    ratraw = rr(1).rates(O16_He4_to_Ne20_forward);
    rr(1).rates(O16_He4_to_Ne20_forward) = ratraw * sc1a;
    rr(2).rates(O16_He4_to_Ne20_forward) = rr(2).rates(O16_He4_to_Ne20_forward)*sc1a + ratraw*sc1adt;

    ratraw = rr(1).rates(O16_He4_to_Ne20_reverse);
    rr(1).rates(O16_He4_to_Ne20_reverse) = ratraw * sc1a;
    rr(2).rates(O16_He4_to_Ne20_reverse) = rr(2).rates(O16_He4_to_Ne20_reverse)*sc1a + ratraw*sc1adt;

    // ne20 to mg24
    jscr++;
    screen5(state,jscr,
            zion[Ne20-1], aion[Ne20-1], zion[He4-1], aion[He4-1],
            sc1a,sc1adt,sc1add);


    // ne20(a,g)mg24
    ratraw = rr(1).rates(Ne20_He4_to_Mg24_forward);
    rr(1).rates(Ne20_He4_to_Mg24_forward) = ratraw * sc1a;
    rr(2).rates(Ne20_He4_to_Mg24_forward) = rr(2).rates(Ne20_He4_to_Mg24_forward)*sc1a + ratraw*sc1adt;

    ratraw = rr(1).rates(Ne20_He4_to_Mg24_reverse);
    rr(1).rates(Ne20_He4_to_Mg24_reverse) = ratraw * sc1a;
    rr(2).rates(Ne20_He4_to_Mg24_reverse) = rr(2).rates(Ne20_He4_to_Mg24_reverse)*sc1a + ratraw*sc1adt;

    // mg24 to si28
    jscr++;
    screen5(state,jscr,
            zion[Mg24-1], aion[Mg24-1], zion[He4-1], aion[He4-1],
            sc1a,sc1adt,sc1add);


    // mg24(a,g)si28
    ratraw = rr(1).rates(Mg24_He4_to_Si28_forward);
    rr(1).rates(Mg24_He4_to_Si28_forward) = ratraw * sc1a;
    rr(2).rates(Mg24_He4_to_Si28_forward) = rr(2).rates(Mg24_He4_to_Si28_forward)*sc1a + ratraw*sc1adt;

    ratraw = rr(1).rates(Mg24_He4_to_Si28_reverse);
    rr(1).rates(Mg24_He4_to_Si28_reverse) = ratraw * sc1a;
    rr(2).rates(Mg24_He4_to_Si28_reverse) = rr(2).rates(Mg24_He4_to_Si28_reverse)*sc1a + ratraw*sc1adt;


    // mg24(a,p)al27
    ratraw = rr(1).rates(irmgap);
    rr(1).rates(irmgap) = ratraw * sc1a;
    rr(2).rates(irmgap) = rr(2).rates(irmgap)*sc1a + ratraw*sc1adt;

    ratraw = rr(1).rates(iralpa);
    rr(1).rates(iralpa) = ratraw * sc1a;
    rr(2).rates(iralpa) = rr(2).rates(iralpa)*sc1a + ratraw*sc1adt;

    jscr++;
    screen5(state,jscr,
            13.0_rt, 27.0_rt, 1.0_rt, 1.0_rt,
            sc1a,sc1adt,sc1add);

    // al27(p,g)si28
    ratraw = rr(1).rates(iralpg);
    rr(1).rates(iralpg) = ratraw * sc1a;
    rr(2).rates(iralpg) = rr(2).rates(iralpg)*sc1a + ratraw*sc1adt;

    ratraw = rr(1).rates(irsigp);
    rr(1).rates(irsigp) = ratraw * sc1a;
    rr(2).rates(irsigp) = rr(2).rates(irsigp)*sc1a + ratraw*sc1adt;


    // si28 to s32
    jscr++;
    screen5(state,jscr,
            zion[Si28-1], aion[Si28-1], zion[He4-1], aion[He4-1],
            sc1a,sc1adt,sc1add);


    // si28(a,g)s32
    ratraw = rr(1).rates(Si28_He4_to_S32_forward);
    rr(1).rates(Si28_He4_to_S32_forward) = ratraw * sc1a;
    rr(2).rates(Si28_He4_to_S32_forward) = rr(2).rates(Si28_He4_to_S32_forward)*sc1a + ratraw*sc1adt;

    ratraw = rr(1).rates(Si28_He4_to_S32_reverse);
    rr(1).rates(Si28_He4_to_S32_reverse) = ratraw * sc1a;
    rr(2).rates(Si28_He4_to_S32_reverse) = rr(2).rates(Si28_He4_to_S32_reverse)*sc1a + ratraw*sc1adt;


    // si28(a,p)p31
    ratraw = rr(1).rates(irsiap);
    rr(1).rates(irsiap) = ratraw * sc1a;
    rr(2).rates(irsiap) = rr(2).rates(irsiap)*sc1a + ratraw*sc1adt;

    ratraw = rr(1).rates(irppa);
    rr(1).rates(irppa)  = ratraw * sc1a;
    rr(2).rates(irppa)  = rr(2).rates(irppa)*sc1a  + ratraw*sc1adt;


    jscr++;
    screen5(state,jscr,
            15.0_rt, 31.0_rt, 1.0_rt, 1.0_rt,
            sc1a,sc1adt,sc1add);

    // p31(p,g)s32
    ratraw = rr(1).rates(irppg);
    rr(1).rates(irppg)  = ratraw * sc1a;
    rr(2).rates(irppg)  = rr(2).rates(irppg)*sc1a + ratraw*sc1adt;

    ratraw = rr(1).rates(irsgp);
    rr(1).rates(irsgp)  = ratraw * sc1a;
    rr(2).rates(irsgp)  = rr(2).rates(irsgp)*sc1a + ratraw*sc1adt;


    // s32 to ar36
    jscr++;
    screen5(state,jscr,
            zion[S32-1], aion[S32-1], zion[He4-1], aion[He4-1], 
            sc1a,sc1adt,sc1add);


    // s32(a,g)ar36
    ratraw = rr(1).rates(S32_He4_to_Ar36_forward);
    rr(1).rates(S32_He4_to_Ar36_forward)  = ratraw * sc1a;
    rr(2).rates(S32_He4_to_Ar36_forward)  = rr(2).rates(S32_He4_to_Ar36_forward)*sc1a + ratraw*sc1adt;

    ratraw = rr(1).rates(S32_He4_to_Ar36_reverse);
    rr(1).rates(S32_He4_to_Ar36_reverse)  = ratraw * sc1a;
    rr(2).rates(S32_He4_to_Ar36_reverse)  = rr(2).rates(S32_He4_to_Ar36_reverse)*sc1a + ratraw*sc1adt;

    // s32(a,p)cl35
    ratraw = rr(1).rates(irsap);
    rr(1).rates(irsap)  = ratraw * sc1a;
    rr(2).rates(irsap)  = rr(2).rates(irsap)*sc1a + ratraw*sc1adt;

    ratraw = rr(1).rates(irclpa);
    rr(1).rates(irclpa) = ratraw * sc1a;
    rr(2).rates(irclpa) = rr(2).rates(irclpa)*sc1a + ratraw*sc1adt;


    jscr++;
    screen5(state,jscr,
            17.0_rt, 35.0_rt, 1.0_rt, 1.0_rt,
            sc1a,sc1adt,sc1add);

    // cl35(p,g)ar36
    ratraw = rr(1).rates(irclpg);
    rr(1).rates(irclpg) = ratraw * sc1a;
    rr(2).rates(irclpg) = rr(2).rates(irclpg)*sc1a + ratraw*sc1adt;

    ratraw = rr(1).rates(irargp);
    rr(1).rates(irargp) = ratraw * sc1a;
    rr(2).rates(irargp) = rr(2).rates(irargp)*sc1a + ratraw*sc1adt;


    // ar36 to ca40
    jscr++;
    screen5(state,jscr,
            zion[Ar36-1], aion[Ar36-1], zion[He4-1], aion[He4-1], 
            sc1a,sc1adt,sc1add);


    // ar36(a,g)ca40
    ratraw = rr(1).rates(Ar36_He4_to_Ca40_forward);
    rr(1).rates(Ar36_He4_to_Ca40_forward) = ratraw * sc1a;
    rr(2).rates(Ar36_He4_to_Ca40_forward) = rr(2).rates(Ar36_He4_to_Ca40_forward)*sc1a + ratraw*sc1adt;

    ratraw = rr(1).rates(Ar36_He4_to_Ca40_reverse);
    rr(1).rates(Ar36_He4_to_Ca40_reverse) = ratraw * sc1a;
    rr(2).rates(Ar36_He4_to_Ca40_reverse) = rr(2).rates(Ar36_He4_to_Ca40_reverse)*sc1a + ratraw*sc1adt;


    // ar36(a,p)k39
    ratraw = rr(1).rates(irarap);
    rr(1).rates(irarap) = ratraw * sc1a;
    rr(2).rates(irarap) = rr(2).rates(irarap)*sc1a + ratraw*sc1adt;

    ratraw = rr(1).rates(irkpa);
    rr(1).rates(irkpa) = ratraw * sc1a;
    rr(2).rates(irkpa) = rr(2).rates(irkpa)*sc1a  + ratraw*sc1adt;


    jscr++;
    screen5(state,jscr,
            19.0_rt, 39.0_rt, 1.0_rt, 1.0_rt,
            sc1a,sc1adt,sc1add);

    // k39(p,g)ca40
    ratraw = rr(1).rates(irkpg);
    rr(1).rates(irkpg) = ratraw * sc1a;
    rr(2).rates(irkpg) = rr(2).rates(irkpg)*sc1a  + ratraw*sc1adt;

    ratraw = rr(1).rates(ircagp);
    rr(1).rates(ircagp) = ratraw * sc1a;
    rr(2).rates(ircagp) = rr(2).rates(ircagp)*sc1a  + ratraw*sc1adt;


    // ca40 to ti44
    jscr++;
    screen5(state,jscr,
            zion[Ca40-1], aion[Ca40-1], zion[He4-1], aion[He4-1], 
            sc1a,sc1adt,sc1add);


    // ca40(a,g)ti44
    ratraw = rr(1).rates(Ca40_He4_to_Ti44_forward);
    rr(1).rates(Ca40_He4_to_Ti44_forward) = ratraw * sc1a;
    rr(2).rates(Ca40_He4_to_Ti44_forward) = rr(2).rates(Ca40_He4_to_Ti44_forward)*sc1a + ratraw*sc1adt;

    ratraw = rr(1).rates(Ca40_He4_to_Ti44_reverse);
    rr(1).rates(Ca40_He4_to_Ti44_reverse) = ratraw * sc1a;
    rr(2).rates(Ca40_He4_to_Ti44_reverse) = rr(2).rates(Ca40_He4_to_Ti44_reverse)*sc1a + ratraw*sc1adt;


    // ca40(a,p)sc43
    ratraw = rr(1).rates(ircaap);
    rr(1).rates(ircaap) = ratraw * sc1a;
    rr(2).rates(ircaap) = rr(2).rates(ircaap)*sc1a + ratraw*sc1adt;

    ratraw = rr(1).rates(irscpa);
    rr(1).rates(irscpa) = ratraw * sc1a;
    rr(2).rates(irscpa) = rr(2).rates(irscpa)*sc1a + ratraw*sc1adt;


    jscr++;
    screen5(state,jscr,
            21.0_rt, 43.0_rt, 1.0_rt, 1.0_rt,
            sc1a,sc1adt,sc1add);

    // sc43(p,g)ti44
    ratraw = rr(1).rates(irscpg);
    rr(1).rates(irscpg) = ratraw * sc1a;
    rr(2).rates(irscpg) = rr(2).rates(irscpg)*sc1a + ratraw*sc1adt;

    ratraw = rr(1).rates(irtigp);
    rr(1).rates(irtigp) = ratraw * sc1a;
    rr(2).rates(irtigp) = rr(2).rates(irtigp)*sc1a + ratraw*sc1adt;


    // ti44 to cr48
    jscr++;
    screen5(state,jscr,
            zion[Ti44-1], aion[Ti44-1], zion[He4-1], aion[He4-1], 
            sc1a,sc1adt,sc1add);


    // ti44(a,g)cr48
    ratraw = rr(1).rates(Ti44_He4_to_Cr48_forward);
    rr(1).rates(Ti44_He4_to_Cr48_forward) = ratraw * sc1a;
    rr(2).rates(Ti44_He4_to_Cr48_forward) = rr(2).rates(Ti44_He4_to_Cr48_forward)*sc1a + ratraw*sc1adt;

    ratraw = rr(1).rates(Ti44_He4_to_Cr48_reverse);
    rr(1).rates(Ti44_He4_to_Cr48_reverse) = ratraw * sc1a;
    rr(2).rates(Ti44_He4_to_Cr48_reverse) = rr(2).rates(Ti44_He4_to_Cr48_reverse)*sc1a + ratraw*sc1adt;

    // ti44(a,p)v47
    ratraw = rr(1).rates(irtiap);
    rr(1).rates(irtiap) = ratraw * sc1a;
    rr(2).rates(irtiap) = rr(2).rates(irtiap)*sc1a + ratraw*sc1adt;

    ratraw = rr(1).rates(irvpa);
    rr(1).rates(irvpa) = ratraw * sc1a;
    rr(2).rates(irvpa) = rr(2).rates(irvpa)*sc1a  + ratraw*sc1adt;


    jscr++;
    screen5(state,jscr,
            23.0_rt, 47.0_rt, 1.0_rt, 1.0_rt,
            sc1a,sc1adt,sc1add);

    // v47(p,g)cr48
    ratraw = rr(1).rates(irvpg);
    rr(1).rates(irvpg) = ratraw * sc1a;
    rr(2).rates(irvpg) = rr(2).rates(irvpg)*sc1a  + ratraw*sc1adt;

    ratraw = rr(1).rates(ircrgp);
    rr(1).rates(ircrgp)  = ratraw * sc1a;
    rr(2).rates(ircrgp)  = rr(2).rates(ircrgp)*sc1a  + ratraw*sc1adt;


    // cr48 to fe52
    jscr++;
    screen5(state,jscr,
            zion[Cr48-1], aion[Cr48-1], zion[He4-1], aion[He4-1], 
            sc1a,sc1adt,sc1add);


    // cr48(a,g)fe52
    ratraw = rr(1).rates(Cr48_He4_to_Fe52_forward);
    rr(1).rates(Cr48_He4_to_Fe52_forward) = ratraw * sc1a;
    rr(2).rates(Cr48_He4_to_Fe52_forward) = rr(2).rates(Cr48_He4_to_Fe52_forward)*sc1a + ratraw*sc1adt;

    ratraw = rr(1).rates(Cr48_He4_to_Fe52_reverse);
    rr(1).rates(Cr48_He4_to_Fe52_reverse) = ratraw * sc1a;
    rr(2).rates(Cr48_He4_to_Fe52_reverse) = rr(2).rates(Cr48_He4_to_Fe52_reverse)*sc1a + ratraw*sc1adt;


    // cr48(a,p)mn51
    ratraw = rr(1).rates(ircrap);
    rr(1).rates(ircrap) = ratraw * sc1a;
    rr(2).rates(ircrap) = rr(2).rates(ircrap)*sc1a + ratraw*sc1adt;

    ratraw = rr(1).rates(irmnpa);
    rr(1).rates(irmnpa) = ratraw * sc1a;
    rr(2).rates(irmnpa) = rr(2).rates(irmnpa)*sc1a + ratraw*sc1adt;


    jscr++;
    screen5(state,jscr,
            25.0_rt, 51.0_rt, 1.0_rt, 1.0_rt,
            sc1a,sc1adt,sc1add);

    // mn51(p,g)fe52
    ratraw = rr(1).rates(irmnpg);
    rr(1).rates(irmnpg) = ratraw * sc1a;
    rr(2).rates(irmnpg) = rr(2).rates(irmnpg)*sc1a + ratraw*sc1adt;

    ratraw = rr(1).rates(irfegp);
    rr(1).rates(irfegp) = ratraw * sc1a;
    rr(2).rates(irfegp) = rr(2).rates(irfegp)*sc1a + ratraw*sc1adt;


    // fe52 to ni56
    jscr++;
    screen5(state,jscr,
            zion[Fe52-1], aion[Fe52-1], zion[He4-1], aion[He4-1], 
            sc1a,sc1adt,sc1add);


    // fe52(a,g)ni56
    ratraw = rr(1).rates(Fe52_He4_to_Ni56_forward);
    rr(1).rates(Fe52_He4_to_Ni56_forward) = ratraw * sc1a;
    rr(2).rates(Fe52_He4_to_Ni56_forward) = rr(2).rates(Fe52_He4_to_Ni56_forward)*sc1a + ratraw*sc1adt;

    ratraw = rr(1).rates(Fe52_He4_to_Ni56_reverse);
    rr(1).rates(Fe52_He4_to_Ni56_reverse) = ratraw * sc1a;
    rr(2).rates(Fe52_He4_to_Ni56_reverse) = rr(2).rates(Fe52_He4_to_Ni56_reverse)*sc1a + ratraw*sc1adt;


    // fe52(a,p)co55
    ratraw = rr(1).rates(irfeap);
    rr(1).rates(irfeap) = ratraw * sc1a;
    rr(2).rates(irfeap) = rr(2).rates(irfeap)*sc1a + ratraw*sc1adt;

    ratraw = rr(1).rates(ircopa);
    rr(1).rates(ircopa) = ratraw * sc1a;
    rr(2).rates(ircopa) = rr(2).rates(ircopa)*sc1a + ratraw*sc1adt;


    jscr++;
    screen5(state,jscr,
            27.0_rt, 55.0_rt, 1.0_rt, 1.0_rt,
            sc1a,sc1adt,sc1add);


    // co55(p,g)ni56
    ratraw = rr(1).rates(ircopg);
    rr(1).rates(ircopg) = ratraw * sc1a;
    rr(2).rates(ircopg) = rr(2).rates(ircopg)*sc1a + ratraw*sc1adt;

    ratraw = rr(1).rates(irnigp);
    rr(1).rates(irnigp) = ratraw * sc1a;
    rr(2).rates(irnigp) = rr(2).rates(irnigp)*sc1a + ratraw*sc1adt;


    // now form those lovely dummy proton link rates

    // mg24(a,p)27al(p,g)28si
    Real rate_irr1    = 0.0e0_rt;
    Real dratedt_irr1 = 0.0e0_rt;
    denom    = rr(1).rates(iralpa) + rr(1).rates(iralpg);
    denomdt  = rr(2).rates(iralpa) + rr(2).rates(iralpg);

    if (denom > 1.0e-30_rt) {
       zz = 1.0e0_rt/denom;
       rate_irr1    = rr(1).rates(iralpa)*zz;
       dratedt_irr1 = (rr(2).rates(iralpa) - rate_irr1*denomdt)*zz;
    }

    // add to the effective Mg24 <-> Si28 rates

    rr(1).rates(Mg24_He4_to_Si28_forward) += rr(1).rates(irmgap) * (1.0_rt - rate_irr1);
    rr(2).rates(Mg24_He4_to_Si28_forward) += rr(2).rates(irmgap) * (1.0_rt - rate_irr1) - rr(1).rates(irmgap) * dratedt_irr1;

    rr(1).rates(Mg24_He4_to_Si28_reverse) += rr(1).rates(irsigp) * rate_irr1;
    rr(2).rates(Mg24_He4_to_Si28_reverse) += rr(2).rates(irsigp) * rate_irr1 + rr(1).rates(irsigp) * dratedt_irr1;

    // si28(a,p)p31(p,g)s32
    Real rate_irs1  = 0.0e0_rt;
    Real dratedt_irs1  = 0.0e0_rt;
    denom    = rr(1).rates(irppa) + rr(1).rates(irppg);
    denomdt  = rr(2).rates(irppa) + rr(2).rates(irppg);
    if (denom > 1.0e-30_rt) {
       zz = 1.0e0_rt/denom;
       rate_irs1 = rr(1).rates(irppa)*zz;
       dratedt_irs1 = (rr(2).rates(irppa) - rate_irs1*denomdt)*zz;
    }

    // add to the effective Si28 <-> S32 rates

    rr(1).rates(Si28_He4_to_S32_forward) += rr(1).rates(irsiap) * (1.0_rt - rate_irs1);
    rr(2).rates(Si28_He4_to_S32_forward) += rr(2).rates(irsiap) * (1.0_rt - rate_irs1) - rr(1).rates(irsiap) * dratedt_irs1;

    rr(1).rates(Si28_He4_to_S32_reverse) += rr(1).rates(irsgp) * rate_irs1;
    rr(2).rates(Si28_He4_to_S32_reverse) += rr(2).rates(irsgp) * rate_irs1 + rr(1).rates(irsgp) * dratedt_irs1;

    // Branching ratios for O16 + O16:
    // 0.56: O16 + O16 -> Si28 + He4
    // 0.34: O16 + O16 -> P31 + proton
    // 0.10: O16 + O16 -> S32

    rr(1).rates(O16_O16_to_Si28_He4_forward) = 0.56_rt * O16_O16_rate + 0.34_rt * O16_O16_rate * rate_irs1;
    rr(2).rates(O16_O16_to_Si28_He4_forward) = 0.56_rt * O16_O16_dratedt + 0.34_rt * (O16_O16_dratedt * rate_irs1 + O16_O16_rate * dratedt_irs1);

    rr(1).rates(O16_O16_to_S32_forward) = 0.10_rt * O16_O16_rate + 0.34_rt * O16_O16_rate * (1.0_rt - rate_irs1);
    rr(2).rates(O16_O16_to_S32_forward) = 0.10_rt * O16_O16_dratedt + 0.34_rt * (O16_O16_dratedt * (1.0_rt - rate_irs1) - O16_O16_rate * dratedt_irs1);

    // s32(a,p)cl35(p,g)ar36
    Real rate_irt1  = 0.0e0_rt;
    Real dratedt_irt1  = 0.0e0_rt;
    denom    = rr(1).rates(irclpa) + rr(1).rates(irclpg);
    denomdt  = rr(2).rates(irclpa) + rr(2).rates(irclpg);
    if (denom > 1.0e-30_rt) {
       zz = 1.0e0_rt/denom;
       rate_irt1 = rr(1).rates(irclpa)*zz;
       dratedt_irt1 = (rr(2).rates(irclpa) - rate_irt1*denomdt)*zz;
    }

    // add to the effective S32 <-> Ar36 rates

    rr(1).rates(S32_He4_to_Ar36_forward) += rr(1).rates(irsap) * (1.0_rt - rate_irt1);
    rr(2).rates(S32_He4_to_Ar36_forward) += rr(2).rates(irsap) * (1.0_rt - rate_irt1) - rr(1).rates(irsap) * dratedt_irt1;

    rr(1).rates(S32_He4_to_Ar36_reverse) += rr(1).rates(irargp) * rate_irt1;
    rr(2).rates(S32_He4_to_Ar36_reverse) += rr(2).rates(irargp) * rate_irt1 + rr(1).rates(irargp) * dratedt_irt1;

    // ar36(a,p)k39(p,g)ca40
    Real rate_iru1  = 0.0e0_rt;
    Real dratedt_iru1  = 0.0e0_rt;
    denom    = rr(1).rates(irkpa) + rr(1).rates(irkpg);
    denomdt  = rr(2).rates(irkpa) + rr(2).rates(irkpg);
    if (denom > 1.0e-30_rt) {
       zz   = 1.0e0_rt/denom;
       rate_iru1   = rr(1).rates(irkpa)*zz;
       dratedt_iru1 = (rr(2).rates(irkpa) - rate_iru1*denomdt)*zz;
    }

    // add to the effective Ar36 <-> Ca40 rates

    rr(1).rates(Ar36_He4_to_Ca40_forward) += rr(1).rates(irarap) * (1.0_rt - rate_iru1);
    rr(2).rates(Ar36_He4_to_Ca40_forward) += rr(2).rates(irarap) * (1.0_rt - rate_iru1) - rr(1).rates(irarap) * dratedt_iru1;

    rr(1).rates(Ar36_He4_to_Ca40_reverse) += rr(1).rates(ircagp) * rate_iru1;
    rr(2).rates(Ar36_He4_to_Ca40_reverse) += rr(2).rates(ircagp) * rate_iru1 + rr(1).rates(ircagp) * dratedt_iru1;

    // ca40(a,p)sc43(p,g)ti44
    Real rate_irv1  = 0.0e0_rt;
    Real dratedt_irv1  = 0.0e0_rt;
    denom    = rr(1).rates(irscpa) + rr(1).rates(irscpg);
    denomdt  = rr(2).rates(irscpa) + rr(2).rates(irscpg);
    if (denom > 1.0e-30_rt) {
       zz  = 1.0e0_rt/denom;
       rate_irv1 = rr(1).rates(irscpa)*zz;
       dratedt_irv1 = (rr(2).rates(irscpa) - rate_irv1*denomdt)*zz;
    }

    // add to the effective Ca40 <-> Ti44 rates

    rr(1).rates(Ca40_He4_to_Ti44_forward) += rr(1).rates(ircaap) * (1.0_rt - rate_irv1);
    rr(2).rates(Ca40_He4_to_Ti44_forward) += rr(2).rates(ircaap) * (1.0_rt - rate_irv1) - rr(1).rates(ircaap) * dratedt_irv1;

    rr(1).rates(Ca40_He4_to_Ti44_reverse) += rr(1).rates(irtigp) * rate_irv1;
    rr(2).rates(Ca40_He4_to_Ti44_reverse) += rr(2).rates(irtigp) * rate_irv1 + rr(1).rates(irtigp) * dratedt_irv1;

    // ti44(a,p)v47(p,g)cr48
    Real rate_irw1 = 0.0e0_rt;
    Real dratedt_irw1 = 0.0e0_rt;
    denom    = rr(1).rates(irvpa) + rr(1).rates(irvpg);
    denomdt  = rr(2).rates(irvpa) + rr(2).rates(irvpg);
    if (denom > 1.0e-30_rt) {
       zz = 1.0e0_rt/denom;
       rate_irw1 = rr(1).rates(irvpa)*zz;
       dratedt_irw1 = (rr(2).rates(irvpa) - rate_irw1*denomdt)*zz;
    }

    // add to the effective Ti44 <-> Cr48 rates

    rr(1).rates(Ti44_He4_to_Cr48_forward) += rr(1).rates(irtiap) * (1.0_rt - rate_irw1);
    rr(2).rates(Ti44_He4_to_Cr48_forward) += rr(2).rates(irtiap) * (1.0_rt - rate_irw1) - rr(1).rates(irtiap) * dratedt_irw1;

    rr(1).rates(Ti44_He4_to_Cr48_reverse) += rr(1).rates(ircrgp) * rate_irw1;
    rr(2).rates(Ti44_He4_to_Cr48_reverse) += rr(2).rates(ircrgp) * rate_irw1 + rr(1).rates(ircrgp) * dratedt_irw1;

    // cr48(a,p)mn51(p,g)fe52
    Real rate_irx1 = 0.0e0_rt;
    Real dratedt_irx1 = 0.0e0_rt;
    denom    = rr(1).rates(irmnpa) + rr(1).rates(irmnpg);
    denomdt  = rr(2).rates(irmnpa) + rr(2).rates(irmnpg);
    if (denom > 1.0e-30_rt) {
       zz = 1.0e0_rt/denom;
       rate_irx1 = rr(1).rates(irmnpa)*zz;
       dratedt_irx1 = (rr(2).rates(irmnpa) - rate_irx1*denomdt)*zz;
    }

    // add to the effective Cr48 <-> Fe52 rates

    rr(1).rates(Cr48_He4_to_Fe52_forward) += rr(1).rates(ircrap) * (1.0_rt - rate_irx1);
    rr(2).rates(Cr48_He4_to_Fe52_forward) += rr(2).rates(ircrap) * (1.0_rt - rate_irx1) - rr(1).rates(ircrap) * dratedt_irx1;

    rr(1).rates(Cr48_He4_to_Fe52_reverse) += rr(1).rates(irfegp) * rate_irx1;
    rr(2).rates(Cr48_He4_to_Fe52_reverse) += rr(2).rates(irfegp) * rate_irx1 + rr(1).rates(irfegp) * dratedt_irx1;

    // fe52(a,p)co55(p,g)ni56
    Real rate_iry1 = 0.0e0_rt;
    Real dratedt_iry1 = 0.0e0_rt;
    denom    = rr(1).rates(ircopa) + rr(1).rates(ircopg);
    denomdt  = rr(2).rates(ircopa) + rr(2).rates(ircopg);
    if (denom > 1.0e-30_rt) {
       zz = 1.0e0_rt/denom;
       rate_iry1 = rr(1).rates(ircopa)*zz;
       dratedt_iry1 = (rr(2).rates(ircopa) - rate_iry1*denomdt)*zz;
    }

    // add to the effective Fe52 <-> Ni56 rates

    rr(1).rates(Fe52_He4_to_Ni56_forward) += rr(1).rates(irfeap) * (1.0_rt - rate_iry1);
    rr(2).rates(Fe52_He4_to_Ni56_forward) += rr(2).rates(irfeap) * (1.0_rt - rate_iry1) - rr(1).rates(irfeap) * dratedt_iry1;

    rr(1).rates(Fe52_He4_to_Ni56_reverse) += rr(1).rates(irnigp) * rate_iry1;
    rr(2).rates(Fe52_He4_to_Ni56_reverse) += rr(2).rates(irnigp) * rate_iry1 + rr(1).rates(irnigp) * dratedt_iry1;

}


AMREX_GPU_HOST_DEVICE AMREX_INLINE
void evaluate_rates(burn_t const& state, Array1D<rate_t, 1, Rates::NumGroups>& rr)
{
    Real rho, temp;
    Array1D<Real, 1, NumSpec> y;

    // Get the data from the state
    rho  = state.rho;
    temp = state.T;

    for (int i = 1; i <= NumSpec; ++i) {
        y(i) = state.xn[i-1] * aion_inv[i-1];
    }

    // Get the raw reaction rates
    if (use_tables) {
        aprox13tab(temp, rho, rr);
    } else {
        aprox13rat(temp, rho, rr);
    }

    // Do the screening here because the corrections depend on the composition
    screen_aprox13(temp, rho, y, rr);
}


template<class MatrixType>
AMREX_GPU_HOST_DEVICE AMREX_INLINE
void dfdy_isotopes_aprox13(Array1D<Real, 1, NumSpec> const& y,
                           burn_t const& state, Array1D<rate_t, 1, Rates::NumGroups> const& rr,
                           MatrixType& jac)
{

    // this routine sets up the dense aprox13 jacobian for the isotopes

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

// Evaluates the right hand side of the aprox13 ODEs
AMREX_GPU_HOST_DEVICE AMREX_INLINE
void rhs(Array1D<Real, 1, NumSpec> const& y, const rate_t& rr,
         Array1D<Real, 1, neqs>& dydt)
{
    using namespace Species;
    using namespace Rates;

    for (int i = 1; i <= NumSpec; ++i) {
        dydt(i) = 0.0_rt;
    }

    Array1D<Real, 1, 17> a;

    // he4 reactions
    // heavy ion reactions
    a(1)  = y(C12) * y(C12) * rr.rates(C12_C12_to_Ne20_He4_forward);
    a(2)  = y(C12) * y(O16) * rr.rates(C12_O16_to_Mg24_He4_forward);
    a(3)  = y(O16) * y(O16) * rr.rates(O16_O16_to_Si28_He4_forward);

    dydt(He4) = dydt(He4) + esum3(a);

    // (a,g) and (g,a) reactions
    a(1)  = -3.0e0_rt * y(He4) * y(He4) * y(He4) * rr.rates(He4_He4_He4_to_C12_forward);
    a(2)  =  3.0e0_rt * y(C12) * rr.rates(He4_He4_He4_to_C12_reverse);
    a(3)  = -y(He4)  * y(C12) * rr.rates(C12_He4_to_O16_forward);
    a(4)  =  y(O16)  * rr.rates(C12_He4_to_O16_reverse);
    a(5)  = -y(He4)  * y(O16) * rr.rates(O16_He4_to_Ne20_forward);
    a(6)  =  y(Ne20) * rr.rates(O16_He4_to_Ne20_reverse);
    a(7)  = -y(He4)  * y(Ne20) * rr.rates(Ne20_He4_to_Mg24_forward);
    a(8)  =  y(Mg24) * rr.rates(Ne20_He4_to_Mg24_reverse);
    a(9)  = -y(He4)  * y(Mg24) * rr.rates(Mg24_He4_to_Si28_forward);
    a(10) =  y(Si28) * rr.rates(Mg24_He4_to_Si28_reverse);
    a(11) = -y(He4)  * y(Si28) * rr.rates(Si28_He4_to_S32_forward);
    a(12) =  y(S32)  * rr.rates(Si28_He4_to_S32_reverse);

    dydt(He4) = dydt(He4) + esum12(a);

    a(1)  = -y(He4)  * y(S32) * rr.rates(S32_He4_to_Ar36_forward);
    a(2)  =  y(Ar36) * rr.rates(S32_He4_to_Ar36_reverse);
    a(3)  = -y(He4)  * y(Ar36) * rr.rates(Ar36_He4_to_Ca40_forward);
    a(4)  =  y(Ca40) * rr.rates(Ar36_He4_to_Ca40_reverse);
    a(5)  = -y(He4)  * y(Ca40) * rr.rates(Ca40_He4_to_Ti44_forward);
    a(6)  =  y(Ti44) * rr.rates(Ca40_He4_to_Ti44_reverse);
    a(7)  = -y(He4)  * y(Ti44) * rr.rates(Ti44_He4_to_Cr48_forward);
    a(8)  =  y(Cr48) * rr.rates(Ti44_He4_to_Cr48_reverse);
    a(9)  = -y(He4)  * y(Cr48) * rr.rates(Cr48_He4_to_Fe52_forward);
    a(10) =  y(Fe52) * rr.rates(Cr48_He4_to_Fe52_reverse);
    a(11) = -y(He4)  * y(Fe52) * rr.rates(Fe52_He4_to_Ni56_forward);
    a(12) =  y(Ni56) * rr.rates(Fe52_He4_to_Ni56_reverse);

    dydt(He4) = dydt(He4) + esum12(a);

    // c12 reactions
    a(1) = -2.0e0_rt * y(C12) * y(C12) * rr.rates(C12_C12_to_Ne20_He4_forward);
    a(2) = -y(C12) * y(O16) * rr.rates(C12_O16_to_Mg24_He4_forward);
    a(3) = -y(C12) * y(O16) * rr.rates(C12_O16_to_Si28_forward);
    a(4) =  y(He4) * y(He4) * y(He4) * rr.rates(He4_He4_He4_to_C12_forward);
    a(5) = -y(C12) * rr.rates(He4_He4_He4_to_C12_reverse);
    a(6) = -y(C12) * y(He4) * rr.rates(C12_He4_to_O16_forward);
    a(7) =  y(O16) * rr.rates(C12_He4_to_O16_reverse);

    dydt(C12) = dydt(C12) + esum7(a);


    // o16 reactions
    a(1) = -y(C12) * y(O16) * rr.rates(C12_O16_to_Mg24_He4_forward);
    a(2) = -y(C12) * y(O16) * rr.rates(C12_O16_to_Si28_forward);
    a(3) = -2.0_rt * y(O16) * y(O16) * rr.rates(O16_O16_to_Si28_He4_forward);
    a(4) = -2.0_rt * y(O16) * y(O16) * rr.rates(O16_O16_to_S32_forward);
    a(5) =  y(C12) * y(He4) * rr.rates(C12_He4_to_O16_forward);
    a(6) = -y(O16) * y(He4) * rr.rates(O16_He4_to_Ne20_forward);
    a(7) = -y(O16) * rr.rates(C12_He4_to_O16_reverse);
    a(8) =  y(Ne20) * rr.rates(O16_He4_to_Ne20_reverse);

    dydt(O16) = dydt(O16) + esum8(a);


    // ne20 reactions
    a(1) =  y(C12) * y(C12) * rr.rates(C12_C12_to_Ne20_He4_forward);
    a(2) =  y(O16) * y(He4) * rr.rates(O16_He4_to_Ne20_forward);
    a(3) = -y(Ne20) * y(He4) * rr.rates(Ne20_He4_to_Mg24_forward);
    a(4) = -y(Ne20) * rr.rates(O16_He4_to_Ne20_reverse);
    a(5) =  y(Mg24) * rr.rates(Ne20_He4_to_Mg24_reverse);

    dydt(Ne20) = dydt(Ne20) + esum5(a);


    // mg24 reactions
    a(1) =  y(C12) * y(O16) * rr.rates(C12_O16_to_Mg24_He4_forward);
    a(2) =  y(Ne20) * y(He4) * rr.rates(Ne20_He4_to_Mg24_forward);
    a(3) = -y(Mg24) * y(He4) * rr.rates(Mg24_He4_to_Si28_forward);
    a(4) = -y(Mg24) * rr.rates(Ne20_He4_to_Mg24_reverse);
    a(5) =  y(Si28) * rr.rates(Mg24_He4_to_Si28_reverse);

    dydt(Mg24) = dydt(Mg24) + esum5(a);


    // si28 reactions
    a(1) =  y(C12) * y(O16) * rr.rates(C12_O16_to_Si28_forward);
    a(2) =  y(O16) * y(O16) * rr.rates(O16_O16_to_Si28_He4_forward);
    a(3) =  y(Mg24) * y(He4) * rr.rates(Mg24_He4_to_Si28_forward);
    a(4) = -y(Si28) * y(He4) * rr.rates(Si28_He4_to_S32_forward);
    a(5) = -y(Si28) * rr.rates(Mg24_He4_to_Si28_reverse);
    a(6) =  y(S32)  * rr.rates(Si28_He4_to_S32_reverse);

    dydt(Si28) = dydt(Si28) + esum6(a);


    // s32 reactions
    a(1) =  y(O16) * y(O16) * rr.rates(O16_O16_to_S32_forward);
    a(2) =  y(Si28) * y(He4) * rr.rates(Si28_He4_to_S32_forward);
    a(3) = -y(S32) * y(He4) * rr.rates(S32_He4_to_Ar36_forward);
    a(4) = -y(S32) * rr.rates(Si28_He4_to_S32_reverse);
    a(5) =  y(Ar36) * rr.rates(S32_He4_to_Ar36_reverse);

    dydt(S32) = dydt(S32) + esum5(a);


    // ar36 reactions
    a(1) =  y(S32)  * y(He4) * rr.rates(S32_He4_to_Ar36_forward);
    a(2) = -y(Ar36) * y(He4) * rr.rates(Ar36_He4_to_Ca40_forward);
    a(3) = -y(Ar36) * rr.rates(S32_He4_to_Ar36_reverse);
    a(4) =  y(Ca40) * rr.rates(Ar36_He4_to_Ca40_reverse);

    dydt(Ar36) = dydt(Ar36) + esum4(a);


    // ca40 reactions
    a(1) =  y(Ar36) * y(He4) * rr.rates(Ar36_He4_to_Ca40_forward);
    a(2) = -y(Ca40) * y(He4) * rr.rates(Ca40_He4_to_Ti44_forward);
    a(3) = -y(Ca40) * rr.rates(Ar36_He4_to_Ca40_reverse);
    a(4) =  y(Ti44) * rr.rates(Ca40_He4_to_Ti44_reverse);

    dydt(Ca40) = dydt(Ca40) + esum4(a);


    // ti44 reactions
    a(1) =  y(Ca40) * y(He4) * rr.rates(Ca40_He4_to_Ti44_forward);
    a(2) = -y(Ti44) * y(He4) * rr.rates(Ti44_He4_to_Cr48_forward);
    a(3) = -y(Ti44) * rr.rates(Ca40_He4_to_Ti44_reverse);
    a(4) =  y(Cr48) * rr.rates(Ti44_He4_to_Cr48_reverse);

    dydt(Ti44) = dydt(Ti44) + esum4(a);


    // cr48 reactions
    a(1) =  y(Ti44) * y(He4) * rr.rates(Ti44_He4_to_Cr48_forward);
    a(2) = -y(Cr48) * y(He4) * rr.rates(Cr48_He4_to_Fe52_forward);
    a(3) = -y(Cr48) * rr.rates(Ti44_He4_to_Cr48_reverse);
    a(4) =  y(Fe52) * rr.rates(Cr48_He4_to_Fe52_reverse);

    dydt(Cr48) = dydt(Cr48) + esum4(a);


    // fe52 reactions
    a(1) =  y(Cr48) * y(He4) * rr.rates(Cr48_He4_to_Fe52_forward);
    a(2) = -y(Fe52) * y(He4) * rr.rates(Fe52_He4_to_Ni56_forward);
    a(3) = -y(Fe52) * rr.rates(Cr48_He4_to_Fe52_reverse);
    a(4) =  y(Ni56) * rr.rates(Fe52_He4_to_Ni56_reverse);

    dydt(Fe52) = dydt(Fe52) + esum4(a);


    // ni56 reactions
    a(1) =  y(Fe52) * y(He4) * rr.rates(Fe52_He4_to_Ni56_forward);
    a(2) = -y(Ni56) * rr.rates(Fe52_He4_to_Ni56_reverse);

    dydt(Ni56) = dydt(Ni56) + a(1) + a(2);
}


AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
void actual_rhs(burn_t& state, Array1D<Real, 1, neqs>& ydot)
{
    // Evaluate the rates

    Array1D<rate_t, 1, Rates::NumGroups> rr;
    evaluate_rates(state, rr);

    // Call the RHS to get dydt

    Array1D<Real, 1, NumSpec> spec_rhs = RHS::species_rhs(state, rr(1));

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
    Array1D<Real, 1, neqs> yderivs;

    // Initialize jac to 0

    jac.zero();

    // Evaluate the rates

    evaluate_rates(state, rr);

    // Get the data from the state

    rho  = state.rho;
    temp = state.T;
    abar = state.abar;
    zbar = state.zbar;

    for (int i = 1; i <= NumSpec; ++i) {
        y(i) = state.xn[i-1] * aion_inv[i-1];
    }

    // Species Jacobian elements with respect to other species

    dfdy_isotopes_aprox13(y, state, rr, jac);

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

    // Evaluate the Jacobian elements with respect to temperature by
    // calling the RHS using d(rate) / dT

    rhs(y, rr(2), yderivs);

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

    add_screening_factor(jscr++, zion[He4-1], aion[He4-1], 4.0e0_rt, 8.0e0_rt);

    add_screening_factor(jscr++, zion[C12-1], aion[C12-1], zion[He4-1], aion[He4-1]);

    add_screening_factor(jscr++, zion[C12-1], aion[C12-1], zion[C12-1], aion[C12-1]);

    add_screening_factor(jscr++, zion[C12-1], aion[C12-1], zion[O16-1], aion[O16-1]);

    add_screening_factor(jscr++, zion[O16-1], aion[O16-1], zion[O16-1], aion[O16-1]);

    add_screening_factor(jscr++, zion[O16-1], aion[O16-1], zion[He4-1], aion[He4-1]);

    add_screening_factor(jscr++, zion[Ne20-1], aion[Ne20-1], zion[He4-1], aion[He4-1]);

    add_screening_factor(jscr++, zion[Mg24-1], aion[Mg24-1], zion[He4-1], aion[He4-1]);

    add_screening_factor(jscr++, 13.0e0_rt, 27.0e0_rt, 1.0e0_rt, 1.0e0_rt);

    add_screening_factor(jscr++, zion[Si28-1], aion[Si28-1], zion[He4-1], aion[He4-1]);

    add_screening_factor(jscr++, 15.0e0_rt, 31.0e0_rt, 1.0e0_rt, 1.0e0_rt);

    add_screening_factor(jscr++, zion[S32-1], aion[S32-1], zion[He4-1], aion[He4-1]);

    add_screening_factor(jscr++, 17.0e0_rt, 35.0e0_rt, 1.0e0_rt, 1.0e0_rt);

    add_screening_factor(jscr++, zion[Ar36-1], aion[Ar36-1], zion[He4-1], aion[He4-1]);

    add_screening_factor(jscr++, 19.0e0_rt, 39.0e0_rt, 1.0e0_rt, 1.0e0_rt);

    add_screening_factor(jscr++, zion[Ca40-1], aion[Ca40-1], zion[He4-1], aion[He4-1]);

    add_screening_factor(jscr++, 21.0e0_rt, 43.0e0_rt, 1.0e0_rt, 1.0e0_rt);

    add_screening_factor(jscr++, zion[Ti44-1], aion[Ti44-1], zion[He4-1], aion[He4-1]);

    add_screening_factor(jscr++, 23.0e0_rt, 47.0e0_rt, 1.0e0_rt, 1.0e0_rt);

    add_screening_factor(jscr++, zion[Cr48-1], aion[Cr48-1], zion[He4-1], aion[He4-1]);

    add_screening_factor(jscr++, 25.0e0_rt, 51.0e0_rt, 1.0e0_rt, 1.0e0_rt);

    add_screening_factor(jscr++, zion[Fe52-1], aion[Fe52-1], zion[He4-1], aion[He4-1]);

    add_screening_factor(jscr++, 27.0e0_rt, 55.0e0_rt, 1.0e0_rt, 1.0e0_rt);

}

#endif
