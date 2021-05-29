#ifndef actual_rhs_H
#define actual_rhs_H

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
#include <jacobian_utilities.H>

using namespace amrex;
using namespace ArrayUtil;
using namespace Rates;

constexpr Real c54 = 56.0e0_rt / 54.0e0_rt;

void actual_rhs_init();

AMREX_GPU_HOST_DEVICE AMREX_INLINE
void aprox21rat (Real btemp, Real bden,
                 Array1D<Real, 1, NumRates>& ratraw,
                 Array1D<Real, 1, NumRates>& dratrawdt,
                 Array1D<Real, 1, NumRates>& dratrawdd)
{
    // this routine generates unscreened
    // nuclear reaction rates for the aprox21 network.

    Real rrate, drratedt;
    Real ff1, dff1dt, ff2, dff2dt, tot, dtotdt, invtot;

    for (int i = 1; i <= NumRates; ++i) {
        ratraw(i) = 0.0_rt;
        dratrawdt(i) = 0.0_rt;
        dratrawdd(i) = 0.0_rt;
    }

    if (btemp < 1.0e6_rt) return;

    // get the temperature factors
    tf_t tf = get_tfactors(btemp);

    // p(p,e+nu)d
    rate_pp(tf, bden, ratraw(irpp), dratrawdt(irpp), rrate, drratedt);

    // p(n,g)2d
    rate_png(tf, bden, ratraw(irhng), dratrawdt(irhng), ratraw(irdgn), dratrawdt(irdgn));

    // d(p,g)he3
    rate_dpg(tf, bden, ratraw(irdpg), dratrawdt(irdpg), ratraw(irhegp), dratrawdt(irhegp));

    // he3(n,g)he4
    rate_he3ng(tf, bden, ratraw(irheng), dratrawdt(irheng), ratraw(irhegn), dratrawdt(irhegn));

    // he3(he3,2p)he4
    rate_he3he3(tf, bden, ratraw(ir33), dratrawdt(ir33), rrate, drratedt);

    // he3(he4,g)be7
    rate_he3he4(tf, bden, ratraw(irhe3ag), dratrawdt(irhe3ag), rrate, drratedt);

    // triple alpha to c12
    rate_triplealf(tf, bden, ratraw(ir3a), dratrawdt(ir3a), ratraw(irg3a), dratrawdt(irg3a));

    // c12(p,g)13n
    rate_c12pg(tf, bden, ratraw(ircpg), dratrawdt(ircpg), rrate, drratedt);

    // n14(p,g)o15
    rate_n14pg(tf, bden, ratraw(irnpg), dratrawdt(irnpg), rrate, drratedt);

    // fraction fg of n15 goes (pg) to o16, fraction fa of n15 goes (pa) to c12
    // result is  n14+2p  goes to o16    at a rate = rnpg*fg
    //                    goes to c12+a  at a rate = rnpg*fa

    rate_n15pg(tf, bden, ff1, dff1dt, rrate, drratedt);

    rate_n15pa(tf, bden, ff2, dff2dt, rrate, drratedt);

    tot            = ff1 + ff2;
    dtotdt         = dff1dt + dff2dt;
    //dtotdd         = dff1dd + dff2dd;
    invtot         = 1.0e0_rt/tot;

    ratraw(ifa)    = ff2 * invtot;
    dratrawdt(ifa) = dff2dt * invtot - ff2 * invtot*invtot * dtotdt;
    //dratrawdd(ifa) = dff2dd * invtot - ff2 * invtot*invtot * dtotdd;

    ratraw(ifg)    = 1.0e0_rt - ratraw(ifa);
    dratrawdt(ifg) = -dratrawdt(ifa);
    //dratrawdd(ifg) = -dratrawdd(ifa);

    // o16(p,g)f17
    rate_o16pg(tf, bden, ratraw(iropg), dratrawdt(iropg), rrate, drratedt);

    // n14(a,g)f18
    rate_n14ag(tf, bden, ratraw(irnag), dratrawdt(irnag), rrate, drratedt);

    // Determine which c12(a,g)o16 rate to use
    if (use_c12ag_deboer17) {
        // deboer + 2017 c12(a,g)o16 rate
       rate_c12ag_deboer17(tf, bden, ratraw(ircag), dratrawdt(ircag),
                           ratraw(iroga), dratrawdt(iroga));
    } else {
        // 1.7 times cf88 c12(a,g)o16 rate
       rate_c12ag(tf, bden, ratraw(ircag), dratrawdt(ircag),
                  ratraw(iroga), dratrawdt(iroga));
    }

    // c12 + c12
    rate_c12c12(tf, bden, ratraw(ir1212), dratrawdt(ir1212), rrate, drratedt);

    // c12 + o16
    rate_c12o16(tf, bden, ratraw(ir1216), dratrawdt(ir1216), rrate, drratedt);

    // o16 + o16
    rate_o16o16(tf, bden, ratraw(ir1616), dratrawdt(ir1616), rrate, drratedt);

    // o16(a,g)ne20
    rate_o16ag(tf,bden, ratraw(iroag), dratrawdt(iroag), ratraw(irnega), dratrawdt(irnega));

    // ne20(a,g)mg24
    rate_ne20ag(tf, bden, ratraw(irneag), dratrawdt(irneag), ratraw(irmgga), dratrawdt(irmgga));

    // mg24(a,g)si28
    rate_mg24ag(tf, bden, ratraw(irmgag), dratrawdt(irmgag), ratraw(irsiga), dratrawdt(irsiga));

    // mg24(a,p)al27
    rate_mg24ap(tf, bden, ratraw(irmgap), dratrawdt(irmgap), ratraw(iralpa), dratrawdt(iralpa));

    // al27(p,g)si28
    rate_al27pg(tf, bden, ratraw(iralpg), dratrawdt(iralpg), ratraw(irsigp), dratrawdt(irsigp));

    // si28(a,g)s32
    rate_si28ag(tf, bden, ratraw(irsiag), dratrawdt(irsiag), ratraw(irsga), dratrawdt(irsga));

    // si28(a,p)p31
    rate_si28ap(tf, bden, ratraw(irsiap), dratrawdt(irsiap), ratraw(irppa), dratrawdt(irppa));

    // p31(p,g)s32
    rate_p31pg(tf, bden, ratraw(irppg), dratrawdt(irppg), ratraw(irsgp), dratrawdt(irsgp));

    // s32(a,g)ar36
    rate_s32ag(tf, bden, ratraw(irsag), dratrawdt(irsag), ratraw(irarga), dratrawdt(irarga));

    // s32(a,p)cl35
    rate_s32ap(tf, bden, ratraw(irsap), dratrawdt(irsap), ratraw(irclpa), dratrawdt(irclpa));

    // cl35(p,g)ar36
    rate_cl35pg(tf, bden, ratraw(irclpg), dratrawdt(irclpg), ratraw(irargp), dratrawdt(irargp));

    // ar36(a,g)ca40
    rate_ar36ag(tf, bden, ratraw(irarag), dratrawdt(irarag), ratraw(ircaga), dratrawdt(ircaga));

    // ar36(a,p)k39
    rate_ar36ap(tf, bden, ratraw(irarap), dratrawdt(irarap), ratraw(irkpa), dratrawdt(irkpa));

    // k39(p,g)ca40
    rate_k39pg(tf, bden, ratraw(irkpg), dratrawdt(irkpg), ratraw(ircagp), dratrawdt(ircagp));

    // ca40(a,g)ti44
    rate_ca40ag(tf, bden, ratraw(ircaag), dratrawdt(ircaag), ratraw(irtiga), dratrawdt(irtiga));

    // ca40(a,p)sc43
    rate_ca40ap(tf, bden, ratraw(ircaap), dratrawdt(ircaap), ratraw(irscpa), dratrawdt(irscpa));

    // sc43(p,g)ti44
    rate_sc43pg(tf, bden, ratraw(irscpg), dratrawdt(irscpg), ratraw(irtigp), dratrawdt(irtigp));

    // ti44(a,g)cr48
    rate_ti44ag(tf, bden, ratraw(irtiag), dratrawdt(irtiag), ratraw(ircrga), dratrawdt(ircrga));

    // ti44(a,p)v47
    rate_ti44ap(tf, bden, ratraw(irtiap), dratrawdt(irtiap), ratraw(irvpa), dratrawdt(irvpa));

    // v47(p,g)cr48
    rate_v47pg(tf, bden, ratraw(irvpg), dratrawdt(irvpg), ratraw(ircrgp), dratrawdt(ircrgp));

    // cr48(a,g)fe52
    rate_cr48ag(tf, bden, ratraw(ircrag), dratrawdt(ircrag), ratraw(irfega), dratrawdt(irfega));

    // cr48(a,p)mn51
    rate_cr48ap(tf, bden, ratraw(ircrap), dratrawdt(ircrap), ratraw(irmnpa), dratrawdt(irmnpa));

    // mn51(p,g)fe52
    rate_mn51pg(tf, bden, ratraw(irmnpg), dratrawdt(irmnpg), ratraw(irfegp), dratrawdt(irfegp));

    // fe52(a,g)ni56
    rate_fe52ag(tf, bden, ratraw(irfeag), dratrawdt(irfeag), ratraw(irniga), dratrawdt(irniga));

    // fe52(a,p)co55
    rate_fe52ap(tf, bden, ratraw(irfeap), dratrawdt(irfeap), ratraw(ircopa), dratrawdt(ircopa));

    // co55(p,g)ni56
    rate_co55pg(tf, bden, ratraw(ircopg), dratrawdt(ircopg), ratraw(irnigp), dratrawdt(irnigp));

    // fe52(n,g)fe53
    rate_fe52ng(tf, bden, ratraw(ir52ng), dratrawdt(ir52ng), ratraw(ir53gn), dratrawdt(ir53gn));

    // fe53(n,g)fe54
    rate_fe53ng(tf, bden, ratraw(ir53ng), dratrawdt(ir53ng), ratraw(ir54gn), dratrawdt(ir54gn));

    // fe54(n,g)fe55
    rate_fe54ng(tf, bden, ratraw(ir54ng), dratrawdt(ir54ng), ratraw(ir55gn), dratrawdt(ir55gn));

    // fe54(p,g)co55
    rate_fe54pg(tf, bden, ratraw(irfepg), dratrawdt(irfepg), ratraw(ircogp), dratrawdt(ircogp));

    // fe54(a,p)co57
    rate_fe54ap(tf, bden, ratraw(irfe54ap), dratrawdt(irfe54ap), ratraw(irco57pa), dratrawdt(irco57pa));

    // fe55(n,g)fe56
    rate_fe55ng(tf, bden, ratraw(ir55ng), dratrawdt(ir55ng), ratraw(ir56gn), dratrawdt(ir56gn));

    // fe56(p,g)co57
    rate_fe56pg(tf, bden, ratraw(irfe56pg), dratrawdt(irfe56pg), ratraw(irco57gp), dratrawdt(irco57gp));
}


// electron capture rates on nucleons for aprox21
// note they are composition dependent

AMREX_GPU_HOST_DEVICE AMREX_INLINE
void weak_aprox21 (Array1D<Real, 1, NumSpec> const& y,
                   const burn_t& state,
                   Array1D<Real, 1, NumRates>& ratraw,
                   Array1D<Real, 1, NumRates>& dratrawdt,
                   Array1D<Real, 1, NumRates>& dratrawdd)
{
    using namespace Species;

    Real xx, spen, snep;

    // initialize
    ratraw(irpen)       = 0.0e0_rt;
    dratrawdt(irpen)    = 0.0e0_rt;
    //dratrawdd(irpen)   = 0.0e0_rt;
    ratraw(irnep)       = 0.0e0_rt;
    dratrawdt(irnep)    = 0.0e0_rt;
    //dratrawdd(irnep)   = 0.0e0_rt;
    ratraw(irn56ec)     = 0.0e0_rt;
    dratrawdt(irn56ec)  = 0.0e0_rt;
    //dratrawdd(irn56ec) = 0.0e0_rt;

    if (state.T < 1.0e6_rt || state.rho < 1.0e-9_rt) return;

    // get the p <-> n electron capture rates
    ecapnuc(state.eta, state.T, ratraw(irpen), ratraw(irnep), spen, snep);

    // ni56 electron capture rate
    langanke(state.T, state.rho, y(Ni56), state.y_e, ratraw(irn56ec), xx);
}


AMREX_GPU_HOST_DEVICE AMREX_INLINE
void screen_aprox21 (Real btemp, Real bden,
                     Array1D<Real, 1, NumSpec> const& y,
                     Array1D<Real, 1, NumRates> const& ratraw,
                     Array1D<Real, 1, NumRates> const& dratrawdt,
                     Array1D<Real, 1, NumRates> const& dratrawdd,
                     Array1D<Real, 1, NumRates>& ratdum,
                     Array1D<Real, 1, NumRates>& dratdumdt,
                     Array1D<Real, 1, NumRates>& dratdumdd,
                     Array1D<Real, 1, NumRates>& dratdumdy1,
                     Array1D<Real, 1, NumRates>& dratdumdy2)
{
    using namespace Species;

    // this routine computes the screening factors
    // and applies them to the raw reaction rates,
    // producing the final reaction rates used by the
    // right hand sides and jacobian matrix elements

    Real sc1a, sc1adt, sc1add, sc2a, sc2adt, sc2add;
    Real sc3a, sc3adt, denom, denomdt, xx, zz;
    //Real sc3add, denomdd;

    plasma_state_t state;

    // initialize
    for (int i = 1; i <= NumRates; ++i) {
       ratdum(i)     = ratraw(i);
       dratdumdt(i)  = dratrawdt(i);
       dratdumdd(i)  = dratrawdd(i);
       dratdumdy1(i) = 0.0_rt;
       dratdumdy2(i) = 0.0_rt;
    }

    // set up the state data, which is the same for all screening factors.

    fill_plasma_state(state, btemp, bden, y);

    // first the always fun triple alpha and its inverse
    int jscr = 0;
    screen5(state, jscr,
            zion[He4-1], aion[He4-1], zion[He4-1], aion[He4-1],
            sc1a, sc1adt, sc1add);

    jscr++;
    screen5(state, jscr,
            zion[He4-1], aion[He4-1], 4.0e0_rt, 8.0e0_rt,
            sc2a, sc2adt, sc2add);

    sc3a   = sc1a * sc2a;
    sc3adt = sc1adt*sc2a + sc1a*sc2adt;
    //sc3add = sc1add*sc2a + sc1a*sc2add;

    ratdum(ir3a)    = ratraw(ir3a) * sc3a;
    dratdumdt(ir3a) = dratrawdt(ir3a)*sc3a + ratraw(ir3a)*sc3adt;
    //dratdumdd(ir3a) = dratrawdd(ir3a)*sc3a + ratraw(ir3a)*sc3add;

    ratdum(irg3a)    = ratraw(irg3a) * sc3a;
    dratdumdt(irg3a) = dratrawdt(irg3a)*sc3a + ratraw(irg3a)*sc3adt;
    //dratdumdd(irg3a) = dratrawdd(irg3a)*sc3a + ratraw(irg3a)*sc3add;

    // Identical particle factor (He4 + He4 + He4)

    ratdum(ir3a) *= 1.0_rt / 6.0_rt;
    dratdumdt(ir3a) *= 1.0_rt / 6.0_rt;



    // c12 to o16
    // c12(a,g)o16
    jscr++;
    screen5(state, jscr,
            zion[C12-1], aion[C12-1], zion[He4-1], aion[He4-1],
            sc1a, sc1adt, sc1add);

    ratdum(ircag)     = ratraw(ircag) * sc1a;
    dratdumdt(ircag)  = dratrawdt(ircag)*sc1a + ratraw(ircag)*sc1adt;
    //dratdumdd(ircag)  = dratrawdd(ircag)*sc1a + ratraw(ircag)*sc1add;

    ratdum(iroga)     = ratraw(iroga) * sc1a;
    dratdumdt(iroga)  = dratrawdt(iroga)*sc1a + ratraw(iroga)*sc1adt;
    //dratdumdd(iroga)  = dratrawdd(iroga)*sc1a + ratraw(iroga)*sc1add;



    // c12 + c12
    jscr++;
    screen5(state, jscr,
            zion[C12-1], aion[C12-1], zion[C12-1], aion[C12-1],
            sc1a, sc1adt, sc1add);


    ratdum(ir1212)    = ratraw(ir1212) * sc1a;
    dratdumdt(ir1212) = dratrawdt(ir1212)*sc1a + ratraw(ir1212)*sc1adt;
    //dratdumdd(ir1212) = dratrawdd(ir1212)*sc1a + ratraw(ir1212)*sc1add;

    // Identical particle factor (C12 + C12)

    ratdum(ir1212) *= 0.5_rt;
    dratdumdt(ir1212) *= 0.5_rt;



    // c12 + o16
    jscr++;
    screen5(state, jscr,
            zion[C12-1], aion[C12-1], zion[O16-1], aion[O16-1],
            sc1a, sc1adt, sc1add);

    ratdum(ir1216)    = ratraw(ir1216) * sc1a;
    dratdumdt(ir1216) = dratrawdt(ir1216)*sc1a + ratraw(ir1216)*sc1adt;
    //dratdumdd(ir1216) = dratrawdd(ir1216)*sc1a + ratraw(ir1216)*sc1add;


    // o16 + o16
    jscr++;
    screen5(state, jscr,
            zion[O16-1], aion[O16-1], zion[O16-1], aion[O16-1],
            sc1a, sc1adt, sc1add);


    ratdum(ir1616)    = ratraw(ir1616) * sc1a;
    dratdumdt(ir1616) = dratrawdt(ir1616)*sc1a + ratraw(ir1616)*sc1adt;
    //dratdumdd(ir1616) = dratrawdd(ir1616)*sc1a + ratraw(ir1616)*sc1add;

    // Identical particle factor (O16 + O16)

    ratdum(ir1616) *= 0.5_rt;
    dratdumdt(ir1616) *= 0.5_rt;



    // o16 to ne20
    jscr++;
    screen5(state, jscr,
            zion[O16-1], aion[O16-1], zion[He4-1], aion[He4-1],
            sc1a, sc1adt, sc1add);


    // o16(a,g)ne20
    ratdum(iroag)    = ratraw(iroag) * sc1a;
    dratdumdt(iroag) = dratrawdt(iroag)*sc1a + ratraw(iroag)*sc1adt;
    //dratdumdd(iroag) = dratrawdd(iroag)*sc1a + ratraw(iroag)*sc1add;

    ratdum(irnega)    = ratraw(irnega) * sc1a;
    dratdumdt(irnega) = dratrawdt(irnega)*sc1a + ratraw(irnega)*sc1adt;
    //dratdumdd(irnega) = dratrawdd(irnega)*sc1a + ratraw(irnega)*sc1add;


    // ne20 to mg24
    jscr++;
    screen5(state, jscr,
            zion[Ne20-1], aion[Ne20-1], zion[He4-1], aion[He4-1],
            sc1a, sc1adt, sc1add);


    // ne20(a,g)mg24
    ratdum(irneag)    = ratraw(irneag) * sc1a;
    dratdumdt(irneag) = dratrawdt(irneag)*sc1a + ratraw(irneag)*sc1adt;
    //dratdumdd(irneag) = dratrawdd(irneag)*sc1a + ratraw(irneag)*sc1add;


    ratdum(irmgga)    = ratraw(irmgga) * sc1a;
    dratdumdt(irmgga) = dratrawdt(irmgga)*sc1a + ratraw(irmgga)*sc1adt;
    //dratdumdd(irmgga) = dratrawdd(irmgga)*sc1a + ratraw(irmgga)*sc1add;




    // mg24 to si28
    jscr++;
    screen5(state, jscr,
            zion[Mg24-1], aion[Mg24-1], zion[He4-1], aion[He4-1],
            sc1a, sc1adt, sc1add);


    // mg24(a,g)si28
    ratdum(irmgag)    = ratraw(irmgag) * sc1a;
    dratdumdt(irmgag) = dratrawdt(irmgag)*sc1a + ratraw(irmgag)*sc1adt;
    //dratdumdd(irmgag) = dratrawdd(irmgag)*sc1a + ratraw(irmgag)*sc1add;

    ratdum(irsiga)    = ratraw(irsiga) * sc1a;
    dratdumdt(irsiga) = dratrawdt(irsiga)*sc1a + ratraw(irsiga)*sc1adt;
    //dratdumdd(irsiga) = dratrawdd(irsiga)*sc1a + ratraw(irsiga)*sc1add;



    // mg24(a,p)al27
    ratdum(irmgap)    = ratraw(irmgap) * sc1a;
    dratdumdt(irmgap) = dratrawdt(irmgap)*sc1a + ratraw(irmgap)*sc1adt;
    //dratdumdd(irmgap) = dratrawdd(irmgap)*sc1a + ratraw(irmgap)*sc1add;

    ratdum(iralpa)    = ratraw(iralpa) * sc1a;
    dratdumdt(iralpa) = dratrawdt(iralpa)*sc1a + ratraw(iralpa)*sc1adt;
    //dratdumdd(iralpa) = dratrawdd(iralpa)*sc1a + ratraw(iralpa)*sc1add;


    jscr++;
    screen5(state, jscr,
            13.0e0_rt, 27.0e0_rt, 1.0e0_rt, 1.0e0_rt,
            sc1a, sc1adt, sc1add);

    // al27(p,g)si28
    ratdum(iralpg)    = ratraw(iralpg) * sc1a;
    dratdumdt(iralpg) = dratrawdt(iralpg)*sc1a + ratraw(iralpg)*sc1adt;
    //dratdumdd(iralpg) = dratrawdd(iralpg)*sc1a + ratraw(iralpg)*sc1add;


    ratdum(irsigp)    = ratraw(irsigp) * sc1a;
    dratdumdt(irsigp) = dratrawdt(irsigp)*sc1a + ratraw(irsigp)*sc1adt;
    //dratdumdd(irsigp) = dratrawdd(irsigp)*sc1a + ratraw(irsigp)*sc1add;



    // si28 to s32
    jscr++;
    screen5(state, jscr,
            zion[Si28-1], aion[Si28-1], zion[He4-1], aion[He4-1],
            sc1a, sc1adt, sc1add);


    // si28(a,g)s32
    ratdum(irsiag)    = ratraw(irsiag) * sc1a;
    dratdumdt(irsiag) = dratrawdt(irsiag)*sc1a + ratraw(irsiag)*sc1adt;
    //dratdumdd(irsiag) = dratrawdd(irsiag)*sc1a + ratraw(irsiag)*sc1add;

    ratdum(irsga)    = ratraw(irsga) * sc1a;
    dratdumdt(irsga) = dratrawdt(irsga)*sc1a + ratraw(irsga)*sc1adt;
    //dratdumdd(irsga) = dratrawdd(irsga)*sc1a + ratraw(irsga)*sc1add;



    // si28(a,p)p31
    ratdum(irsiap)    = ratraw(irsiap) * sc1a;
    dratdumdt(irsiap) = dratrawdt(irsiap)*sc1a + ratraw(irsiap)*sc1adt;
    //dratdumdd(irsiap) = dratrawdd(irsiap)*sc1a + ratraw(irsiap)*sc1add;

    ratdum(irppa)     = ratraw(irppa) * sc1a;
    dratdumdt(irppa)  = dratrawdt(irppa)*sc1a  + ratraw(irppa)*sc1adt;
    //dratdumdd(irppa)  = dratrawdd(irppa)*sc1a  + ratraw(irppa)*sc1add;


    jscr++;
    screen5(state, jscr,
            15.0e0_rt, 31.0e0_rt, 1.0e0_rt, 1.0e0_rt,
            sc1a, sc1adt, sc1add);

    // p31(p,g)s32
    ratdum(irppg)     = ratraw(irppg) * sc1a;
    dratdumdt(irppg)  = dratrawdt(irppg)*sc1a + ratraw(irppg)*sc1adt;
    //dratdumdd(irppg)  = dratrawdd(irppg)*sc1a + ratraw(irppg)*sc1add;

    ratdum(irsgp)     = ratraw(irsgp) * sc1a;
    dratdumdt(irsgp)  = dratrawdt(irsgp)*sc1a + ratraw(irsgp)*sc1adt;
    //dratdumdd(irsgp)  = dratrawdd(irsgp)*sc1a + ratraw(irsgp)*sc1add;



    // s32 to ar36
    jscr++;
    screen5(state, jscr,
            zion[S32-1], aion[S32-1], zion[He4-1], aion[He4-1],
            sc1a, sc1adt, sc1add);


    // s32(a,g)ar36
    ratdum(irsag)     = ratraw(irsag) * sc1a;
    dratdumdt(irsag)  = dratrawdt(irsag)*sc1a + ratraw(irsag)*sc1adt;
    //dratdumdd(irsag)  = dratrawdd(irsag)*sc1a + ratraw(irsag)*sc1add;

    ratdum(irarga)     = ratraw(irarga) * sc1a;
    dratdumdt(irarga)  = dratrawdt(irarga)*sc1a + ratraw(irarga)*sc1adt;
    //dratdumdd(irarga)  = dratrawdd(irarga)*sc1a + ratraw(irarga)*sc1add;

    // s32(a,p)cl35
    ratdum(irsap)     = ratraw(irsap) * sc1a;
    dratdumdt(irsap)  = dratrawdt(irsap)*sc1a + ratraw(irsap)*sc1adt;
    //dratdumdd(irsap)  = dratrawdd(irsap)*sc1a + ratraw(irsap)*sc1add;

    ratdum(irclpa)    = ratraw(irclpa) * sc1a;
    dratdumdt(irclpa) = dratrawdt(irclpa)*sc1a + ratraw(irclpa)*sc1adt;
    //dratdumdd(irclpa) = dratrawdd(irclpa)*sc1a + ratraw(irclpa)*sc1add;


    jscr++;
    screen5(state, jscr,
            17.0e0_rt, 35.0e0_rt, 1.0e0_rt, 1.0e0_rt,
            sc1a, sc1adt, sc1add);

    // cl35(p,g)ar36
    ratdum(irclpg)    = ratraw(irclpg) * sc1a;
    dratdumdt(irclpg) = dratrawdt(irclpg)*sc1a + ratraw(irclpg)*sc1adt;
    //dratdumdd(irclpg) = dratrawdd(irclpg)*sc1a + ratraw(irclpg)*sc1add;

    ratdum(irargp)    = ratraw(irargp) * sc1a;
    dratdumdt(irargp) = dratrawdt(irargp)*sc1a + ratraw(irargp)*sc1adt;
    //dratdumdd(irargp) = dratrawdd(irargp)*sc1a + ratraw(irargp)*sc1add;




    // ar36 to ca40
    jscr++;
    screen5(state, jscr,
            zion[Ar36-1], aion[Ar36-1], zion[He4-1], aion[He4-1],
            sc1a, sc1adt, sc1add);


    // ar36(a,g)ca40
    ratdum(irarag)    = ratraw(irarag) * sc1a;
    dratdumdt(irarag) = dratrawdt(irarag)*sc1a + ratraw(irarag)*sc1adt;
    //dratdumdd(irarag) = dratrawdd(irarag)*sc1a + ratraw(irarag)*sc1add;

    ratdum(ircaga)    = ratraw(ircaga) * sc1a;
    dratdumdt(ircaga) = dratrawdt(ircaga)*sc1a + ratraw(ircaga)*sc1adt;
    //dratdumdd(ircaga) = dratrawdd(ircaga)*sc1a + ratraw(ircaga)*sc1add;


    // ar36(a,p)k39
    ratdum(irarap)    = ratraw(irarap) * sc1a;
    dratdumdt(irarap) = dratrawdt(irarap)*sc1a + ratraw(irarap)*sc1adt;
    //dratdumdd(irarap) = dratrawdd(irarap)*sc1a + ratraw(irarap)*sc1add;

    ratdum(irkpa)     = ratraw(irkpa) * sc1a;
    dratdumdt(irkpa)  = dratrawdt(irkpa)*sc1a  + ratraw(irkpa)*sc1adt;
    //dratdumdd(irkpa)  = dratrawdd(irkpa)*sc1a  + ratraw(irkpa)*sc1add;



    jscr++;
    screen5(state, jscr,
            19.0e0_rt, 39.0e0_rt, 1.0e0_rt, 1.0e0_rt,
            sc1a, sc1adt, sc1add);

    // k39(p,g)ca40
    ratdum(irkpg)     = ratraw(irkpg) * sc1a;
    dratdumdt(irkpg)  = dratrawdt(irkpg)*sc1a  + ratraw(irkpg)*sc1adt;
    //dratdumdd(irkpg)  = dratrawdd(irkpg)*sc1a  + ratraw(irkpg)*sc1add;


    ratdum(ircagp)     = ratraw(ircagp) * sc1a;
    dratdumdt(ircagp)  = dratrawdt(ircagp)*sc1a  + ratraw(ircagp)*sc1adt;
    //dratdumdd(ircagp)  = dratrawdd(ircagp)*sc1a  + ratraw(ircagp)*sc1add;



    // ca40 to ti44
    jscr++;
    screen5(state, jscr,
            zion[Ca40-1], aion[Ca40-1], zion[He4-1], aion[He4-1],
            sc1a, sc1adt, sc1add);


    // ca40(a,g)ti44
    ratdum(ircaag)    = ratraw(ircaag) * sc1a;
    dratdumdt(ircaag) = dratrawdt(ircaag)*sc1a + ratraw(ircaag)*sc1adt;
    //dratdumdd(ircaag) = dratrawdd(ircaag)*sc1a + ratraw(ircaag)*sc1add;

    ratdum(irtiga)    = ratraw(irtiga) * sc1a;
    dratdumdt(irtiga) = dratrawdt(irtiga)*sc1a + ratraw(irtiga)*sc1adt;
    //dratdumdd(irtiga) = dratrawdd(irtiga)*sc1a + ratraw(irtiga)*sc1add;


    // ca40(a,p)sc43
    ratdum(ircaap)    = ratraw(ircaap) * sc1a;
    dratdumdt(ircaap) = dratrawdt(ircaap)*sc1a + ratraw(ircaap)*sc1adt;
    //dratdumdd(ircaap) = dratrawdd(ircaap)*sc1a + ratraw(ircaap)*sc1add;

    ratdum(irscpa)    = ratraw(irscpa) * sc1a;
    dratdumdt(irscpa) = dratrawdt(irscpa)*sc1a + ratraw(irscpa)*sc1adt;
    //dratdumdd(irscpa) = dratrawdd(irscpa)*sc1a + ratraw(irscpa)*sc1add;



    jscr++;
    screen5(state, jscr,
            21.0e0_rt, 43.0e0_rt, 1.0e0_rt, 1.0e0_rt,
            sc1a, sc1adt, sc1add);

    // sc43(p,g)ti44
    ratdum(irscpg)    = ratraw(irscpg) * sc1a;
    dratdumdt(irscpg) = dratrawdt(irscpg)*sc1a + ratraw(irscpg)*sc1adt;
    //dratdumdd(irscpg) = dratrawdd(irscpg)*sc1a + ratraw(irscpg)*sc1add;

    ratdum(irtigp)    = ratraw(irtigp) * sc1a;
    dratdumdt(irtigp) = dratrawdt(irtigp)*sc1a + ratraw(irtigp)*sc1adt;
    //dratdumdd(irtigp) = dratrawdd(irtigp)*sc1a + ratraw(irtigp)*sc1add;



    // ti44 to cr48
    jscr++;
    screen5(state, jscr,
            zion[Ti44-1], aion[Ti44-1], zion[He4-1], aion[He4-1],
            sc1a, sc1adt, sc1add);


    // ti44(a,g)cr48
    ratdum(irtiag)    = ratraw(irtiag) * sc1a;
    dratdumdt(irtiag) = dratrawdt(irtiag)*sc1a + ratraw(irtiag)*sc1adt;
    //dratdumdd(irtiag) = dratrawdd(irtiag)*sc1a + ratraw(irtiag)*sc1add;

    ratdum(ircrga)    = ratraw(ircrga) * sc1a;
    dratdumdt(ircrga) = dratrawdt(ircrga)*sc1a + ratraw(ircrga)*sc1adt;
    //dratdumdd(ircrga) = dratrawdd(ircrga)*sc1a + ratraw(ircrga)*sc1add;

    // ti44(a,p)v47
    ratdum(irtiap)    = ratraw(irtiap) * sc1a;
    dratdumdt(irtiap) = dratrawdt(irtiap)*sc1a + ratraw(irtiap)*sc1adt;
    //dratdumdd(irtiap) = dratrawdd(irtiap)*sc1a + ratraw(irtiap)*sc1add;

    ratdum(irvpa)     = ratraw(irvpa) * sc1a;
    dratdumdt(irvpa)  = dratrawdt(irvpa)*sc1a  + ratraw(irvpa)*sc1adt;
    //dratdumdd(irvpa)  = dratrawdd(irvpa)*sc1a  + ratraw(irvpa)*sc1add;


    jscr++;
    screen5(state, jscr,
            23.0e0_rt, 47.0e0_rt, 1.0e0_rt, 1.0e0_rt,
            sc1a, sc1adt, sc1add);

    // v47(p,g)cr48
    ratdum(irvpg)     = ratraw(irvpg) * sc1a;
    dratdumdt(irvpg)  = dratrawdt(irvpg)*sc1a  + ratraw(irvpg)*sc1adt;
    //dratdumdd(irvpg)  = dratrawdd(irvpg)*sc1a  + ratraw(irvpg)*sc1add;

    ratdum(ircrgp)     = ratraw(ircrgp) * sc1a;
    dratdumdt(ircrgp)  = dratrawdt(ircrgp)*sc1a  + ratraw(ircrgp)*sc1adt;
    //dratdumdd(ircrgp)  = dratrawdd(ircrgp)*sc1a  + ratraw(ircrgp)*sc1add;



    // cr48 to fe52
    jscr++;
    screen5(state, jscr,
            zion[Cr48-1], aion[Cr48-1], zion[He4-1], aion[He4-1],
            sc1a, sc1adt, sc1add);


    // cr48(a,g)fe52
    ratdum(ircrag)    = ratraw(ircrag) * sc1a;
    dratdumdt(ircrag) = dratrawdt(ircrag)*sc1a + ratraw(ircrag)*sc1adt;
    //dratdumdd(ircrag) = dratrawdd(ircrag)*sc1a + ratraw(ircrag)*sc1add;

    ratdum(irfega)    = ratraw(irfega) * sc1a;
    dratdumdt(irfega) = dratrawdt(irfega)*sc1a + ratraw(irfega)*sc1adt;
    //dratdumdd(irfega) = dratrawdd(irfega)*sc1a + ratraw(irfega)*sc1add;



    // cr48(a,p)mn51
    ratdum(ircrap)    = ratraw(ircrap) * sc1a;
    dratdumdt(ircrap) = dratrawdt(ircrap)*sc1a + ratraw(ircrap)*sc1adt;
    //dratdumdd(ircrap) = dratrawdd(ircrap)*sc1a + ratraw(ircrap)*sc1add;

    ratdum(irmnpa)    = ratraw(irmnpa) * sc1a;
    dratdumdt(irmnpa) = dratrawdt(irmnpa)*sc1a + ratraw(irmnpa)*sc1adt;
    //dratdumdd(irmnpa) = dratrawdd(irmnpa)*sc1a + ratraw(irmnpa)*sc1add;


    jscr++;
    screen5(state, jscr,
            25.0e0_rt, 51.0e0_rt, 1.0e0_rt, 1.0e0_rt,
            sc1a, sc1adt, sc1add);

    // mn51(p,g)fe52
    ratdum(irmnpg)    = ratraw(irmnpg) * sc1a;
    dratdumdt(irmnpg) = dratrawdt(irmnpg)*sc1a + ratraw(irmnpg)*sc1adt;
    //dratdumdd(irmnpg) = dratrawdd(irmnpg)*sc1a + ratraw(irmnpg)*sc1add;

    ratdum(irfegp)    = ratraw(irfegp) * sc1a;
    dratdumdt(irfegp) = dratrawdt(irfegp)*sc1a + ratraw(irfegp)*sc1adt;
    //dratdumdd(irfegp) = dratrawdd(irfegp)*sc1a + ratraw(irfegp)*sc1add;


    // fe to ni
    jscr++;
    screen5(state, jscr,
            zion[Fe52-1], aion[Fe52-1], zion[He4-1], aion[He4-1],
            sc1a, sc1adt, sc1add);


    // fe52(a,g)ni56
    ratdum(irfeag)    = ratraw(irfeag) * sc1a;
    dratdumdt(irfeag) = dratrawdt(irfeag)*sc1a + ratraw(irfeag)*sc1adt;
    //dratdumdd(irfeag) = dratrawdd(irfeag)*sc1a + ratraw(irfeag)*sc1add;

    ratdum(irniga)    = ratraw(irniga) * sc1a;
    dratdumdt(irniga) = dratrawdt(irniga)*sc1a + ratraw(irniga)*sc1adt;
    //dratdumdd(irniga) = dratrawdd(irniga)*sc1a + ratraw(irniga)*sc1add;


    // fe52(a,p)co55
    ratdum(irfeap) = ratraw(irfeap) * sc1a;
    dratdumdt(irfeap) = dratrawdt(irfeap)*sc1a + ratraw(irfeap)*sc1adt;
    //dratdumdd(irfeap) = dratrawdd(irfeap)*sc1a + ratraw(irfeap)*sc1add;

    ratdum(ircopa)    = ratraw(ircopa) * sc1a;
    dratdumdt(ircopa) = dratrawdt(ircopa)*sc1a + ratraw(ircopa)*sc1adt;
    //dratdumdd(ircopa) = dratrawdd(ircopa)*sc1a + ratraw(ircopa)*sc1add;


    jscr++;
    screen5(state, jscr,
            27.0e0_rt, 55.0e0_rt, 1.0e0_rt, 1.0e0_rt,
            sc1a, sc1adt, sc1add);


    // co55(p,g)ni56
    ratdum(ircopg)    = ratraw(ircopg) * sc1a;
    dratdumdt(ircopg) = dratrawdt(ircopg)*sc1a + ratraw(ircopg)*sc1adt;
    //dratdumdd(ircopg) = dratrawdd(ircopg)*sc1a + ratraw(ircopg)*sc1add;

    ratdum(irnigp)    = ratraw(irnigp) * sc1a;
    dratdumdt(irnigp) = dratrawdt(irnigp)*sc1a + ratraw(irnigp)*sc1adt;
    //dratdumdd(irnigp) = dratrawdd(irnigp)*sc1a + ratraw(irnigp)*sc1add;


    jscr++;
    screen5(state, jscr,
            zion[Fe54-1], aion[Fe54-1], 1.0e0_rt, 1.0e0_rt,
            sc1a, sc1adt, sc1add);

    // fe54(p,g)co55
    ratdum(irfepg)    = ratraw(irfepg) * sc1a;
    dratdumdt(irfepg) = dratrawdt(irfepg)*sc1a + ratraw(irfepg)*sc1adt;
    //dratdumdd(irfepg) = dratrawdd(irfepg)*sc1a + ratraw(irfepg)*sc1add;

    ratdum(ircogp)    = ratraw(ircogp) * sc1a;
    dratdumdt(ircogp) = dratrawdt(ircogp)*sc1a + ratraw(ircogp)*sc1adt;
    //dratdumdd(ircogp) = dratrawdd(ircogp)*sc1a + ratraw(ircogp)*sc1add;



    // fe54(a,p)co57(g,p)fe56
    jscr++;
    screen5(state, jscr,
            zion[Fe54-1], aion[Fe54-1], zion[He4-1], aion[He4-1],
            sc1a, sc1adt, sc1add);


    // fe54(a,p)co57
    ratdum(irfe54ap)    = ratraw(irfe54ap) * sc1a;
    dratdumdt(irfe54ap) = dratrawdt(irfe54ap)*sc1a + ratraw(irfe54ap)*sc1adt;
    //dratdumdd(irfe54ap) = dratrawdd(irfe54ap)*sc1a + ratraw(irfe54ap)*sc1add;

    ratdum(irco57pa)    = ratraw(irco57pa) * sc1a;
    dratdumdt(irco57pa) = dratrawdt(irco57pa)*sc1a + ratraw(irco57pa)*sc1adt;
    //dratdumdd(irco57pa) = dratrawdd(irco57pa)*sc1a + ratraw(irco57pa)*sc1add;


    jscr++;
    screen5(state, jscr,
            zion[Fe56-1], aion[Fe56-1], 1.0e0_rt, 1.0e0_rt,
            sc1a, sc1adt, sc1add);

    // fe56(p,g)co57
    ratdum(irfe56pg)    = ratraw(irfe56pg) * sc1a;
    dratdumdt(irfe56pg) = dratrawdt(irfe56pg)*sc1a + ratraw(irfe56pg)*sc1adt;
    //dratdumdd(irfe56pg) = dratrawdd(irfe56pg)*sc1a + ratraw(irfe56pg)*sc1add;

    ratdum(irco57gp)    = ratraw(irco57gp) * sc1a;
    dratdumdt(irco57gp) = dratrawdt(irco57gp)*sc1a + ratraw(irco57gp)*sc1adt;
    //dratdumdd(irco57gp) = dratrawdd(irco57gp)*sc1a + ratraw(irco57gp)*sc1add;



    // d(p,g)he4
    jscr++;
    screen5(state, jscr,
            1.0e0_rt, 2.0e0_rt, zion[H1-1], aion[H1-1],
            sc1a, sc1adt, sc1add);


    // d(p,g)he3
    ratdum(irdpg)    = ratraw(irdpg) * sc1a;
    dratdumdt(irdpg) = dratrawdt(irdpg)*sc1a + ratraw(irdpg)*sc1adt;
    //dratdumdd(irdpg) = dratrawdd(irdpg)*sc1a + ratraw(irdpg)*sc1add;

    ratdum(irhegp)    = ratraw(irhegp) * sc1a;
    dratdumdt(irhegp) = dratrawdt(irhegp)*sc1a + ratraw(irhegp)*sc1adt;
    //dratdumdd(irhegp) = dratrawdd(irhegp)*sc1a + ratraw(irhegp)*sc1add;



    // pp
    jscr++;
    screen5(state, jscr,
            zion[H1-1], aion[H1-1], zion[H1-1], aion[H1-1],
            sc1a, sc1adt, sc1add);

    ratdum(irpp)    = ratraw(irpp) * sc1a;
    dratdumdt(irpp) = dratrawdt(irpp)*sc1a + ratraw(irpp)*sc1adt;
    //dratdumdd(irpp) = dratrawdd(irpp)*sc1a + ratraw(irpp)*sc1add;

    // Identical particle factor (H1 + H1)

    ratdum(irpp) *= 0.5_rt;
    dratdumdt(irpp) *= 0.5_rt;



    // he3 + he3
    jscr++;
    screen5(state, jscr,
            zion[He3-1], aion[He3-1], zion[He3-1], aion[He3-1],
            sc1a, sc1adt, sc1add);

    // he3(he3,2p)he4
    ratdum(ir33)    = ratraw(ir33) * sc1a;
    dratdumdt(ir33) = dratrawdt(ir33)*sc1a + ratraw(ir33)*sc1adt;
    //dratdumdd(ir33) = dratrawdd(ir33)*sc1a + ratraw(ir33)*sc1add;

    // Identical particle factor (He3 + He3)

    ratdum(ir33) *= 0.5_rt;
    dratdumdt(ir33) *= 0.5_rt;



    // he3 + he4
    jscr++;
    screen5(state, jscr,
            zion[He3-1], aion[He3-1], zion[He4-1], aion[He4-1],
            sc1a, sc1adt, sc1add);

    // he3(he4,g)be7
    ratdum(irhe3ag)    = ratraw(irhe3ag) * sc1a;
    dratdumdt(irhe3ag) = dratrawdt(irhe3ag)*sc1a + ratraw(irhe3ag)*sc1adt;
    //dratdumdd(irhe3ag) = dratrawdd(irhe3ag)*sc1a + ratraw(irhe3ag)*sc1add;



    // cno cycles

    jscr++;
    screen5(state, jscr,
            zion[C12-1], aion[C12-1], zion[H1-1], aion[H1-1],
            sc1a, sc1adt, sc1add);

    // c12(p,g)13n
    ratdum(ircpg)    = ratraw(ircpg) * sc1a;
    dratdumdt(ircpg) = dratrawdt(ircpg)*sc1a + ratraw(ircpg)*sc1adt;
    //dratdumdd(ircpg) = dratrawdd(ircpg)*sc1a + ratraw(ircpg)*sc1add;


    jscr++;
    screen5(state, jscr,
            zion[N14-1], aion[N14-1], zion[H1-1], aion[H1-1],
            sc1a, sc1adt, sc1add);

    // n14(p,g)o15
    ratdum(irnpg)    = ratraw(irnpg) * sc1a;
    dratdumdt(irnpg) = dratrawdt(irnpg)*sc1a + ratraw(irnpg)*sc1adt;
    //dratdumdd(irnpg) = dratrawdd(irnpg)*sc1a + ratraw(irnpg)*sc1add;


    jscr++;
    screen5(state, jscr,
            zion[O16-1], aion[O16-1], zion[H1-1], aion[H1-1],
            sc1a, sc1adt, sc1add);


    // o16(p,g)f17
    ratdum(iropg)    = ratraw(iropg) * sc1a;
    dratdumdt(iropg) = dratrawdt(iropg)*sc1a+ratraw(iropg)*sc1adt;
    //dratdumdd(iropg) = dratrawdd(iropg)*sc1a+ratraw(iropg)*sc1add;


    jscr++;
    screen5(state, jscr,
            zion[N14-1], aion[N14-1], zion[He4-1], aion[He4-1],
            sc1a, sc1adt, sc1add);

    // n14(a,g)f18
    ratdum(irnag)    = ratraw(irnag) * sc1a;
    dratdumdt(irnag) = dratrawdt(irnag)*sc1a + ratraw(irnag)*sc1adt;
    //dratdumdd(irnag) = dratrawdd(irnag)*sc1a + ratraw(irnag)*sc1add;



    // now form those lovely equilibrium rates

    // mg24(a,p)27al(p,g)28si
    ratdum(irr1)     = 0.0_rt;
    dratdumdt(irr1)  = 0.0_rt;
    //dratdumdd(irr1)  = 0.0_rt;
    denom    = ratdum(iralpa) + ratdum(iralpg);
    denomdt  = dratdumdt(iralpa) + dratdumdt(iralpg);
    //denomdd  = dratdumdd(iralpa) + dratdumdd(iralpg);
    if (denom > 1.0e-50_rt) {
       zz = 1.0e0_rt/denom;
       ratdum(irr1)    = ratdum(iralpa)*zz;
       dratdumdt(irr1) = (dratdumdt(iralpa) - ratdum(irr1)*denomdt)*zz;
       //dratdumdd(irr1) = (dratdumdd(iralpa) - ratdum(irr1)*denomdd)*zz;
    }

    // si28(a,p)p31(p,g)s32
    ratdum(irs1)     = 0.0_rt;
    dratdumdt(irs1)  = 0.0_rt;
    //dratdumdd(irs1)  = 0.0_rt;
    denom    = ratdum(irppa) + ratdum(irppg);
    denomdt  = dratdumdt(irppa) + dratdumdt(irppg);
    //denomdd  = dratdumdd(irppa) + dratdumdd(irppg);
    if (denom > 1.0e-50_rt) {
        zz = 1.0e0_rt/denom;
        ratdum(irs1)    = ratdum(irppa)*zz;
        dratdumdt(irs1) = (dratdumdt(irppa) - ratdum(irs1)*denomdt)*zz;
        //dratdumdd(irs1) = (dratdumdd(irppa) - ratdum(irs1)*denomdd)*zz;
    }

    // s32(a,p)cl35(p,g)ar36
    ratdum(irt1)     = 0.0_rt;
    dratdumdt(irt1)  = 0.0_rt;
    //dratdumdd(irt1)  = 0.0_rt;
    denom    = ratdum(irclpa) + ratdum(irclpg);
    denomdt  = dratdumdt(irclpa) + dratdumdt(irclpg);
    //denomdd  = dratdumdd(irclpa) + dratdumdd(irclpg);
    if (denom > 1.0e-50_rt) {
        zz = 1.0e0_rt/denom;
        ratdum(irt1)    = ratdum(irclpa)*zz;
        dratdumdt(irt1) = (dratdumdt(irclpa) - ratdum(irt1)*denomdt)*zz;
        //dratdumdd(irt1) = (dratdumdd(irclpa) - ratdum(irt1)*denomdd)*zz;
    }

    // ar36(a,p)k39(p,g)ca40
    ratdum(iru1)     = 0.0_rt;
    dratdumdt(iru1)  = 0.0_rt;
    //dratdumdd(iru1)  = 0.0_rt;
    denom    = ratdum(irkpa) + ratdum(irkpg);
    denomdt  = dratdumdt(irkpa) + dratdumdt(irkpg);
    //denomdd  = dratdumdd(irkpa) + dratdumdd(irkpg);
    if (denom > 1.0e-50_rt) {
        zz   = 1.0e0_rt/denom;
        ratdum(iru1)   = ratdum(irkpa)*zz;
        dratdumdt(iru1) = (dratdumdt(irkpa) - ratdum(iru1)*denomdt)*zz;
        //dratdumdd(iru1) = (dratdumdd(irkpa) - ratdum(iru1)*denomdd)*zz;
    }

    // ca40(a,p)sc43(p,g)ti44
    ratdum(irv1)     = 0.0_rt;
    dratdumdt(irv1)  = 0.0_rt;
    //dratdumdd(irv1)  = 0.0_rt;
    denom    = ratdum(irscpa) + ratdum(irscpg);
    denomdt  = dratdumdt(irscpa) + dratdumdt(irscpg);
    //denomdd  = dratdumdd(irscpa) + dratdumdd(irscpg);
    if (denom > 1.0e-50_rt) {
        zz  = 1.0e0_rt/denom;
        ratdum(irv1)    = ratdum(irscpa)*zz;
        dratdumdt(irv1) = (dratdumdt(irscpa) - ratdum(irv1)*denomdt)*zz;
        //dratdumdd(irv1) = (dratdumdd(irscpa) - ratdum(irv1)*denomdd)*zz;
    }

    // ti44(a,p)v47(p,g)cr48
    ratdum(irw1)    = 0.0_rt;
    dratdumdt(irw1) = 0.0_rt;
    //dratdumdd(irw1) = 0.0_rt;
    denom    = ratdum(irvpa) + ratdum(irvpg);
    denomdt  = dratdumdt(irvpa) + dratdumdt(irvpg);
    //denomdd  = dratdumdd(irvpa) + dratdumdd(irvpg);
    if (denom > 1.0e-50_rt) {
        zz = 1.0e0_rt/denom;
        ratdum(irw1)    = ratdum(irvpa)*zz;
        dratdumdt(irw1) = (dratdumdt(irvpa) - ratdum(irw1)*denomdt)*zz;
        //dratdumdd(irw1) = (dratdumdd(irvpa) - ratdum(irw1)*denomdd)*zz;
    }

    // cr48(a,p)mn51(p,g)fe52
    ratdum(irx1)    = 0.0_rt;
    dratdumdt(irx1) = 0.0_rt;
    //dratdumdd(irx1) = 0.0_rt;
    denom    = ratdum(irmnpa) + ratdum(irmnpg);
    denomdt  = dratdumdt(irmnpa) + dratdumdt(irmnpg);
    //denomdd  = dratdumdd(irmnpa) + dratdumdd(irmnpg);
    if (denom > 1.0e-50_rt) {
        zz = 1.0e0_rt/denom;
        ratdum(irx1)    = ratdum(irmnpa)*zz;
        dratdumdt(irx1) = (dratdumdt(irmnpa) - ratdum(irx1)*denomdt)*zz;
        //dratdumdd(irx1) = (dratdumdd(irmnpa) - ratdum(irx1)*denomdd)*zz;
    }


    // fe52(n,g)fe53(n,g)fe54 equilibrium links

    ratdum(ir1f54) = 0.0_rt;
    dratdumdy1(ir1f54) = 0.0_rt;
    dratdumdt(ir1f54)  = 0.0_rt;
    //dratdumdd(ir1f54)  = 0.0_rt;
    ratdum(ir2f54) = 0.0_rt;
    dratdumdy1(ir2f54) = 0.0_rt;
    dratdumdt(ir2f54)  = 0.0_rt;
    //dratdumdd(ir2f54)  = 0.0_rt;

    denom   = ratdum(ir53gn) + y(N)*ratdum(ir53ng);
    denomdt = dratdumdt(ir53gn) + y(N)*dratdumdt(ir53ng);
    //denomdd = dratdumdd(ir53gn) + y(N)*dratdumdd(ir53ng);

    if (denom > 1.0e-50_rt && btemp > 1.5e9_rt) {
        zz      = 1.0e0_rt/denom;

        ratdum(ir1f54)     = ratdum(ir54gn)*ratdum(ir53gn)*zz;
        dratdumdy1(ir1f54) = -ratdum(ir1f54)*zz * ratdum(ir53ng);
       dratdumdt(ir1f54)  = dratdumdt(ir54gn)*ratdum(ir53gn)*zz
                            + ratdum(ir54gn)*dratdumdt(ir53gn)*zz
                            - ratdum(ir1f54)*zz*denomdt;
       //dratdumdd(ir1f54)  = dratdumdd(ir54gn)*ratdum(ir53gn)*zz + ratdum(ir54gn)*dratdumdd(ir53gn)*zz - ratdum(ir1f54)*zz*denomdd;

       ratdum(ir2f54)     = ratdum(ir52ng)*ratdum(ir53ng)*zz;
       dratdumdy1(ir2f54) = -ratdum(ir2f54)*zz * ratdum(ir53ng);
       dratdumdt(ir2f54)  = dratdumdt(ir52ng)*ratdum(ir53ng)*zz
                            + ratdum(ir52ng)*dratdumdt(ir53ng)*zz
                            - ratdum(ir2f54)*zz*denomdt;
       //dratdumdd(ir2f54)  = dratdumdd(ir52ng)*ratdum(ir53ng)*zz + ratdum(ir52ng)*dratdumdd(ir53ng)*zz - ratdum(ir2f54)*zz*denomdd;
    }


    // fe54(n,g)fe55(n,g)fe56 equilibrium links

    ratdum(irfe56_aux1) = 0.0_rt;
    dratdumdy1(irfe56_aux1) = 0.0_rt;
    dratdumdt(irfe56_aux1)  = 0.0_rt;
    //dratdumdd(irfe56_aux1)  = 0.0_rt;
    ratdum(irfe56_aux2) = 0.0_rt;
    dratdumdy1(irfe56_aux2) = 0.0_rt;
    dratdumdt(irfe56_aux2)  = 0.0_rt;
    //dratdumdd(irfe56_aux2)  = 0.0_rt;

    denom   = ratdum(ir55gn)    + y(N)*ratdum(ir55ng);
    denomdt = dratdumdt(ir55gn) + y(N)*dratdumdt(ir55ng);
    //denomdd = dratdumdd(ir55gn) + y(N)*dratdumdd(ir55ng);

    if (denom > 1.0e-50_rt && btemp > 1.5e9_rt) {
        zz      = 1.0e0_rt/denom;

        ratdum(irfe56_aux1)     = ratdum(ir56gn)*ratdum(ir55gn)*zz;
        dratdumdy1(irfe56_aux1) = -ratdum(irfe56_aux1)*zz * ratdum(ir55ng);
        dratdumdt(irfe56_aux1)  = dratdumdt(ir56gn)*ratdum(ir55gn)*zz 
                                  + ratdum(ir56gn)*dratdumdt(ir55gn)*zz
                                  - ratdum(irfe56_aux1)*zz*denomdt;
        //dratdumdd(irfe56_aux1)  = dratdumdd(ir56gn)*ratdum(ir55gn)*zz &
        //                        + ratdum(ir56gn)*dratdumdd(ir55gn)*zz - ratdum(irfe56_aux1)*zz*denomdd;

        ratdum(irfe56_aux2)     = ratdum(ir54ng)*ratdum(ir55ng)*zz;
        dratdumdy1(irfe56_aux2) = -ratdum(irfe56_aux2)*zz * ratdum(ir55ng);
        dratdumdt(irfe56_aux2)  = dratdumdt(ir54ng)*ratdum(ir55ng)*zz 
                                  + ratdum(ir54ng)*dratdumdt(ir55ng)*zz
                                  - ratdum(irfe56_aux2)*zz*denomdt;
        //dratdumdd(irfe56_aux2)  = dratdumdd(ir54ng)*ratdum(ir55ng)*zz &
        //                        + ratdum(ir54ng)*dratdumdd(ir55ng)*zz  - ratdum(irfe56_aux2)*zz*denomdd
    }


    // fe54(a,p)co57(g,p)fe56 equilibrium links

    ratdum(irfe56_aux3)     = 0.0_rt;
    dratdumdy1(irfe56_aux3) = 0.0_rt;
    dratdumdt(irfe56_aux3)  = 0.0_rt;
    //dratdumdd(irfe56_aux3)  = 0.0_rt;
    ratdum(irfe56_aux4)     = 0.0_rt;
    dratdumdy1(irfe56_aux4) = 0.0_rt;
    dratdumdt(irfe56_aux4)  = 0.0_rt;
    //dratdumdd(irfe56_aux4)  = 0.0_rt;

    denom   = ratdum(irco57gp)    + y(P)*ratdum(irco57pa);
    denomdt = dratdumdt(irco57gp) + y(P)*dratdumdt(irco57pa);
    //denomdd = dratdumdd(irco57gp) + y(P)*dratdumdd(irco57pa);

    if (denom > 1.0e-50_rt && btemp > 1.5e9_rt) {
        zz      = 1.0e0_rt/denom;

        ratdum(irfe56_aux3)     = ratdum(irfe56pg) * ratdum(irco57pa) * zz;
        dratdumdy1(irfe56_aux3) = -ratdum(irfe56_aux3) * zz * ratdum(irco57pa);
        dratdumdt(irfe56_aux3)  = dratdumdt(irfe56pg) * ratdum(irco57pa) * zz
                                  + ratdum(irfe56pg) * dratdumdt(irco57pa) * zz
                                  - ratdum(irfe56_aux3) * zz * denomdt;
        //dratdumdd(irfe56_aux3)  = dratdumdd(irfe56pg) * ratdum(irco57pa) * zz &
        //                        + ratdum(irfe56pg) * dratdumdd(irco57pa) * zz - ratdum(irfe56_aux3) * zz * denomdd

        ratdum(irfe56_aux4)     = ratdum(irfe54ap) * ratdum(irco57gp) * zz;
        dratdumdy1(irfe56_aux4) = -ratdum(irfe56_aux4) * zz * ratdum(irco57pa);
        dratdumdt(irfe56_aux4)  = dratdumdt(irfe54ap) * ratdum(irco57gp) * zz
                                  + ratdum(irfe54ap) * dratdumdt(irco57gp) * zz
                                  - ratdum(irfe56_aux4) * zz * denomdt;
        //dratdumdd(irfe56_aux4)  = dratdumdd(irfe54ap) * ratdum(irco57gp) * zz &
        //                        + ratdum(irfe54ap) * dratdumdd(irco57gp) * zz  - ratdum(irfe56_aux4) * zz * denomdd
    }


    // fe54(p,g)co55(p,g)ni56 equilibrium links r3f54 r4f54
    // fe52(a,p)co55(g,p)fe54 equilibrium links r5f54 r6f54
    // fe52(a,p)co55(p,g)ni56 equilibrium links r7f54 r8f54

    ratdum(ir3f54) = 0.0_rt;
    dratdumdy1(ir3f54) = 0.0_rt;
    dratdumdt(ir3f54) = 0.0_rt;
    //dratdumdd(ir3f54) = 0.0_rt;
    ratdum(ir4f54) = 0.0_rt;
    dratdumdy1(ir4f54) = 0.0_rt;
    dratdumdt(ir4f54) = 0.0_rt;
    //dratdumdd(ir4f54) = 0.0_rt;
    ratdum(ir5f54) = 0.0_rt;
    dratdumdy1(ir5f54) = 0.0_rt;
    dratdumdt(ir5f54) = 0.0_rt;
    //dratdumdd(ir5f54) = 0.0_rt;
    ratdum(ir6f54) = 0.0_rt;
    dratdumdy1(ir6f54) = 0.0_rt;
    dratdumdt(ir6f54) = 0.0_rt;
    //dratdumdd(ir6f54) = 0.0_rt;
    ratdum(ir7f54) = 0.0_rt;
    dratdumdy1(ir7f54) = 0.0_rt;
    dratdumdt(ir7f54) = 0.0_rt;
    //dratdumdd(ir7f54) = 0.0_rt;
    ratdum(ir8f54) = 0.0_rt;
    dratdumdy1(ir8f54) = 0.0_rt;
    dratdumdt(ir8f54) = 0.0_rt;
    //dratdumdd(ir8f54) = 0.0_rt;

    denom   = ratdum(ircogp) + y(P)*(ratdum(ircopg) + ratdum(ircopa));

    if (denom > 1.0e-50_rt && btemp > 1.5e9_rt) {

        denomdt = dratdumdt(ircogp) + y(P)*(dratdumdt(ircopg) + dratdumdt(ircopa));
        //denomdd = dratdumdd(ircogp) + y(P)*(dratdumdd(ircopg) + dratdumdd(ircopa));

        zz      = 1.0e0_rt/denom;

        ratdum(ir3f54)     = ratdum(irfepg) * ratdum(ircopg) * zz;
        dratdumdy1(ir3f54) = -ratdum(ir3f54) * zz * (ratdum(ircopg) + ratdum(ircopa));
        dratdumdt(ir3f54)  = dratdumdt(irfepg) * ratdum(ircopg) * zz
                             + ratdum(irfepg) * dratdumdt(ircopg) * zz
                             - ratdum(ir3f54)*zz*denomdt;
        //dratdumdd(ir3f54)  = dratdumdd(irfepg) * ratdum(ircopg) * zz &
        //                   + ratdum(irfepg) * dratdumdd(ircopg) * zz - ratdum(ir3f54)*zz*denomdd;

        ratdum(ir4f54)     = ratdum(irnigp) * ratdum(ircogp) * zz;
        dratdumdy1(ir4f54) = -ratdum(ir4f54) * zz * (ratdum(ircopg)+ratdum(ircopa));
        dratdumdt(ir4f54)  = dratdumdt(irnigp) * ratdum(ircogp) * zz
                             + ratdum(irnigp) * dratdumdt(ircogp) * zz
                             - ratdum(ir4f54)*zz*denomdt;
        //dratdumdd(ir4f54)  = dratdumdd(irnigp) * ratdum(ircogp) * zz &
        //                   + ratdum(irnigp) * dratdumdd(ircogp) * zz  - ratdum(ir4f54)*zz*denomdd;

        ratdum(ir5f54)     = ratdum(irfepg) * ratdum(ircopa) * zz;
        dratdumdy1(ir5f54) = -ratdum(ir5f54) * zz * (ratdum(ircopg)+ratdum(ircopa));
        dratdumdt(ir5f54)  = dratdumdt(irfepg) * ratdum(ircopa) * zz
                             + ratdum(irfepg) * dratdumdt(ircopa) * zz
                             - ratdum(ir5f54) * zz * denomdt;
        //dratdumdd(ir5f54)  = dratdumdd(irfepg) * ratdum(ircopa) * zz &
        //                   + ratdum(irfepg) * dratdumdd(ircopa) * zz - ratdum(ir5f54) * zz * denomdd;

        ratdum(ir6f54)     = ratdum(irfeap) * ratdum(ircogp) * zz;
        dratdumdy1(ir6f54) = -ratdum(ir6f54) * zz * (ratdum(ircopg)+ratdum(ircopa));
        dratdumdt(ir6f54)  = dratdumdt(irfeap) * ratdum(ircogp) * zz
                             + ratdum(irfeap) * dratdumdt(ircogp) * zz
                             - ratdum(ir6f54) * zz * denomdt;
        //dratdumdd(ir6f54)  = dratdumdd(irfeap) * ratdum(ircogp) * zz &
        //                   + ratdum(irfeap) * dratdumdd(ircogp) * zz - ratdum(ir6f54) * zz * denomdd;

        ratdum(ir7f54)     = ratdum(irfeap) * ratdum(ircopg) * zz;
        dratdumdy1(ir7f54) = -ratdum(ir7f54) * zz * (ratdum(ircopg)+ratdum(ircopa));
        dratdumdt(ir7f54)  = dratdumdt(irfeap) * ratdum(ircopg) * zz
                             + ratdum(irfeap) * dratdumdt(ircopg) * zz
                             - ratdum(ir7f54) * zz * denomdt;
        //dratdumdd(ir7f54)  = dratdumdd(irfeap) * ratdum(ircopg) * zz &
        //                   + ratdum(irfeap) * dratdumdd(ircopg) * zz - ratdum(ir7f54) * zz * denomdd;

        ratdum(ir8f54)     = ratdum(irnigp) * ratdum(ircopa) * zz;
        dratdumdy1(ir8f54) = -ratdum(ir8f54) * zz * (ratdum(ircopg)+ratdum(ircopa));
        dratdumdt(ir8f54)  = dratdumdt(irnigp) * ratdum(ircopa) * zz
                             + ratdum(irnigp) * dratdumdt(ircopa) * zz
                             - ratdum(ir8f54) * zz * denomdt;
        //dratdumdd(ir8f54)  = dratdumdd(irnigp) * ratdum(ircopa) * zz &
        //                   + ratdum(irnigp) * dratdumdd(ircopa) * zz - ratdum(ir8f54) * zz * denomdd;
    }


    // p(n,g)h2(n,g)3h(p,g)he4   photodisintegrated n and p back to he4 equilibrium links
    // p(n,g)h2(p,g)he3(n,g)he4

    ratdum(iralf1)     = 0.0_rt;
    dratdumdy1(iralf1) = 0.0_rt;
    dratdumdy2(iralf1) = 0.0_rt;
    dratdumdt(iralf1)  = 0.0_rt;
    //dratdumdd(iralf1)  = 0.0_rt;
    ratdum(iralf2)     = 0.0_rt;
    dratdumdy1(iralf2) = 0.0_rt;
    dratdumdy2(iralf2) = 0.0_rt;
    dratdumdt(iralf2)  = 0.0_rt;
    //dratdumdd(iralf2)  = 0.0_rt;

    denom  = ratdum(irhegp)*ratdum(irdgn) + y(N)*ratdum(irheng)*ratdum(irdgn) + y(N)*y(P)*ratdum(irheng)*ratdum(irdpg);

    if (denom > 1.0e-50_rt && btemp > 1.5e9_rt) {

        denomdt  = dratdumdt(irhegp)*ratdum(irdgn) + ratdum(irhegp)*dratdumdt(irdgn)
                   + y(N) * (dratdumdt(irheng)*ratdum(irdgn) + ratdum(irheng)*dratdumdt(irdgn))
                   + y(N)*y(P) * (dratdumdt(irheng)*ratdum(irdpg) + ratdum(irheng)*dratdumdt(irdpg));

        //denomdd  = dratdumdd(irhegp)*ratdum(irdgn) + ratdum(irhegp)*dratdumdd(irdgn)
        //     +  y(N) * (dratdumdd(irheng)*ratdum(irdgn) + ratdum(irheng)*dratdumdd(irdgn))
        //     +  y(N)*y(P) * (dratdumdd(irheng)*ratdum(irdpg) + ratdum(irheng)*dratdumdd(irdpg));

        zz = 1.0e0_rt/denom;

        ratdum(iralf1)     = ratdum(irhegn) * ratdum(irhegp) * ratdum(irdgn) * zz;
        dratdumdy1(iralf1) = -ratdum(iralf1) * zz * (ratdum(irheng)*ratdum(irdgn) + y(P)*ratdum(irheng)*ratdum(irdpg));
        dratdumdy2(iralf1) = -ratdum(iralf1) * zz * y(N) * ratdum(irheng) * ratdum(irdpg);
        dratdumdt(iralf1)  = dratdumdt(irhegn)*ratdum(irhegp) * ratdum(irdgn) * zz
                             + ratdum(irhegn)*dratdumdt(irhegp)*ratdum(irdgn)*zz
                             + ratdum(irhegn)*ratdum(irhegp)*dratdumdt(irdgn)*zz
                             - ratdum(iralf1)*zz*denomdt;
        //dratdumdd(iralf1)  = dratdumdd(irhegn) * ratdum(irhegp) * ratdum(irdgn) * zz
        //     + ratdum(irhegn)*dratdumdd(irhegp)*ratdum(irdgn)*zz
        //     + ratdum(irhegn)*ratdum(irhegp)*dratdumdd(irdgn)*zz
        //     - ratdum(iralf1)*zz*denomdt;


        ratdum(iralf2)     = ratdum(irheng)*ratdum(irdpg) * ratdum(irhng)*zz;
        dratdumdy1(iralf2) = -ratdum(iralf2) * zz * (ratdum(irheng)*ratdum(irdgn) + y(P)*ratdum(irheng)*ratdum(irdpg));
        dratdumdy2(iralf2) = -ratdum(iralf2) * zz * y(N) * ratdum(irheng) * ratdum(irdpg);
        dratdumdt(iralf2)  = dratdumdt(irheng)*ratdum(irdpg) * ratdum(irhng) * zz
                             + ratdum(irheng)*dratdumdt(irdpg)*ratdum(irhng)*zz
                             + ratdum(irheng)*ratdum(irdpg)*dratdumdt(irhng)*zz
                             - ratdum(iralf2)*zz*denomdt;
        //dratdumdd(iralf2)  = dratdumdd(irheng)*ratdum(irdpg) * ratdum(irhng)*zz
        //     + ratdum(irheng)*dratdumdd(irdpg)*ratdum(irhng)*zz
        //     + ratdum(irheng)*ratdum(irdpg)*dratdumdd(irhng)*zz
        //     - ratdum(iralf2)*zz*denomdd;
    }



    // he3(a,g)be7(p,g)8b(e+nu)8be(2a)
    // beta limit he3+he4 by the 8B decay half life

    if (y(He4) > 1.0e-30_rt) {
        xx = 0.896e0_rt/y(He4);
        ratdum(irhe3ag) = amrex::min(ratdum(irhe3ag),xx);
        if (ratdum(irhe3ag) == xx) {
            dratdumdy1(irhe3ag) = -xx/y(He4);
            dratdumdt(irhe3ag)  = 0.0_rt;
            //dratdumdd(irhe3ag)  = 0.0_rt;
        } else {
            dratdumdy1(irhe3ag) = 0.0_rt;
        }
    }


    // beta limit n14(p,g)o15(enu)o16  and o16(p,g)f17(e+nu)17o(p,a)n14

    if (y(H1) > 1.0e-30_rt) {
       xx = 5.68e-03_rt/(y(H1)*1.57e0_rt);
       ratdum(irnpg) = amrex::min(ratdum(irnpg),xx);
       if (ratdum(irnpg) == xx) {
           dratdumdy1(irnpg) = -xx/y(H1);
           dratdumdt(irnpg)  = 0.0_rt;
           //dratdumdd(irnpg)  = 0.0_rt;
       } else {
           dratdumdy1(irnpg) = 0.0_rt;
       }

       xx = 0.0105e0_rt/y(H1);
       ratdum(iropg) = amrex::min(ratdum(iropg),xx);
       if (ratdum(iropg) == xx) {
           dratdumdy1(iropg) = -xx/y(H1);
           dratdumdt(iropg)  = 0.0_rt;
           //dratdumdd(iropg)  = 0.0_rt;
       } else {
           dratdumdy1(iropg) = 0.0_rt;
       }
    }
}


AMREX_GPU_HOST_DEVICE AMREX_INLINE
void evaluate_rates (burn_t const& state, Array1D<rate_t, 1, Rates::NumGroups>& rr)
{
    Array1D<Real, 1, NumRates> ratraw, dratrawdt, dratrawdd;
    Array1D<Real, 1, NumRates> ratdum, dratdumdt, dratdumdd;
    Array1D<Real, 1, NumRates> dratdumdy1, dratdumdy2;

    Real rho, temp;
    Array1D<Real, 1, NumSpec> y;

    // Get the data from the state

    rho  = state.rho;
    temp = state.T;

    for (int i = 1; i <= NumSpec; ++i) {
        y(i) = state.xn[i-1] * aion_inv[i-1];
    }

    // Get the raw reaction rates

    aprox21rat(temp, rho, ratraw, dratrawdt, dratrawdd);

    // Weak screening rates

    weak_aprox21(y, state, ratraw, dratrawdt, dratrawdd);

    // Do the screening here because the corrections depend on the composition

    screen_aprox21(temp, rho, y,
                   ratraw, dratrawdt, dratrawdd, 
                   ratdum, dratdumdt, dratdumdd,
                   dratdumdy1, dratdumdy2);

    // Save the rate data

    for (int i = 1; i <= NumRates; ++i) {
        rr(1).rates(i) = ratdum(i);
        rr(2).rates(i) = dratdumdt(i);
        rr(3).rates(i) = dratdumdy1(i);
        rr(4).rates(i) = dratdumdy2(i);
    }
}



// Evaluates the right hand side of the aprox21 ODEs
AMREX_GPU_HOST_DEVICE AMREX_INLINE
void rhs (Array1D<Real, 1, NumSpec> const& y,
          const Array1D<rate_t, 1, Rates::NumGroups>& rr,
          Array1D<Real, 1, neqs>& dydt,
          bool deriva, bool for_jacobian_tderiv)
{
    using namespace Species;

    // deriva is used in forming the analytic jacobian to get
    // the derivative wrt a

    Array1D<Real, 1, 20> a;

    int index_rate = 1;
    if (for_jacobian_tderiv) {
        index_rate = 2;
    }

    for (int i = 1; i <= NumSpec; ++i) {
        dydt(i) = 0.0_rt;
    }

    // hydrogen reactions
    a(1) = -3.0e0_rt * y(H1) * y(H1) * rr(index_rate).rates(irpp);
    a(2) =  2.0e0_rt * y(He3) * y(He3) * rr(index_rate).rates(ir33);
    a(3) = -y(He3) * y(He4) * rr(index_rate).rates(irhe3ag);
    a(4) = -2.0e0_rt * y(C12) * y(H1) * rr(index_rate).rates(ircpg);
    a(5) = -2.0e0_rt * y(N14) * y(H1) * rr(index_rate).rates(irnpg);
    a(6) = -2.0e0_rt * y(O16) * y(H1) * rr(index_rate).rates(iropg);
    a(7) = -3.0e0_rt * y(H1) * rr(index_rate).rates(irpen);

    dydt(H1) = dydt(H1) + esum7(a);



    // he3 reactions
    a(1)  =  y(H1) * y(H1) * rr(index_rate).rates(irpp);
    a(2)  = -2.0_rt * y(He3) * y(He3) * rr(index_rate).rates(ir33);
    a(3)  = -y(He3) * y(He4) * rr(index_rate).rates(irhe3ag);
    a(4)  =  y(H1) * rr(index_rate).rates(irpen);

    dydt(He3) = dydt(He3) + esum4(a);


    // he4 reactions
    // heavy ion reactions
    a(1)  = y(C12) * y(C12) * rr(index_rate).rates(ir1212);
    a(2)  = 0.5e0_rt * y(C12) * y(O16) * rr(index_rate).rates(ir1216);
    a(3)  = 0.56e0_rt * y(O16) * y(O16) * rr(index_rate).rates(ir1616);
    dydt(He4) =  dydt(He4) + esum3(a);


    // (a,g) and (g,a) reactions
    a(1)  = -3.0e0_rt * y(He4) * y(He4) * y(He4) * rr(index_rate).rates(ir3a);
    a(2)  =  3.0e0_rt * y(C12) * rr(index_rate).rates(irg3a);
    a(3)  = -y(He4)  * y(C12) * rr(index_rate).rates(ircag);
    a(4)  =  y(O16)  * rr(index_rate).rates(iroga);
    a(5)  = -y(He4)  * y(O16) * rr(index_rate).rates(iroag);
    a(6)  =  y(Ne20) * rr(index_rate).rates(irnega);
    a(7)  = -y(He4)  * y(Ne20) * rr(index_rate).rates(irneag);
    a(8)  =  y(Mg24) * rr(index_rate).rates(irmgga);
    a(9)  = -y(He4)  * y(Mg24)* rr(index_rate).rates(irmgag);
    a(10) =  y(Si28) * rr(index_rate).rates(irsiga);
    a(11) = -y(He4)  * y(Si28)*rr(index_rate).rates(irsiag);
    a(12) =  y(S32)  * rr(index_rate).rates(irsga);

    dydt(He4) =  dydt(He4) + esum12(a);

    a(1)  = -y(He4)  * y(S32) * rr(index_rate).rates(irsag);
    a(2)  =  y(Ar36) * rr(index_rate).rates(irarga);
    a(3)  = -y(He4)  * y(Ar36)*rr(index_rate).rates(irarag);
    a(4)  =  y(Ca40) * rr(index_rate).rates(ircaga);
    a(5)  = -y(He4)  * y(Ca40)*rr(index_rate).rates(ircaag);
    a(6)  =  y(Ti44) * rr(index_rate).rates(irtiga);
    a(7)  = -y(He4)  * y(Ti44)*rr(index_rate).rates(irtiag);
    a(8)  =  y(Cr48) * rr(index_rate).rates(ircrga);
    a(9)  = -y(He4)  * y(Cr48)*rr(index_rate).rates(ircrag);
    a(10) =  y(Fe52) * rr(index_rate).rates(irfega);
    a(11) = -y(He4)  * y(Fe52) * rr(index_rate).rates(irfeag);
    a(12) =  y(Ni56) * rr(index_rate).rates(irniga);

    dydt(He4) =  dydt(He4) + esum12(a);


    // (a,p)(p,g) and (g,p)(p,a) reactions

    if (!deriva) {
       a(1)  =  0.34e0_rt*y(O16)*y(O16)*rr(index_rate).rates(irs1)*rr(index_rate).rates(ir1616);
       a(2)  = -y(He4)  * y(Mg24) * rr(index_rate).rates(irmgap)*(1.0e0_rt-rr(index_rate).rates(irr1));
       a(3)  =  y(Si28) * rr(index_rate).rates(irsigp) * rr(index_rate).rates(irr1);
       a(4)  = -y(He4)  * y(Si28) * rr(index_rate).rates(irsiap)*(1.0e0_rt-rr(index_rate).rates(irs1));
       a(5)  =  y(S32)  * rr(index_rate).rates(irsgp) * rr(index_rate).rates(irs1);
       a(6)  = -y(He4)  * y(S32) * rr(index_rate).rates(irsap)*(1.0e0_rt-rr(index_rate).rates(irt1));
       a(7)  =  y(Ar36) * rr(index_rate).rates(irargp) * rr(index_rate).rates(irt1);
       a(8)  = -y(He4)  * y(Ar36) * rr(index_rate).rates(irarap)*(1.0e0_rt-rr(index_rate).rates(iru1));
       a(9)  =  y(Ca40) * rr(index_rate).rates(ircagp) * rr(index_rate).rates(iru1);
       a(10) = -y(He4)  * y(Ca40) * rr(index_rate).rates(ircaap)*(1.0e0_rt-rr(index_rate).rates(irv1));
       a(11) =  y(Ti44) * rr(index_rate).rates(irtigp) * rr(index_rate).rates(irv1);
       a(12) = -y(He4)  * y(Ti44) * rr(index_rate).rates(irtiap)*(1.0e0_rt-rr(index_rate).rates(irw1));
       a(13) =  y(Cr48) * rr(index_rate).rates(ircrgp) * rr(index_rate).rates(irw1);
       a(14) = -y(He4)  * y(Cr48) * rr(index_rate).rates(ircrap)*(1.0e0_rt-rr(index_rate).rates(irx1));
       a(15) =  y(Fe52) * rr(index_rate).rates(irfegp) * rr(index_rate).rates(irx1);

       dydt(He4) =  dydt(He4) + esum15(a);

    } else {
       a(1)  =  0.34e0_rt*y(O16)*y(O16) * rr(1).rates(irs1) * rr(index_rate).rates(ir1616);
       a(2)  =  0.34e0_rt*y(O16)*y(O16) * rr(index_rate).rates(irs1) * rr(1).rates(ir1616);
       a(3)  = -y(He4)*y(Mg24) * rr(index_rate).rates(irmgap)*(1.0e0_rt - rr(1).rates(irr1));
       a(4)  =  y(He4)*y(Mg24) * rr(1).rates(irmgap)*rr(index_rate).rates(irr1);
       a(5)  =  y(Si28) * rr(1).rates(irsigp) * rr(index_rate).rates(irr1);
       a(6)  =  y(Si28) * rr(index_rate).rates(irsigp) * rr(1).rates(irr1);
       a(7)  = -y(He4)*y(Si28) * rr(index_rate).rates(irsiap)*(1.0e0_rt - rr(1).rates(irs1));
       a(8)  =  y(He4)*y(Si28) * rr(1).rates(irsiap) * rr(index_rate).rates(irs1);
       a(9)  =  y(S32)  * rr(1).rates(irsgp) * rr(index_rate).rates(irs1);
       a(10) =  y(S32)  * rr(index_rate).rates(irsgp) * rr(1).rates(irs1);

       dydt(He4) =  dydt(He4) + esum10(a);

       a(1)  = -y(He4)*y(S32) * rr(index_rate).rates(irsap)*(1.0e0_rt - rr(1).rates(irt1));
       a(2)  =  y(He4)*y(S32) * rr(1).rates(irsap)*rr(index_rate).rates(irt1);
       a(3)  =  y(Ar36) * rr(1).rates(irargp) * rr(index_rate).rates(irt1);
       a(4)  =  y(Ar36) * rr(index_rate).rates(irargp) * rr(1).rates(irt1);
       a(5)  = -y(He4)*y(Ar36) * rr(index_rate).rates(irarap)*(1.0e0_rt - rr(1).rates(iru1));
       a(6)  =  y(He4)*y(Ar36) * rr(1).rates(irarap)*rr(index_rate).rates(iru1);
       a(7)  =  y(Ca40) * rr(1).rates(ircagp) * rr(index_rate).rates(iru1);
       a(8)  =  y(Ca40) * rr(index_rate).rates(ircagp) * rr(1).rates(iru1);
       a(9)  = -y(He4)*y(Ca40) * rr(index_rate).rates(ircaap)*(1.0e0_rt-rr(1).rates(irv1));
       a(10) =  y(He4)*y(Ca40) * rr(1).rates(ircaap)*rr(index_rate).rates(irv1);
       a(11) =  y(Ti44) * rr(1).rates(irtigp) * rr(index_rate).rates(irv1);
       a(12) =  y(Ti44) * rr(index_rate).rates(irtigp) * rr(1).rates(irv1);

       dydt(He4) =  dydt(He4) + esum12(a);

       a(1)  = -y(He4)*y(Ti44) * rr(index_rate).rates(irtiap)*(1.0e0_rt - rr(1).rates(irw1));
       a(2)  =  y(He4)*y(Ti44) * rr(1).rates(irtiap)*rr(index_rate).rates(irw1);
       a(3)  =  y(Cr48) * rr(1).rates(ircrgp) * rr(index_rate).rates(irw1);
       a(4)  =  y(Cr48) * rr(index_rate).rates(ircrgp) * rr(1).rates(irw1);
       a(5)  = -y(He4)*y(Cr48) * rr(index_rate).rates(ircrap)*(1.0e0_rt - rr(1).rates(irx1));
       a(6)  =  y(He4)*y(Cr48) * rr(1).rates(ircrap)*rr(index_rate).rates(irx1);
       a(7)  =  y(Fe52) * rr(1).rates(irfegp) * rr(index_rate).rates(irx1);
       a(8)  =  y(Fe52) * rr(index_rate).rates(irfegp) * rr(1).rates(irx1);

       dydt(He4) =  dydt(He4) + esum8(a);
    }


    // photodisintegration reactions
    a(1) =  y(Fe54) * y(P) * y(P) * rr(index_rate).rates(ir5f54);
    a(2) = -y(Fe52) * y(He4) * rr(index_rate).rates(ir6f54);
    a(3) = -y(Fe52) * y(He4) * y(P) * rr(index_rate).rates(ir7f54);
    a(4) =  y(Ni56) * y(P) * rr(index_rate).rates(ir8f54);
    a(5) = -y(He4) * rr(index_rate).rates(iralf1);
    a(6) =  y(N)*y(N) * y(P)*y(P) * rr(index_rate).rates(iralf2);
    a(7) =  y(Fe56) * y(P) * y(P) * rr(index_rate).rates(irfe56_aux3);
    a(8) = -y(Fe54) * y(He4) * rr(index_rate).rates(irfe56_aux4);

    dydt(He4) =  dydt(He4) + esum8(a);


    // ppchain
    a(1) = y(He3) * y(He3) * rr(index_rate).rates(ir33);
    a(2) = y(He3) * y(He4) * rr(index_rate).rates(irhe3ag);

    dydt(He4) =  dydt(He4) + (a(1) + a(2));


    // cno cycles
    a(1) = y(O16) * y(H1) * rr(index_rate).rates(iropg);
    a(2) = -y(He4) * y(N14) * rr(index_rate).rates(irnag) * 1.5e0_rt;

    dydt(He4) =  dydt(He4) + (a(1) + a(2));

    if (!deriva) {
       a(1) = y(N14) * y(H1) * rr(index_rate).rates(ifa) * rr(index_rate).rates(irnpg);
       dydt(He4) =  dydt(He4) + a(1);
    } else {
       a(1) = y(N14) * y(H1) * rr(index_rate).rates(ifa) * rr(1).rates(irnpg);
       a(2) = y(N14) * y(H1) * rr(1).rates(ifa) * rr(index_rate).rates(irnpg);
       dydt(He4) =  dydt(He4) + (a(1) + a(2));
    }


    // c12 reactions
    a(1) = -2.0_rt * y(C12) * y(C12) * rr(index_rate).rates(ir1212);
    a(2) = -y(C12) * y(O16) * rr(index_rate).rates(ir1216);
    a(3) =  y(He4) * y(He4) * y(He4) * rr(index_rate).rates(ir3a);
    a(4) = -y(C12) * rr(index_rate).rates(irg3a);
    a(5) = -y(C12) * y(He4) * rr(index_rate).rates(ircag);
    a(6) =  y(O16) * rr(index_rate).rates(iroga);
    a(7) = -y(C12) * y(H1) * rr(index_rate).rates(ircpg);

    dydt(C12) =  dydt(C12) + esum7(a);

    if (!deriva) {
        a(1) =  y(N14) * y(H1) * rr(index_rate).rates(ifa) * rr(index_rate).rates(irnpg);
        dydt(C12) =  dydt(C12) + a(1);
    } else {
        a(1) =  y(N14) * y(H1) * rr(index_rate).rates(ifa) * rr(1).rates(irnpg);
        a(2) =  y(N14) * y(H1) * rr(1).rates(ifa) * rr(index_rate).rates(irnpg);
        dydt(C12) =  dydt(C12) + (a(1) + a(2));
    }


    // n14 reactions
    a(1) =  y(C12) * y(H1) * rr(index_rate).rates(ircpg);
    a(2) = -y(N14) * y(H1) * rr(index_rate).rates(irnpg);
    a(3) =  y(O16) * y(H1) * rr(index_rate).rates(iropg);
    a(4) = -y(He4) * y(N14) * rr(index_rate).rates(irnag);

    dydt(N14) =  dydt(N14) + esum4(a);


    // o16 reactions
    a(1) = -y(C12) * y(O16) * rr(index_rate).rates(ir1216);
    a(2) = -2.0_rt * y(O16) * y(O16) * rr(index_rate).rates(ir1616);
    a(3) =  y(C12) * y(He4) * rr(index_rate).rates(ircag);
    a(4) = -y(O16) * y(He4) * rr(index_rate).rates(iroag);
    a(5) = -y(O16) * rr(index_rate).rates(iroga);
    a(6) =  y(Ne20) * rr(index_rate).rates(irnega);
    a(7) = -y(O16) * y(H1) * rr(index_rate).rates(iropg);

    dydt(O16) =  dydt(O16) + esum7(a);

    if (!deriva) {
        a(1) =  y(N14) * y(H1) * rr(index_rate).rates(ifg) * rr(index_rate).rates(irnpg);
        dydt(O16) =  dydt(O16) + a(1);
    } else {
        a(1) =  y(N14) * y(H1) * rr(index_rate).rates(ifg) * rr(1).rates(irnpg);
        a(2) =  y(N14) * y(H1) * rr(1).rates(ifg) * rr(index_rate).rates(irnpg);
        dydt(O16) =  dydt(O16) + (a(1) + a(2));
    }


    // ne20 reactions
    a(1) =  y(C12) * y(C12) * rr(index_rate).rates(ir1212);
    a(2) =  y(O16) * y(He4) * rr(index_rate).rates(iroag);
    a(3) = -y(Ne20) * y(He4) * rr(index_rate).rates(irneag);
    a(4) = -y(Ne20) * rr(index_rate).rates(irnega);
    a(5) =  y(Mg24) * rr(index_rate).rates(irmgga);
    a(6) =  y(N14) * y(He4) * rr(index_rate).rates(irnag);

    dydt(Ne20) =  dydt(Ne20) + esum6(a);


    // mg24 reactions
    a(1) =  0.5e0_rt * y(C12) * y(O16) * rr(index_rate).rates(ir1216);
    a(2) =  y(Ne20) * y(He4) * rr(index_rate).rates(irneag);
    a(3) = -y(Mg24) * y(He4) * rr(index_rate).rates(irmgag);
    a(4) = -y(Mg24) * rr(index_rate).rates(irmgga);
    a(5) =  y(Si28) * rr(index_rate).rates(irsiga);

    dydt(Mg24) =  dydt(Mg24) + esum5(a);

    if (!deriva) {
        a(1) = -y(Mg24) * y(He4) * rr(index_rate).rates(irmgap)*(1.0e0_rt-rr(index_rate).rates(irr1));
        a(2) =  y(Si28) * rr(index_rate).rates(irr1) * rr(index_rate).rates(irsigp);

        dydt(Mg24) =  dydt(Mg24) + (a(1) + a(2));

    } else {
       a(1) = -y(Mg24)*y(He4) * rr(index_rate).rates(irmgap)*(1.0e0_rt - rr(1).rates(irr1));
       a(2) =  y(Mg24)*y(He4) * rr(1).rates(irmgap)*rr(index_rate).rates(irr1);
       a(3) =  y(Si28) * rr(1).rates(irr1) * rr(index_rate).rates(irsigp);
       a(4) =  y(Si28) * rr(index_rate).rates(irr1) * rr(1).rates(irsigp);

       dydt(Mg24) =  dydt(Mg24) + esum4(a);
    }


    // si28 reactions
    a(1) =  0.5e0_rt * y(C12) * y(O16) * rr(index_rate).rates(ir1216);
    a(2) =  0.56e0_rt * y(O16) * y(O16) * rr(index_rate).rates(ir1616);
    a(3) =  y(Mg24) * y(He4) * rr(index_rate).rates(irmgag);
    a(4) = -y(Si28) * y(He4) * rr(index_rate).rates(irsiag);
    a(5) = -y(Si28) * rr(index_rate).rates(irsiga);
    a(6) =  y(S32)  * rr(index_rate).rates(irsga);

    dydt(Si28) =  dydt(Si28) + esum6(a);

    if (!deriva) {
        a(1) =  0.34e0_rt*y(O16)*y(O16)*rr(index_rate).rates(irs1)*rr(index_rate).rates(ir1616);
        a(2) =  y(Mg24) * y(He4) * rr(index_rate).rates(irmgap)*(1.0e0_rt-rr(index_rate).rates(irr1));
        a(3) = -y(Si28) * rr(index_rate).rates(irr1) * rr(index_rate).rates(irsigp);
        a(4) = -y(Si28) * y(He4) * rr(index_rate).rates(irsiap)*(1.0e0_rt-rr(index_rate).rates(irs1));
        a(5) =  y(S32)  * rr(index_rate).rates(irs1) * rr(index_rate).rates(irsgp);

        dydt(Si28) =  dydt(Si28) + esum5(a);

    } else {
        a(1)  =  0.34e0_rt * y(O16)*y(O16) * rr(1).rates(irs1)*rr(index_rate).rates(ir1616);
        a(2)  =  0.34e0_rt * y(O16)*y(O16) * rr(index_rate).rates(irs1)*rr(1).rates(ir1616);
        a(3)  =  y(Mg24)*y(He4) * rr(index_rate).rates(irmgap)*(1.0e0_rt - rr(1).rates(irr1));
        a(4)  = -y(Mg24)*y(He4) * rr(1).rates(irmgap)*rr(index_rate).rates(irr1);
        a(5)  = -y(Si28) * rr(1).rates(irr1) * rr(index_rate).rates(irsigp);
        a(6)  = -y(Si28) * rr(index_rate).rates(irr1) * rr(1).rates(irsigp);
        a(7)  = -y(Si28)*y(He4) * rr(index_rate).rates(irsiap)*(1.0e0_rt - rr(1).rates(irs1));
        a(8)  =  y(Si28)*y(He4) * rr(1).rates(irsiap)*rr(index_rate).rates(irs1);
        a(9)  = y(S32) * rr(1).rates(irs1) * rr(index_rate).rates(irsgp);
        a(10) = y(S32) * rr(index_rate).rates(irs1) * rr(1).rates(irsgp);

        dydt(Si28) =  dydt(Si28) + esum10(a);
    }


    // s32 reactions
    a(1) =  0.1e0_rt * y(O16) * y(O16) * rr(index_rate).rates(ir1616);
    a(2) =  y(Si28) * y(He4) * rr(index_rate).rates(irsiag);
    a(3) = -y(S32) * y(He4) * rr(index_rate).rates(irsag);
    a(4) = -y(S32) * rr(index_rate).rates(irsga);
    a(5) =  y(Ar36) * rr(index_rate).rates(irarga);

    dydt(S32) =  dydt(S32) + esum5(a);

    if (!deriva) {
        a(1) =  0.34e0_rt * y(O16)*y(O16)* rr(index_rate).rates(ir1616)*(1.0e0_rt-rr(index_rate).rates(irs1));
        a(2) =  y(Si28) * y(He4) * rr(index_rate).rates(irsiap)*(1.0e0_rt-rr(index_rate).rates(irs1));
        a(3) = -y(S32) * rr(index_rate).rates(irs1) * rr(index_rate).rates(irsgp);
        a(4) = -y(S32) * y(He4) * rr(index_rate).rates(irsap)*(1.0e0_rt-rr(index_rate).rates(irt1));
        a(5) =  y(Ar36) * rr(index_rate).rates(irt1) * rr(index_rate).rates(irargp);

        dydt(S32) =  dydt(S32) + esum5(a);

    } else {
        a(1)  =  0.34e0_rt * y(O16)*y(O16) * rr(index_rate).rates(ir1616)*(1.0e0_rt-rr(1).rates(irs1));
        a(2)  = -0.34e0_rt * y(O16)*y(O16) * rr(1).rates(ir1616)*rr(index_rate).rates(irs1);
        a(3)  =  y(Si28)*y(He4) * rr(index_rate).rates(irsiap)*(1.0e0_rt-rr(1).rates(irs1));
        a(4)  = -y(Si28)*y(He4) * rr(1).rates(irsiap)*rr(index_rate).rates(irs1);
        a(5)  = -y(S32) * rr(1).rates(irs1) * rr(index_rate).rates(irsgp);
        a(6)  = -y(S32) * rr(index_rate).rates(irs1) * rr(1).rates(irsgp);
        a(7)  = -y(S32)*y(He4) * rr(index_rate).rates(irsap)*(1.0e0_rt-rr(1).rates(irt1));
        a(8)  =  y(S32)*y(He4) * rr(1).rates(irsap)*rr(index_rate).rates(irt1);
        a(9)  =  y(Ar36) * rr(1).rates(irt1) * rr(index_rate).rates(irargp);
        a(10) =  y(Ar36) * rr(index_rate).rates(irt1) * rr(1).rates(irargp);

        dydt(S32) =  dydt(S32) + esum10(a);
    }


    // ar36 reactions
    a(1) =  y(S32)  * y(He4) * rr(index_rate).rates(irsag);
    a(2) = -y(Ar36) * y(He4) * rr(index_rate).rates(irarag);
    a(3) = -y(Ar36) * rr(index_rate).rates(irarga);
    a(4) =  y(Ca40) * rr(index_rate).rates(ircaga);

    dydt(Ar36) =  dydt(Ar36) + esum4(a);

    if (!deriva) {
        a(1) = y(S32)  * y(He4) * rr(index_rate).rates(irsap)*(1.0e0_rt-rr(index_rate).rates(irt1));
        a(2) = -y(Ar36) * rr(index_rate).rates(irt1) * rr(index_rate).rates(irargp);
        a(3) = -y(Ar36) * y(He4) * rr(index_rate).rates(irarap)*(1.0e0_rt-rr(index_rate).rates(iru1));
        a(4) =  y(Ca40) * rr(index_rate).rates(ircagp) * rr(index_rate).rates(iru1);

        dydt(Ar36) =  dydt(Ar36) + esum4(a);

    } else {
        a(1) =  y(S32)*y(He4) * rr(index_rate).rates(irsap)*(1.0e0_rt - rr(1).rates(irt1));
        a(2) = -y(S32)*y(He4) * rr(1).rates(irsap)*rr(index_rate).rates(irt1);
        a(3) = -y(Ar36) * rr(1).rates(irt1) * rr(index_rate).rates(irargp);
        a(4) = -y(Ar36) * rr(index_rate).rates(irt1) * rr(1).rates(irargp);
        a(5) = -y(Ar36)*y(He4) * rr(index_rate).rates(irarap)*(1.0e0_rt-rr(1).rates(iru1));
        a(6) =  y(Ar36)*y(He4) * rr(1).rates(irarap)*rr(index_rate).rates(iru1);
        a(7) =  y(Ca40) * rr(1).rates(ircagp) * rr(index_rate).rates(iru1);
        a(8) =  y(Ca40) * rr(index_rate).rates(ircagp) * rr(1).rates(iru1);

        dydt(Ar36) =  dydt(Ar36) + esum8(a);
    }


    // ca40 reactions
    a(1) =  y(Ar36) * y(He4) * rr(index_rate).rates(irarag);
    a(2) = -y(Ca40) * y(He4) * rr(index_rate).rates(ircaag);
    a(3) = -y(Ca40) * rr(index_rate).rates(ircaga);
    a(4) =  y(Ti44) * rr(index_rate).rates(irtiga);

    dydt(Ca40) =  dydt(Ca40) + esum4(a);

    if (!deriva) {
        a(1) =  y(Ar36) * y(He4) * rr(index_rate).rates(irarap)*(1.0e0_rt-rr(index_rate).rates(iru1));
        a(2) = -y(Ca40) * rr(index_rate).rates(ircagp) * rr(index_rate).rates(iru1);
        a(3) = -y(Ca40) * y(He4) * rr(index_rate).rates(ircaap)*(1.0e0_rt-rr(index_rate).rates(irv1));
        a(4) =  y(Ti44) * rr(index_rate).rates(irtigp) * rr(index_rate).rates(irv1);

        dydt(Ca40) =  dydt(Ca40) + esum4(a);

    } else {
        a(1) =  y(Ar36)*y(He4) * rr(index_rate).rates(irarap)*(1.0e0_rt-rr(1).rates(iru1));
        a(2) = -y(Ar36)*y(He4) * rr(1).rates(irarap)*rr(index_rate).rates(iru1);
        a(3) = -y(Ca40) * rr(1).rates(ircagp) * rr(index_rate).rates(iru1);
        a(4) = -y(Ca40) * rr(index_rate).rates(ircagp) * rr(1).rates(iru1);
        a(5) = -y(Ca40)*y(He4) * rr(index_rate).rates(ircaap)*(1.0e0_rt-rr(1).rates(irv1));
        a(6) =  y(Ca40)*y(He4) * rr(1).rates(ircaap)*rr(index_rate).rates(irv1);
        a(7) =  y(Ti44) * rr(1).rates(irtigp) * rr(index_rate).rates(irv1);
        a(8) =  y(Ti44) * rr(index_rate).rates(irtigp) * rr(1).rates(irv1);

        dydt(Ca40) =  dydt(Ca40) + esum8(a);
    }


    // ti44 reactions
    a(1) =  y(Ca40) * y(He4) * rr(index_rate).rates(ircaag);
    a(2) = -y(Ti44) * y(He4) * rr(index_rate).rates(irtiag);
    a(3) = -y(Ti44) * rr(index_rate).rates(irtiga);
    a(4) =  y(Cr48) * rr(index_rate).rates(ircrga);

    dydt(Ti44) =  dydt(Ti44) + esum4(a);

    if (!deriva) {
        a(1) =  y(Ca40) * y(He4) * rr(index_rate).rates(ircaap)*(1.0e0_rt-rr(index_rate).rates(irv1));
        a(2) = -y(Ti44) * rr(index_rate).rates(irv1) * rr(index_rate).rates(irtigp);
        a(3) = -y(Ti44) * y(He4) * rr(index_rate).rates(irtiap)*(1.0e0_rt-rr(index_rate).rates(irw1));
        a(4) =  y(Cr48) * rr(index_rate).rates(irw1) * rr(index_rate).rates(ircrgp);

        dydt(Ti44) =  dydt(Ti44) + esum4(a);

    } else {
        a(1) =  y(Ca40)*y(He4) * rr(index_rate).rates(ircaap)*(1.0e0_rt-rr(1).rates(irv1));
        a(2) = -y(Ca40)*y(He4) * rr(1).rates(ircaap)*rr(index_rate).rates(irv1);
        a(3) = -y(Ti44) * rr(1).rates(irv1) * rr(index_rate).rates(irtigp);
        a(4) = -y(Ti44) * rr(index_rate).rates(irv1) * rr(1).rates(irtigp);
        a(5) = -y(Ti44)*y(He4) * rr(index_rate).rates(irtiap)*(1.0e0_rt-rr(1).rates(irw1));
        a(6) =  y(Ti44)*y(He4) * rr(1).rates(irtiap)*rr(index_rate).rates(irw1);
        a(7) =  y(Cr48) * rr(1).rates(irw1) * rr(index_rate).rates(ircrgp);
        a(8) =  y(Cr48) * rr(index_rate).rates(irw1) * rr(1).rates(ircrgp);

       dydt(Ti44) =  dydt(Ti44) + esum8(a);
    }


    // cr48 reactions
    a(1) =  y(Ti44) * y(He4) * rr(index_rate).rates(irtiag);
    a(2) = -y(Cr48) * y(He4) * rr(index_rate).rates(ircrag);
    a(3) = -y(Cr48) * rr(index_rate).rates(ircrga);
    a(4) =  y(Fe52) * rr(index_rate).rates(irfega);

    dydt(Cr48) =  dydt(Cr48) + esum4(a);

    if (!deriva) {
        a(1) =  y(Ti44) * y(He4) * rr(index_rate).rates(irtiap)*(1.0e0_rt-rr(index_rate).rates(irw1));
        a(2) = -y(Cr48) * rr(index_rate).rates(irw1) * rr(index_rate).rates(ircrgp);
        a(3) = -y(Cr48) * y(He4) * rr(index_rate).rates(ircrap)*(1.0e0_rt-rr(index_rate).rates(irx1));
        a(4) =  y(Fe52) * rr(index_rate).rates(irx1) * rr(index_rate).rates(irfegp);

        dydt(Cr48) =  dydt(Cr48) + esum4(a);

    } else {
        a(1) =  y(Ti44)*y(He4) * rr(index_rate).rates(irtiap)*(1.0e0_rt-rr(1).rates(irw1));
        a(2) = -y(Ti44)*y(He4) * rr(1).rates(irtiap)*rr(index_rate).rates(irw1);
        a(3) = -y(Cr48) * rr(1).rates(irw1) * rr(index_rate).rates(ircrgp);
        a(4) = -y(Cr48) * rr(index_rate).rates(irw1) * rr(1).rates(ircrgp);
        a(5) = -y(Cr48)*y(He4) * rr(index_rate).rates(ircrap)*(1.0e0_rt-rr(1).rates(irx1));
        a(6) =  y(Cr48)*y(He4) * rr(1).rates(ircrap)*rr(index_rate).rates(irx1);
        a(7) =  y(Fe52) * rr(1).rates(irx1) * rr(index_rate).rates(irfegp);
        a(8) =  y(Fe52) * rr(index_rate).rates(irx1) * rr(1).rates(irfegp);

        dydt(Cr48) =  dydt(Cr48) + esum8(a);
    }

    // cr56 reactions
    a(1)  = y(Fe56) * 1.0e-04_rt * rr(index_rate).rates(irn56ec);
    dydt(Cr56) =  dydt(Cr56) + a(1);


    // fe52 reactions
    a(1) =  y(Cr48) * y(He4) * rr(index_rate).rates(ircrag);
    a(2) = -y(Fe52) * y(He4) * rr(index_rate).rates(irfeag);
    a(3) = -y(Fe52) * rr(index_rate).rates(irfega);
    a(4) =  y(Ni56) * rr(index_rate).rates(irniga);

    dydt(Fe52) =  dydt(Fe52) + esum4(a);

    if (!deriva) {
        a(1) =  y(Cr48) * y(He4) * rr(index_rate).rates(ircrap)*(1.0e0_rt-rr(index_rate).rates(irx1));
        a(2) = -y(Fe52) * rr(index_rate).rates(irx1) * rr(index_rate).rates(irfegp);
        dydt(Fe52) =  dydt(Fe52) + (a(1) + a(2));

    } else {
        a(1) =  y(Cr48)*y(He4) * rr(index_rate).rates(ircrap)*(1.0e0_rt-rr(1).rates(irx1));
        a(2) = -y(Cr48)*y(He4) * rr(1).rates(ircrap)*rr(index_rate).rates(irx1);
        a(3) = -y(Fe52) * rr(1).rates(irx1) * rr(index_rate).rates(irfegp);
        a(4) = -y(Fe52) * rr(index_rate).rates(irx1) * rr(1).rates(irfegp);
        dydt(Fe52) =  dydt(Fe52) + esum4(a);
    }


    // photodisintegration reactions
    a(1) =  y(Fe54) * rr(index_rate).rates(ir1f54);
    a(2) = -y(Fe52) * y(N) * y(N) * rr(index_rate).rates(ir2f54);
    a(3) =  y(Fe54) * y(P) * y(P) * rr(index_rate).rates(ir5f54);
    a(4) = -y(Fe52) * y(He4) * rr(index_rate).rates(ir6f54);
    a(5) = -y(Fe52) * y(He4) * y(P) * rr(index_rate).rates(ir7f54);
    a(6) =  y(Ni56) * y(P) * rr(index_rate).rates(ir8f54);

    dydt(Fe52) =  dydt(Fe52) + esum6(a);


    // fe54 reactions
    a(1)  = -y(Fe54) * rr(index_rate).rates(ir1f54);
    a(2)  =  y(Fe52) * y(N) * y(N) * rr(index_rate).rates(ir2f54);
    a(3)  = -y(Fe54) * y(P) * y(P) * rr(index_rate).rates(ir3f54);
    a(4)  =  y(Ni56) * rr(index_rate).rates(ir4f54);
    a(5)  = -y(Fe54) * y(P) * y(P) * rr(index_rate).rates(ir5f54);
    a(6)  =  y(Fe52) * y(He4) * rr(index_rate).rates(ir6f54);
    a(7)  =  y(Fe56) * rr(index_rate).rates(irfe56_aux1);
    a(8)  = -y(Fe54) * y(N) * y(N) * rr(index_rate).rates(irfe56_aux2);
    a(9)  =  y(Fe56) * y(P) * y(P) * rr(index_rate).rates(irfe56_aux3);
    a(10) = -y(Fe54) * y(He4) * rr(index_rate).rates(irfe56_aux4);

    dydt(Fe54) =  dydt(Fe54) + esum10(a);

    // fe56 reactions
    a(1) =  y(Ni56) * rr(index_rate).rates(irn56ec);
    a(2) = -y(Fe56) * 1.0e-04_rt * rr(index_rate).rates(irn56ec);
    a(3) = -y(Fe56) * rr(index_rate).rates(irfe56_aux1);
    a(4) =  y(Fe54) * y(N) * y(N) * rr(index_rate).rates(irfe56_aux2);
    a(5) = -y(Fe56) * y(P) * y(P) * rr(index_rate).rates(irfe56_aux3);
    a(6) =  y(Fe54) * y(He4) * rr(index_rate).rates(irfe56_aux4);

    dydt(Fe56) =  dydt(Fe56) + esum6(a);

    // ni56 reactions
    a(1) =  y(Fe52) * y(He4) * rr(index_rate).rates(irfeag);
    a(2) = -y(Ni56) * rr(index_rate).rates(irniga);
    a(3) = -y(Ni56) * rr(index_rate).rates(irn56ec);

    dydt(Ni56) =  dydt(Ni56) + esum3(a);


    // photodisintegration reactions
    a(1) =  y(Fe54) * y(P) * y(P) * rr(index_rate).rates(ir3f54);
    a(2) = -y(Ni56) * rr(index_rate).rates(ir4f54);
    a(3) =  y(Fe52) * y(He4)* y(P) * rr(index_rate).rates(ir7f54);
    a(4) = -y(Ni56) * y(P) * rr(index_rate).rates(ir8f54);

    dydt(Ni56) =  dydt(Ni56) + esum4(a);


    // photodisintegration neutrons
    a(1) =  2.0e0_rt * y(Fe54) * rr(index_rate).rates(ir1f54);
    a(2) = -2.0e0_rt * y(Fe52) * y(N) * y(N) * rr(index_rate).rates(ir2f54);
    a(3) =  2.0e0_rt * y(He4) * rr(index_rate).rates(iralf1);
    a(4) = -2.0e0_rt * y(N)*y(N) * y(P)*y(P) * rr(index_rate).rates(iralf2);
    a(5) =  y(P) * rr(index_rate).rates(irpen);
    a(6) = -y(N) * rr(index_rate).rates(irnep);
    a(7) =  2.0e0_rt * y(Fe56) * rr(index_rate).rates(irfe56_aux1);
    a(8) = -2.0e0_rt * y(Fe54) * y(N) * y(N) * rr(index_rate).rates(irfe56_aux2);

    dydt(N) =  dydt(N) + esum8(a);


    // photodisintegration protons
    a(1)  = -2.0e0_rt * y(Fe54) * y(P) * y(P) * rr(index_rate).rates(ir3f54);
    a(2)  =  2.0e0_rt * y(Ni56) * rr(index_rate).rates(ir4f54);
    a(3)  = -2.0e0_rt * y(Fe54) * y(P) * y(P) * rr(index_rate).rates(ir5f54);
    a(4)  =  2.0e0_rt * y(Fe52) * y(He4) * rr(index_rate).rates(ir6f54);
    a(5)  =  2.0e0_rt * y(He4) * rr(index_rate).rates(iralf1);
    a(6)  = -2.0e0_rt * y(N)*y(N) * y(P)*y(P) * rr(index_rate).rates(iralf2);
    a(7)  = -y(P) * rr(index_rate).rates(irpen);
    a(8)  =  y(N) * rr(index_rate).rates(irnep);
    a(9)  = -2.0e0_rt * y(Fe56) * y(P) * y(P) * rr(index_rate).rates(irfe56_aux3);
    a(10) =  2.0e0_rt * y(Fe54) * y(He4) * rr(index_rate).rates(irfe56_aux4);

    dydt(P) =  dydt(P) + esum10(a);
}


template<class MatrixType>
AMREX_GPU_HOST_DEVICE AMREX_INLINE
void dfdy_isotopes_aprox21 (Array1D<Real, 1, NumSpec> const& y,
                            const Array1D<rate_t, 1, Rates::NumGroups>& rr,
                            MatrixType& dfdy)
{
    using namespace Species;

    // this routine sets up the dense aprox21 jacobian for the isotopes

    Array1D<Real, 1, 30> b;

    // h1 jacobian elements
    // d(h1)/d(h1)
    b(1) = -6.0e0_rt * y(H1)  * rr(1).rates(irpp);
    b(2) = -2.0e0_rt * y(C12) * rr(1).rates(ircpg);
    b(3) = -2.0e0_rt * y(N14) * rr(1).rates(irnpg);
    b(4) = -2.0e0_rt * y(N14) * y(H1) * rr(3).rates(irnpg);
    b(5) = -2.0e0_rt * y(O16) * rr(1).rates(iropg);
    b(6) = -2.0e0_rt * y(O16) * y(H1) * rr(3).rates(iropg);
    b(7) = -3.0e0_rt * rr(1).rates(irpen);
    dfdy(H1, H1) = esum7(b);


    // d(h1)/d(he3)
    b(1) = 4.0e0_rt * y(He3) * rr(1).rates(ir33);
    b(2) = -y(He4) * rr(1).rates(irhe3ag);
    dfdy(H1, He3) = b(1) + b(2);

    // d(h1)/d(he4)
    b(1) = -y(He3) * rr(1).rates(irhe3ag);
    b(2) = -y(He3) * y(He4) * rr(3).rates(irhe3ag);
    dfdy(H1, He4) = b(1) + b(2);

    // d(h1)/d(c12)
    b(1) = -2.0e0_rt * y(H1) * rr(1).rates(ircpg);
    dfdy(H1, C12) = b(1);

    // d(h1)/d(n14)
    b(1) = -2.0e0_rt * y(H1) * rr(1).rates(irnpg);
    dfdy(H1, N14) = b(1);

    // d(h1)/d(o16)
    b(1) = -2.0e0_rt * y(H1) * rr(1).rates(iropg);
    dfdy(H1, O16) = b(1);


    // he3 jacobian elements
    // d(he3)/d(h1)
    b(1) = 2.0_rt * y(H1) * rr(1).rates(irpp);
    b(2) = rr(1).rates(irpen);
    dfdy(He3, H1) = b(1) + b(2);

    // d(he3)/d(he3)
    b(1) = -4.0e0_rt * y(He3) * rr(1).rates(ir33);
    b(2) = -y(He4) * rr(1).rates(irhe3ag);
    dfdy(He3, He3) = b(1) + b(2);

    // d(he3)/d(he4)
    b(1) = -y(He3) * rr(1).rates(irhe3ag);
    b(2) = -y(He3) * y(He4) * rr(3).rates(irhe3ag);
    dfdy(He3, He4) = b(1) + b(2);


    // he4 jacobian elements
    // d(he4)/d(h1)
    b(1) =  y(N14) * rr(1).rates(ifa) * rr(1).rates(irnpg);
    b(2) =  y(N14) * y(H1) * rr(1).rates(ifa) * rr(3).rates(irnpg);
    b(3) =  y(O16) * rr(1).rates(iropg);
    b(4) =  y(O16) * y(H1) * rr(3).rates(iropg);
    dfdy(He4, H1) = esum4(b);


    // d(he4)/d(he3)
    b(1) = 2.0_rt * y(He3) * rr(1).rates(ir33);
    b(2) = y(He4) * rr(1).rates(irhe3ag);
    dfdy(He4, He3) = b(1) + b(2);

    // d(he4)/d(he4)
    b(1)  = -9.0e0_rt * y(He4) * y(He4) * rr(1).rates(ir3a);
    b(2)  = -y(C12)  * rr(1).rates(ircag);
    b(3)  = -y(O16)  * rr(1).rates(iroag);
    b(4)  = -y(Ne20) * rr(1).rates(irneag);
    b(5)  = -y(Mg24) * rr(1).rates(irmgag);
    b(6)  = -y(Si28) * rr(1).rates(irsiag);
    b(7)  = -y(S32)  * rr(1).rates(irsag);
    b(8)  = -y(Ar36) * rr(1).rates(irarag);
    b(9)  = -y(Ca40) * rr(1).rates(ircaag);
    b(10) = -y(Ti44) * rr(1).rates(irtiag);
    b(11) = -y(Cr48) * rr(1).rates(ircrag);
    b(12) = -y(Fe52) * rr(1).rates(irfeag);
    b(13) = -y(Mg24) * rr(1).rates(irmgap) * (1.0e0_rt-rr(1).rates(irr1));
    b(14) = -y(Si28) * rr(1).rates(irsiap) * (1.0e0_rt-rr(1).rates(irs1));
    b(15) = -y(S32)  * rr(1).rates(irsap)  * (1.0e0_rt-rr(1).rates(irt1));
    b(16) = -y(Ar36) * rr(1).rates(irarap) * (1.0e0_rt-rr(1).rates(iru1));
    b(17) = -y(Ca40) * rr(1).rates(ircaap) * (1.0e0_rt-rr(1).rates(irv1));
    b(18) = -y(Ti44) * rr(1).rates(irtiap) * (1.0e0_rt-rr(1).rates(irw1));
    b(19) = -y(Cr48) * rr(1).rates(ircrap) * (1.0e0_rt-rr(1).rates(irx1));
    b(20) = -y(Fe52) * rr(1).rates(ir6f54);
    b(21) = -y(Fe52) * y(P) * rr(1).rates(ir7f54);
    b(22) = -rr(1).rates(iralf1);
    b(23) = -y(Fe54) * rr(1).rates(irfe56_aux4);
    b(24) =  y(He3) * rr(1).rates(irhe3ag);
    b(25) =  y(He3) * y(He4) * rr(3).rates(irhe3ag);
    b(26) = -y(N14) * rr(1).rates(irnag) * 1.5e0_rt;
    dfdy(He4, He4) = esum26(b);

    // d(he4)/d(c12)
    b(1) =  2.0_rt * y(C12) * rr(1).rates(ir1212);
    b(2) =  0.5e0_rt * y(O16) * rr(1).rates(ir1216);
    b(3) =  3.0e0_rt * rr(1).rates(irg3a);
    b(4) = -y(He4) * rr(1).rates(ircag);
    dfdy(He4, C12) = esum4(b);

    // d(he4)/d(n14)
    b(1) =  y(H1) * rr(1).rates(ifa) * rr(1).rates(irnpg);
    b(2) = -y(He4) * rr(1).rates(irnag) * 1.5e0_rt;
    dfdy(He4, N14) = b(1) + b(2);

    // d(he4)/d(o16)
    b(1) =  0.5e0_rt * y(C12) * rr(1).rates(ir1216);
    b(2) =  1.12e0_rt * y(O16) * rr(1).rates(ir1616);
    b(3) =  0.68e0_rt * rr(1).rates(irs1) * y(O16) * rr(1).rates(ir1616);
    b(4) =  rr(1).rates(iroga);
    b(5) = -y(He4) * rr(1).rates(iroag);
    b(6) =  y(H1) * rr(1).rates(iropg);
    dfdy(He4, O16) = esum6(b);

    // d(he4)/d(ne20)
    b(1) =  rr(1).rates(irnega);
    b(2) = -y(He4) * rr(1).rates(irneag);
    dfdy(He4, Ne20) = b(1) + b(2);

    // d(he4)/d(mg24)
    b(1) =  rr(1).rates(irmgga);
    b(2) = -y(He4) * rr(1).rates(irmgag);
    b(3) = -y(He4) * rr(1).rates(irmgap) * (1.0e0_rt-rr(1).rates(irr1));
    dfdy(He4, Mg24) = esum3(b);

    // d(he4)/d(si28)
    b(1) =  rr(1).rates(irsiga);
    b(2) = -y(He4) * rr(1).rates(irsiag);
    b(3) = -y(He4) * rr(1).rates(irsiap) * (1.0e0_rt-rr(1).rates(irs1));
    b(4) =  rr(1).rates(irr1) * rr(1).rates(irsigp);
    dfdy(He4, Si28) = esum4(b);

    // d(he4)/d(s32)
    b(1) =  rr(1).rates(irsga);
    b(2) = -y(He4) * rr(1).rates(irsag);
    b(3) = -y(He4) * rr(1).rates(irsap) * (1.0e0_rt-rr(1).rates(irt1));
    b(4) =  rr(1).rates(irs1) * rr(1).rates(irsgp);
    dfdy(He4, S32) = esum4(b);

    // d(he4)/d(ar36)
    b(1)  =  rr(1).rates(irarga);
    b(2)  = -y(He4) * rr(1).rates(irarag);
    b(3)  = -y(He4) * rr(1).rates(irarap) * (1.0e0_rt-rr(1).rates(iru1));
    b(4)  =  rr(1).rates(irt1) * rr(1).rates(irargp);
    dfdy(He4, Ar36) = esum4(b);

    // d(he4)/d(ca40)
    b(1) =  rr(1).rates(ircaga);
    b(2) = -y(He4) * rr(1).rates(ircaag);
    b(3) = -y(He4) * rr(1).rates(ircaap) * (1.0e0_rt-rr(1).rates(irv1));
    b(4) =  rr(1).rates(iru1) * rr(1).rates(ircagp);
    dfdy(He4, Ca40) = esum4(b);

    // d(he4)/d(ti44)
    b(1) =  rr(1).rates(irtiga);
    b(2) = -y(He4) * rr(1).rates(irtiag);
    b(3) = -y(He4) * rr(1).rates(irtiap) * (1.0e0_rt-rr(1).rates(irw1));
    b(4) =  rr(1).rates(irv1) * rr(1).rates(irtigp);
    dfdy(He4, Ti44) = esum4(b);

    // d(he4)/d(cr48)
    b(1) =  rr(1).rates(ircrga);
    b(2) = -y(He4) * rr(1).rates(ircrag);
    b(3) = -y(He4) * rr(1).rates(ircrap) * (1.0e0_rt-rr(1).rates(irx1));
    b(4) =  rr(1).rates(irw1) * rr(1).rates(ircrgp);
    dfdy(He4, Cr48) = esum4(b);

    // d(he4)/d(fe52)
    b(1) =  rr(1).rates(irfega);
    b(2) = -y(He4) * rr(1).rates(irfeag);
    b(3) =  rr(1).rates(irx1) * rr(1).rates(irfegp);
    b(4) = -y(He4) * rr(1).rates(ir6f54);
    b(5) = -y(He4) * y(P) * rr(1).rates(ir7f54);
    dfdy(He4, Fe52) = esum5(b);

    // d(he4)/d(fe54)
    b(1) =  y(P) * y(P) * rr(1).rates(ir5f54);
    b(2) = -y(He4) * rr(1).rates(irfe56_aux4);
    dfdy(He4, Fe54) = b(1) + b(2);

    // d(he4)/d(fe56)
    b(1) = y(P) * y(P) * rr(1).rates(irfe56_aux3);
    dfdy(He4, Fe56) = b(1);

    // d(he4)/d(ni56)
    b(1) = rr(1).rates(irniga);
    b(2) = y(P) * rr(1).rates(ir8f54);
    dfdy(He4, Ni56) = b(1) + b(2);

    // d(he4)/d(neut)
    b(1) = -y(He4) * rr(3).rates(iralf1);
    b(2) =  2.0e0_rt * y(N) * y(P)*y(P) * rr(1).rates(iralf2);
    b(3) =  y(N)*y(N) * y(P)*y(P) * rr(3).rates(iralf2);
    dfdy(He4, N) = esum3(b);

    // d(he4)/d(prot)
    b(1)  =  2.0e0_rt * y(Fe54) * y(P) * rr(1).rates(ir5f54);
    b(2)  =  y(Fe54) * y(P) * y(P) * rr(3).rates(ir5f54);
    b(3)  = -y(He4) * y(Fe52) * rr(3).rates(ir6f54);
    b(4)  = -y(Fe52) * y(He4) * rr(1).rates(ir7f54);
    b(5)  = -y(Fe52) * y(He4) * y(P) * rr(3).rates(ir7f54);
    b(6)  =  y(Ni56) * rr(1).rates(ir8f54);
    b(7)  =  y(Ni56) * y(P) * rr(3).rates(ir8f54);
    b(8)  = -y(He4) * rr(4).rates(iralf1);
    b(9)  =  2.0e0_rt * y(N)*y(N) * y(P) * rr(1).rates(iralf2);
    b(10) =  y(N)*y(N) * y(P)*y(P) * rr(4).rates(iralf2);
    b(11) =  2.0e0_rt * y(Fe56) * y(P) * rr(1).rates(irfe56_aux3);
    b(12) =  y(Fe56) * y(P) * y(P) * rr(3).rates(irfe56_aux3);
    b(13) = -y(He4) * y(Fe54) * rr(3).rates(irfe56_aux4);
    dfdy(He4, P) = esum13(b);



    // c12 jacobian elements
    // d(c12)/d(h1)
    b(1) = -y(C12) * rr(1).rates(ircpg);
    b(2) =  y(N14) * rr(1).rates(ifa) * rr(1).rates(irnpg);
    b(3) =  y(N14) * y(H1) * rr(1).rates(ifa) * rr(3).rates(irnpg);
    dfdy(C12, H1) = esum3(b);

    // d(c12)/d(he4)
    b(1) =  3.0e0_rt * y(He4) * y(He4) * rr(1).rates(ir3a);
    b(2) = -y(C12) * rr(1).rates(ircag);
    dfdy(C12, He4) = b(1) + b(2);

    // d(c12)/d(c12)
    b(1) = -4.0e0_rt * y(C12) * rr(1).rates(ir1212);
    b(2) = -y(O16) * rr(1).rates(ir1216);
    b(3) = -rr(1).rates(irg3a);
    b(4) = -y(He4) * rr(1).rates(ircag);
    b(5) = -y(H1) * rr(1).rates(ircpg);
    dfdy(C12, C12) = esum5(b);

    // d(c12)/d(n14)
    b(1) = y(H1) * rr(1).rates(ifa) * rr(1).rates(irnpg);
    dfdy(C12, N14) = b(1);

    // d(c12)/d(o16)
    b(1) = -y(C12) * rr(1).rates(ir1216);
    b(2) =  rr(1).rates(iroga);
    dfdy(C12, O16) = b(1) + b(2);


    // n14 jacobian elements
    // d(n14)/d(h1)
    b(1) =   y(C12) * rr(1).rates(ircpg);
    b(2) =  -y(N14) * rr(1).rates(irnpg);
    b(3) =  -y(N14) * y(H1) * rr(3).rates(irnpg);
    b(4) =   y(O16) * rr(1).rates(iropg);
    b(5) =   y(O16) * y(H1) * rr(3).rates(iropg);
    dfdy(N14, H1) = esum5(b);

    // d(n14)/d(he4)
    b(1) = -y(N14) * rr(1).rates(irnag);
    dfdy(N14, He4) = b(1);

    // d(n14)/d(c12)
    b(1) =  y(H1) * rr(1).rates(ircpg);
    dfdy(N14, C12) = b(1);

    // d(n14)/d(n14)
    b(1) = -y(H1) * rr(1).rates(irnpg);
    b(2) = -y(He4) * rr(1).rates(irnag);
    dfdy(N14, N14) = b(1) + b(2);

    // d(n14)/d(o16)
    b(1) = y(H1) * rr(1).rates(iropg);
    dfdy(N14, O16) = b(1);


    // o16 jacobian elements
    // d(o16)/d(h1)
    b(1) =  y(N14) * rr(1).rates(ifg) * rr(1).rates(irnpg);
    b(2) =  y(N14) * y(H1) * rr(1).rates(ifg) * rr(3).rates(irnpg);
    b(3) = -y(O16) * rr(1).rates(iropg);
    b(4) = -y(O16) * y(H1) * rr(3).rates(iropg);
    dfdy(O16, H1) = esum4(b);

    // d(o16)/d(he4)
    b(1) =  y(C12)*rr(1).rates(ircag);
    b(2) = -y(O16)*rr(1).rates(iroag);
    dfdy(O16, He4) = b(1) + b(2);

    // d(o16)/d(c12)
    b(1) = -y(O16)*rr(1).rates(ir1216);
    b(2) =  y(He4)*rr(1).rates(ircag);
    dfdy(O16, C12) = b(1) + b(2);

    // d(o16)/d(n14)
    b(1) = y(H1) * rr(1).rates(ifg) * rr(1).rates(irnpg);
    dfdy(O16, N14) = b(1);

    // d(o16)/d(o16)
    b(1) = -y(C12) * rr(1).rates(ir1216);
    b(2) = -4.0e0_rt * y(O16) * rr(1).rates(ir1616);
    b(3) = -y(He4) * rr(1).rates(iroag);
    b(4) = -rr(1).rates(iroga);
    b(5) = -y(H1) * rr(1).rates(iropg);
    dfdy(O16, O16) = esum5(b);

    // d(o16)/d(ne20)
    b(1) = rr(1).rates(irnega);
    dfdy(O16, Ne20) = b(1);



    // ne20 jacobian elements
    // d(ne20)/d(he4)
    b(1) =  y(O16) * rr(1).rates(iroag);
    b(2) = -y(Ne20) * rr(1).rates(irneag);
    b(3) =  y(N14) * rr(1).rates(irnag);
    dfdy(Ne20, He4) = esum3(b);

    // d(ne20)/d(c12)
    b(1) = 2.0_rt * y(C12) * rr(1).rates(ir1212);
    dfdy(Ne20, C12) = b(1);

    // d(ne20)/d(n14)
    b(1) = y(He4) * rr(1).rates(irnag);
    dfdy(Ne20, N14) = b(1);

    // d(ne20)/d(o16)
    b(1) = y(He4) * rr(1).rates(iroag);
    dfdy(Ne20, O16) = b(1);

    // d(ne20)/d(ne20)
    b(1) = -y(He4) * rr(1).rates(irneag);
    b(2) = -rr(1).rates(irnega);
    dfdy(Ne20, Ne20) = b(1) + b(2);

    // d(ne20)/d(mg24)
    b(1) = rr(1).rates(irmgga);
    dfdy(Ne20, Mg24) = b(1);



    // mg24 jacobian elements
    // d(mg24)/d(he4)
    b(1) =  y(Ne20) * rr(1).rates(irneag);
    b(2) = -y(Mg24) * rr(1).rates(irmgag);
    b(3) = -y(Mg24) * rr(1).rates(irmgap) * (1.0e0_rt-rr(1).rates(irr1));
    dfdy(Mg24, He4) = esum3(b);

    // d(mg24)/d(c12)
    b(1) = 0.5e0_rt * y(O16) * rr(1).rates(ir1216);
    dfdy(Mg24, C12) = b(1);

    // d(mg24)/d(o16)
    b(1) = 0.5e0_rt * y(C12) * rr(1).rates(ir1216);
    dfdy(Mg24, O16) = b(1);

    // d(mg24)/d(ne20)
    b(1) = y(He4) * rr(1).rates(irneag);
    dfdy(Mg24, Ne20) = b(1);

    // d(mg24)/d(mg24)
    b(1) =  -y(He4) * rr(1).rates(irmgag);
    b(2) = -rr(1).rates(irmgga);
    b(3) = -y(He4) * rr(1).rates(irmgap) * (1.0e0_rt-rr(1).rates(irr1));
    dfdy(Mg24, Mg24) = esum3(b);

    // d(mg24)/d(si28)
    b(1) = rr(1).rates(irsiga);
    b(2) = rr(1).rates(irr1) * rr(1).rates(irsigp);
    dfdy(Mg24, Si28) = b(1) + b(2);



    // si28 jacobian elements
    // d(si28)/d(he4)
    b(1) =  y(Mg24) * rr(1).rates(irmgag);
    b(2) = -y(Si28) * rr(1).rates(irsiag);
    b(3) =  y(Mg24) * rr(1).rates(irmgap) * (1.0e0_rt-rr(1).rates(irr1));
    b(4) = -y(Si28) * rr(1).rates(irsiap) * (1.0e0_rt-rr(1).rates(irs1));
    dfdy(Si28, He4) = esum4(b);

    // d(si28)/d(c12)
    b(1) =  0.5e0_rt * y(O16) * rr(1).rates(ir1216);
    dfdy(Si28, C12) = b(1);

    // d(si28)/d(o16)
    b(1) = 0.5e0_rt * y(C12) * rr(1).rates(ir1216);
    b(2) = 1.12e0_rt * y(O16) * rr(1).rates(ir1616);
    b(3) = 0.68e0_rt * y(O16) * rr(1).rates(irs1) * rr(1).rates(ir1616);
    dfdy(Si28, O16) = esum3(b);

    // d(si28)/d(mg24)
    b(1) =  y(He4) * rr(1).rates(irmgag);
    b(2) =  y(He4) * rr(1).rates(irmgap) * (1.0e0_rt-rr(1).rates(irr1));
    dfdy(Si28, Mg24) = b(1) + b(2);

    // d(si28)/d(si28)
    b(1) =  -y(He4) * rr(1).rates(irsiag);
    b(2) = -rr(1).rates(irsiga);
    b(3) = -rr(1).rates(irr1) * rr(1).rates(irsigp);
    b(4) = -y(He4) * rr(1).rates(irsiap) * (1.0e0_rt-rr(1).rates(irs1));
    dfdy(Si28, Si28) = esum4(b);

    // d(si28)/d(s32)
    b(1) = rr(1).rates(irsga);
    b(2) = rr(1).rates(irs1) * rr(1).rates(irsgp);
    dfdy(Si28, S32) = b(1) + b(2);


    // s32 jacobian elements
    // d(s32)/d(he4)
    b(1) =  y(Si28) * rr(1).rates(irsiag);
    b(2) = -y(S32) * rr(1).rates(irsag);
    b(3) =  y(Si28) * rr(1).rates(irsiap) * (1.0e0_rt-rr(1).rates(irs1));
    b(4) = -y(S32) * rr(1).rates(irsap) * (1.0e0_rt-rr(1).rates(irt1));
    dfdy(S32, He4) = esum4(b);

    // d(s32)/d(o16)
    b(1) = 0.68e0_rt * y(O16)*rr(1).rates(ir1616)*(1.0e0_rt-rr(1).rates(irs1));
    b(2) = 0.2e0_rt * y(O16) * rr(1).rates(ir1616);
    dfdy(S32, O16) = b(1) + b(2);

    // d(s32)/d(si28)
    b(1) = y(He4) * rr(1).rates(irsiag);
    b(2) = y(He4) * rr(1).rates(irsiap) * (1.0e0_rt-rr(1).rates(irs1));
    dfdy(S32, Si28) = b(1) + b(2);

    // d(s32)/d(s32)
    b(1) = -y(He4) * rr(1).rates(irsag);
    b(2) = -rr(1).rates(irsga);
    b(3) = -rr(1).rates(irs1) * rr(1).rates(irsgp);
    b(4) = -y(He4) * rr(1).rates(irsap) * (1.0e0_rt-rr(1).rates(irt1));
    dfdy(S32, S32) = esum4(b);

    // d(s32)/d(ar36)
    b(1) = rr(1).rates(irarga);
    b(2) = rr(1).rates(irt1) * rr(1).rates(irargp);
    dfdy(S32, Ar36) = b(1) + b(2);


    // ar36 jacobian elements
    // d(ar36)/d(he4)
    b(1) =  y(S32)  * rr(1).rates(irsag);
    b(2) = -y(Ar36) * rr(1).rates(irarag);
    b(3) =  y(S32)  * rr(1).rates(irsap) * (1.0e0_rt-rr(1).rates(irt1));
    b(4) = -y(Ar36) * rr(1).rates(irarap) * (1.0e0_rt-rr(1).rates(iru1));
    dfdy(Ar36, He4) = esum4(b);

    // d(ar36)/d(s32)
    b(1) = y(He4) * rr(1).rates(irsag);
    b(2) = y(He4) * rr(1).rates(irsap) * (1.0e0_rt-rr(1).rates(irt1));
    dfdy(Ar36, S32) = b(1) + b(2);

    // d(ar36)/d(ar36)
    b(1) = -y(He4) * rr(1).rates(irarag);
    b(2) = -rr(1).rates(irarga);
    b(3) = -rr(1).rates(irt1) * rr(1).rates(irargp);
    b(4) = -y(He4) * rr(1).rates(irarap) * (1.0e0_rt-rr(1).rates(iru1));
    dfdy(Ar36, Ar36) = esum4(b);

    // d(ar36)/d(ca40)
    b(1) = rr(1).rates(ircaga);
    b(2) = rr(1).rates(ircagp) * rr(1).rates(iru1);
    dfdy(Ar36, Ca40) = b(1) + b(2);



    // ca40 jacobian elements
    // d(ca40)/d(he4)
    b(1)  =  y(Ar36) * rr(1).rates(irarag);
    b(2)  = -y(Ca40) * rr(1).rates(ircaag);
    b(3)  =  y(Ar36) * rr(1).rates(irarap)*(1.0e0_rt-rr(1).rates(iru1));
    b(4)  = -y(Ca40) * rr(1).rates(ircaap)*(1.0e0_rt-rr(1).rates(irv1));
    dfdy(Ca40, He4) = esum4(b);

    // d(ca40)/d(ar36)
    b(1) =  y(He4) * rr(1).rates(irarag);
    b(2) =  y(He4) * rr(1).rates(irarap)*(1.0e0_rt-rr(1).rates(iru1));
    dfdy(Ca40, Ar36) = b(1) + b(2);

    // d(ca40)/d(ca40)
    b(1) =  -y(He4) * rr(1).rates(ircaag);
    b(2) = -rr(1).rates(ircaga);
    b(3) = -rr(1).rates(ircagp) * rr(1).rates(iru1);
    b(4) = -y(He4) * rr(1).rates(ircaap)*(1.0e0_rt-rr(1).rates(irv1));
    dfdy(Ca40, Ca40) = esum4(b);

    // d(ca40)/d(ti44)
    b(1) = rr(1).rates(irtiga);
    b(2) = rr(1).rates(irtigp) * rr(1).rates(irv1);
    dfdy(Ca40, Ti44) = b(1) + b(2);


    // ti44 jacobian elements
    // d(ti44)/d(he4)
    b(1) =  y(Ca40) * rr(1).rates(ircaag);
    b(2) = -y(Ti44) * rr(1).rates(irtiag);
    b(3) =  y(Ca40) * rr(1).rates(ircaap)*(1.0e0_rt-rr(1).rates(irv1));
    b(4) = -y(Ti44) * rr(1).rates(irtiap)*(1.0e0_rt-rr(1).rates(irw1));
    dfdy(Ti44, He4) = esum4(b);

    // d(ti44)/d(ca40)
    b(1) =  y(He4) * rr(1).rates(ircaag);
    b(2) =  y(He4) * rr(1).rates(ircaap)*(1.0e0_rt-rr(1).rates(irv1));
    dfdy(Ti44, Ca40) = b(1) + b(2);

    // d(ti44)/d(ti44)
    b(1) = -y(He4) * rr(1).rates(irtiag);
    b(2) = -rr(1).rates(irtiga);
    b(3) = -rr(1).rates(irv1) * rr(1).rates(irtigp);
    b(4) = -y(He4) * rr(1).rates(irtiap)*(1.0e0_rt-rr(1).rates(irw1));
    dfdy(Ti44, Ti44) = esum4(b);

    // d(ti44)/d(cr48)
    b(1) = rr(1).rates(ircrga);
    b(2) = rr(1).rates(irw1) * rr(1).rates(ircrgp);
    dfdy(Ti44, Cr48) = b(1) + b(2);


    // cr48 jacobian elements
    // d(cr48)/d(he4)
    b(1) =  y(Ti44) * rr(1).rates(irtiag);
    b(2) = -y(Cr48) * rr(1).rates(ircrag);
    b(3) =  y(Ti44) * rr(1).rates(irtiap)*(1.0e0_rt-rr(1).rates(irw1));
    b(4) = -y(Cr48) * rr(1).rates(ircrap)*(1.0e0_rt-rr(1).rates(irx1));
    dfdy(Cr48, He4) = esum4(b);

    // d(cr48)/d(ti44)
    b(1) =  y(He4) * rr(1).rates(irtiag);
    b(2) =  y(He4) * rr(1).rates(irtiap)*(1.0e0_rt-rr(1).rates(irw1));
    dfdy(Cr48, Ti44) = b(1) + b(2);

    // d(cr48)/d(cr48)
    b(1) = -y(He4) * rr(1).rates(ircrag);
    b(2) = -rr(1).rates(ircrga);
    b(3) = -rr(1).rates(irw1) * rr(1).rates(ircrgp);
    b(4) = -y(He4) * rr(1).rates(ircrap)*(1.0e0_rt-rr(1).rates(irx1));
    dfdy(Cr48, Cr48) = esum4(b);

    // d(cr48)/d(fe52)
    b(1) = rr(1).rates(irfega);
    b(2) = rr(1).rates(irx1) * rr(1).rates(irfegp);
    dfdy(Cr48, Fe52) = b(1) + b(2);


    // cr56 jacobian elements
    // d(cr56)/d(cr56)
    b(1) = 0.0_rt;
    dfdy(Cr56, Cr56) = b(1);

    // d(cr56)/d(fe56)
    b(1) =  1.0e-04_rt * rr(1).rates(irn56ec);
    dfdy(Cr56, Fe56) = b(1);




    // fe52 jacobian elements
    // d(fe52)/d(he4)
    b(1) =  y(Cr48) * rr(1).rates(ircrag);
    b(2) = -y(Fe52) * rr(1).rates(irfeag);
    b(3) =  y(Cr48) * rr(1).rates(ircrap) * (1.0e0_rt-rr(1).rates(irx1));
    b(4) = -y(Fe52) * rr(1).rates(ir6f54);
    b(5) = -y(Fe52) * y(P) * rr(1).rates(ir7f54);
    dfdy(Fe52, He4) = esum5(b);

    // d(fe52)/d(cr48)
    b(1) = y(He4) * rr(1).rates(ircrag);
    b(2) = y(He4) * rr(1).rates(ircrap) * (1.0e0_rt-rr(1).rates(irx1));
    dfdy(Fe52, Cr48) = b(1) + b(2);

    // d(fe52)/d(fe52)
    b(1) = -y(He4) * rr(1).rates(irfeag);
    b(2) = -rr(1).rates(irfega);
    b(3) = -rr(1).rates(irx1) * rr(1).rates(irfegp);
    b(4) = -y(N) * y(N) * rr(1).rates(ir2f54);
    b(5) = -y(He4) * rr(1).rates(ir6f54);
    b(6) = -y(He4) * y(P) * rr(1).rates(ir7f54);
    dfdy(Fe52, Fe52) = esum6(b);

    // d(fe52)/d(fe54)
    b(1) = rr(1).rates(ir1f54);
    b(2) = y(P) * y(P) * rr(1).rates(ir5f54);
    dfdy(Fe52, Fe54) = b(1) + b(2);

    // d(fe52)/d(ni56)
    b(1) = rr(1).rates(irniga);
    b(2) = y(P) * rr(1).rates(ir8f54);
    dfdy(Fe52, Ni56) = b(1) + b(2);

    // d(fe52)/d(neut)
    b(1) =  y(Fe54) * rr(3).rates(ir1f54);
    b(2) = -2.0e0_rt * y(Fe52) * y(N) * rr(1).rates(ir2f54);
    b(3) = -y(Fe52) * y(N) * y(N) * rr(3).rates(ir2f54);
    dfdy(Fe52, N) = esum3(b);

    // d(fe52)/d(prot)
    b(1) =  2.0e0_rt * y(Fe54) * y(P) * rr(1).rates(ir5f54);
    b(2) =  y(Fe54) * y(P) * y(P) * rr(3).rates(ir5f54);
    b(3) = -y(He4) * y(Fe52) * rr(3).rates(ir6f54);
    b(4) = -y(Fe52) * y(He4) * rr(1).rates(ir7f54);
    b(5) = -y(Fe52) * y(He4) * y(P) * rr(3).rates(ir7f54);
    b(6) =  y(Ni56) * rr(1).rates(ir8f54);
    b(7) =  y(Ni56) * y(P) * rr(3).rates(ir8f54);
    dfdy(Fe52, P) = esum7(b);




    // fe54 jacobian elements
    // d(fe54)/d(he4)
    b(1) =  y(Fe52) * rr(1).rates(ir6f54);
    b(2) = -y(Fe54) * rr(1).rates(irfe56_aux4);
    dfdy(Fe54, He4) = b(1) + b(2);

    // d(fe54)/d(fe52)
    b(1) =  y(N) * y(N) * rr(1).rates(ir2f54);
    b(2) =  y(He4) * rr(1).rates(ir6f54);
    dfdy(Fe54, Fe52) = b(1) + b(2);

    // d(fe54)/d(fe54)
    b(1) = -rr(1).rates(ir1f54);
    b(2) = -y(P) * y(P) * rr(1).rates(ir3f54);
    b(3) = -y(P) * y(P) * rr(1).rates(ir5f54);
    b(4) = -y(N) * y(N) * rr(1).rates(irfe56_aux2);
    b(5) = -y(He4) * rr(1).rates(irfe56_aux4);
    dfdy(Fe54, Fe54) = esum5(b);

    // d(fe54)/d(fe56)
    b(1) = rr(1).rates(irfe56_aux1);
    b(2) = y(P) * y(P) * rr(1).rates(irfe56_aux3);
    dfdy(Fe54, Fe56) = b(1) + b(2);

    // d(fe54)/d(ni56)
    b(1) = rr(1).rates(ir4f54);
    dfdy(Fe54, Ni56) = b(1);

    // d(fe54)/d(neut)
    b(1) = -y(Fe54) * rr(3).rates(ir1f54);
    b(2) =  2.0e0_rt * y(Fe52) * y(N) * rr(1).rates(ir2f54);
    b(3) =  y(Fe52) * y(N) * y(N) * rr(3).rates(ir2f54);
    b(4) =  y(Fe56) * rr(3).rates(irfe56_aux1);
    b(5) = -2.0e0_rt * y(Fe54) * y(N) * rr(1).rates(irfe56_aux2);
    b(6) = -y(Fe54) * y(N) * y(N) * rr(3).rates(irfe56_aux2);
    dfdy(Fe54, N) = esum6(b);

    // d(fe54)/d(prot)
    b(1) = -2.0e0_rt * y(Fe54) * y(P) * rr(1).rates(ir3f54);
    b(2) = -y(Fe54) * y(P) * y(P) * rr(3).rates(ir3f54);
    b(3) =  y(Ni56) * rr(3).rates(ir4f54);
    b(4) = -2.0e0_rt * y(Fe54) * y(P) * rr(1).rates(ir5f54);
    b(5) = -y(Fe54) * y(P) * y(P) * rr(3).rates(ir5f54);
    b(6) =  y(He4) * y(Fe52) * rr(3).rates(ir6f54);
    b(7) =  2.0e0_rt * y(Fe56) * y(P) * rr(1).rates(irfe56_aux3);
    b(8) =  y(Fe56) * y(P) * y(P) * rr(3).rates(irfe56_aux3);
    b(9) = -y(He4) * y(Fe54) * rr(3).rates(irfe56_aux4);
    dfdy(Fe54, P) = esum9(b);


    // fe56 jacobian elements

    // d(fe56)/d(he4)
    b(1) = y(Fe54) * rr(1).rates(irfe56_aux4);
    dfdy(Fe56, He4) = b(1);

    // d(fe56)/d(fe54)
    b(1)  = y(N) * y(N) * rr(1).rates(irfe56_aux2);
    b(2)  = y(He4) * rr(1).rates(irfe56_aux4);
    dfdy(Fe56, Fe54) = b(1) + b(2);

    // d(fe56)/d(fe56)
    b(1) = -1.0e-04_rt * rr(1).rates(irn56ec);
    b(2) = -rr(1).rates(irfe56_aux1);
    b(3) = -y(P) * y(P) * rr(1).rates(irfe56_aux3);
    dfdy(Fe56, Fe56) = esum3(b);

    // d(fe56)/d(ni56)
    b(1) = rr(1).rates(irn56ec);
    dfdy(Fe56, Ni56) = b(1);

    // d(fe56)/d(neut)
    b(1) = -y(Fe56) * rr(3).rates(irfe56_aux1);
    b(2) =  2.0e0_rt * y(Fe54) * y(N) * rr(1).rates(irfe56_aux2);
    b(3) =  y(Fe54) * y(N) * y(N) * rr(3).rates(irfe56_aux2);
    dfdy(Fe56, N) = esum3(b);

    // d(fe56)/d(prot)
    b(1) = -2.0e0_rt * y(Fe56) * y(P) * rr(1).rates(irfe56_aux3);
    b(2) = -y(Fe56) * y(P) * y(P) * rr(3).rates(irfe56_aux3);
    b(3) =  y(He4) * y(Fe54) * rr(3).rates(irfe56_aux4);
    dfdy(Fe56, P) = esum3(b);



    // ni56 jacobian elements
    // d(ni56)/d(he4)
    b(1) =  y(Fe52) * rr(1).rates(irfeag);
    b(2) =  y(Fe52) * y(P) * rr(1).rates(ir7f54);
    dfdy(Ni56, He4) = b(1) + b(2);

    // d(ni56)/d(fe52)
    b(1) = y(He4) * rr(1).rates(irfeag);
    b(2) = y(He4)* y(P) * rr(1).rates(ir7f54);
    dfdy(Ni56, Fe52) = b(1) + b(2);

    // d(ni56)/d(fe54)
    b(1) = y(P) * y(P) * rr(1).rates(ir3f54);
    dfdy(Ni56, Fe54) = b(1);

    // d(ni56)/d(ni56)
    b(1) = -rr(1).rates(irniga);
    b(2) = -rr(1).rates(ir4f54);
    b(3) = -y(P) * rr(1).rates(ir8f54);
    b(4) = -rr(1).rates(irn56ec);
    dfdy(Ni56, Ni56) = esum4(b);

    // d(ni56)/d(prot)
    b(1) =  2.0e0_rt * y(Fe54) * y(P) * rr(1).rates(ir3f54);
    b(2) =  y(Fe54) * y(P) * y(P) * rr(3).rates(ir3f54);
    b(3) = -y(Ni56) * rr(3).rates(ir4f54);
    b(4) =  y(Fe52) * y(He4)* rr(1).rates(ir7f54);
    b(5) =  y(Fe52) * y(He4)* y(P) * rr(3).rates(ir7f54);
    b(6) = -y(Ni56) * rr(1).rates(ir8f54);
    b(7) = -y(Ni56) * y(P) * rr(3).rates(ir8f54);
    dfdy(Ni56, P) = esum7(b);


    // photodisintegration neutron jacobian elements
    // d(neut)/d(he4)
    b(1) = 2.0e0_rt * rr(1).rates(iralf1);
    dfdy(N, He4) = b(1);

    // d(neut)/d(fe52)
    b(1) = -2.0e0_rt * y(N) * y(N) * rr(1).rates(ir2f54);
    dfdy(N, Fe52) = b(1);

    // d(neut)/d(fe54)
    b(1) = 2.0e0_rt * rr(1).rates(ir1f54);
    b(2) = -2.0e0_rt * y(N) * y(N) * rr(1).rates(irfe56_aux2);
    dfdy(N, Fe54) = b(1) + b(2);

    // d(neut)/d(fe56)
    b(1) = 2.0e0_rt * rr(1).rates(irfe56_aux1);
    dfdy(N, Fe56) = b(1);

    // d(neut)/d(neut)
    b(1)  =  2.0e0_rt * y(Fe54) * rr(3).rates(ir1f54);
    b(2)  = -4.0e0_rt * y(Fe52) * y(N) * rr(1).rates(ir2f54);
    b(3)  = -2.0e0_rt * y(Fe52) * y(N) * y(N) * rr(3).rates(ir2f54);
    b(4)  =  2.0e0_rt * y(He4) * rr(3).rates(iralf1);
    b(5)  = -4.0e0_rt * y(N) * y(P)*y(P) * rr(1).rates(iralf2);
    b(6)  = -2.0e0_rt * y(N)*y(N) * y(P)*y(P) * rr(3).rates(iralf2);
    b(7)  = -rr(1).rates(irnep);
    b(8)  =  2.0e0_rt * y(Fe56) * rr(3).rates(irfe56_aux1);
    b(9)  = -4.0e0_rt * y(Fe54) * y(N) * rr(1).rates(irfe56_aux2);
    b(10) = -2.0e0_rt * y(Fe54) * y(N) * y(N) * rr(3).rates(irfe56_aux2);
    dfdy(N, N) = esum10(b);

    // d(neut)/d(prot)
    b(1) =  2.0e0_rt * y(He4) * rr(4).rates(iralf1);
    b(2) = -4.0e0_rt * y(N)*y(N) * y(P) * rr(1).rates(iralf2);
    b(3) = -2.0e0_rt * y(N)*y(N) * y(P)*y(P) * rr(4).rates(iralf2);
    b(4) =  rr(1).rates(irpen);
    dfdy(N, P) = esum4(b);


    // photodisintegration proton jacobian elements
    // d(prot)/d(he4)
    b(1) =  2.0e0_rt * y(Fe52) * rr(1).rates(ir6f54);
    b(2) =  2.0e0_rt * rr(1).rates(iralf1);
    b(3) =  2.0e0_rt * y(Fe54) * rr(1).rates(irfe56_aux4);
    dfdy(P, He4) = esum3(b);

    // d(prot)/d(fe52)
    b(1) = 2.0e0_rt * y(He4) * rr(1).rates(ir6f54);
    dfdy(P, Fe52) = b(1);

    // d(prot)/d(fe54)
    b(1) = -2.0e0_rt * y(P) * y(P) * rr(1).rates(ir3f54);
    b(2) = -2.0e0_rt * y(P) * y(P) * rr(1).rates(ir5f54);
    b(3) =  2.0e0_rt * y(He4) * rr(1).rates(irfe56_aux4);
    dfdy(P, Fe54) = esum3(b);

    // d(prot)/d(fe56)
    b(1) = -2.0e0_rt * y(P) * y(P) * rr(1).rates(irfe56_aux3);
    dfdy(P, Fe56) = b(1);

    // d(prot)/d(ni56)
    b(1) = 2.0e0_rt * rr(1).rates(ir4f54);
    dfdy(P, Ni56) = b(1);

    // d(prot)/d(neut)
    b(1) =  2.0e0_rt * y(He4) * rr(3).rates(iralf1);
    b(2) = -4.0e0_rt * y(N) * y(P)*y(P) * rr(1).rates(iralf2);
    b(3) = -2.0e0_rt * y(N)*y(N) * y(P)*y(P) * rr(3).rates(iralf2);
    b(4) =  rr(1).rates(irnep);
    dfdy(P, N) = esum4(b);

    // d(prot)/d(prot)
    b(1)  =  -4.0e0_rt * y(Fe54) * y(P) * rr(1).rates(ir3f54);
    b(2)  =  -2.0e0_rt * y(Fe54) * y(P)*y(P) * rr(3).rates(ir3f54);
    b(3)  =   2.0e0_rt * y(Ni56) * rr(3).rates(ir4f54);
    b(4)  =  -4.0e0_rt * y(Fe54) * y(P) * rr(1).rates(ir5f54);
    b(5)  =  -2.0e0_rt * y(Fe54) * y(P)*y(P) * rr(3).rates(ir5f54);
    b(6)  =   2.0e0_rt * y(He4) * y(Fe52) * rr(3).rates(ir6f54);
    b(7)  =   2.0e0_rt * y(He4) * rr(4).rates(iralf1);
    b(8)  =  -4.0e0_rt * y(N)*y(N) * y(P) * rr(1).rates(iralf2);
    b(9)  =  -2.0e0_rt * y(N)*y(N) * y(P)*y(P) * rr(4).rates(iralf2);
    b(10) =  -rr(1).rates(irpen);
    b(11) =  -4.0e0_rt * y(Fe56) * y(P) * rr(1).rates(irfe56_aux3);
    b(12) =  -2.0e0_rt * y(Fe56) * y(P)*y(P) * rr(3).rates(irfe56_aux3);
    b(13) =   2.0e0_rt * y(He4) * y(Fe54) * rr(3).rates(irfe56_aux4);
    dfdy(P, P) = esum13(b);
}


// Computes the instantaneous energy generation rate
template<class T>
AMREX_GPU_HOST_DEVICE AMREX_INLINE
void ener_gener_rate (T const& dydt, Real& enuc)
{
    using namespace network;

    // This is basically e = m c**2

    Real Xdot = 0.0_rt;

    // Sum the mass fraction time derivatives
    for (int i = 1; i <= NumSpec; ++i) {
        Xdot += dydt(i) * mion(i);
    }

    // This is basically e = m c**2
    enuc = Xdot * C::Legacy::enuc_conv2;
}


// Compute and store the more expensive screening factors
AMREX_INLINE
void set_up_screening_factors ()
{
    // note: it is critical that these are called in the exact order
    // that the screening calls are done in the RHS routine, since we
    // use that order in the screening

    using namespace Species;

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

    add_screening_factor(jscr++, zion[Fe54-1], aion[Fe54-1], 1.0e0_rt, 1.0e0_rt);

    add_screening_factor(jscr++, zion[Fe54-1], aion[Fe54-1], zion[He4-1], aion[He4-1]);

    add_screening_factor(jscr++, zion[Fe56-1], aion[Fe56-1], 1.0e0_rt, 1.0e0_rt);

    add_screening_factor(jscr++, 1.0e0_rt, 2.0e0_rt, zion[H1-1], aion[H1-1]);

    add_screening_factor(jscr++, zion[H1-1], aion[H1-1], zion[H1-1], aion[H1-1]);

    add_screening_factor(jscr++, zion[He3-1], aion[He3-1], zion[He3-1], aion[He3-1]);

    add_screening_factor(jscr++, zion[He3-1], aion[He3-1], zion[He4-1], aion[He4-1]);

    add_screening_factor(jscr++, zion[C12-1], aion[C12-1], zion[H1-1], aion[H1-1]);

    add_screening_factor(jscr++, zion[N14-1], aion[N14-1], zion[H1-1], aion[H1-1]);

    add_screening_factor(jscr++, zion[O16-1], aion[O16-1], zion[H1-1], aion[H1-1]);

    add_screening_factor(jscr++, zion[N14-1], aion[N14-1], zion[He4-1], aion[He4-1]);
}

AMREX_GPU_HOST_DEVICE AMREX_INLINE
void actual_rhs (burn_t& state, Array1D<Real, 1, neqs>& ydot)
{
    using namespace Rates;

    // This routine sets up the system of ODE's for the aprox21
    // nuclear reactions.  This is an alpha chain + heavy ion network
    // with (a,p)(p,g) links, as well as iron-group elements for NSE
    // and hydrogen and nitrogen for PP and CNO burning.
    //
    // Isotopes: h1,   he3,  he4,  c12,  n14,  o16,  ne20, mg24, si28, s32,
    //           ar36, ca40, ti44, cr48, fe52, fe54, cr56, fe56, ni56, neut, prot

    Array1D<rate_t, 1, Rates::NumGroups> rr;

    bool deriva;

    Real sneut, dsneutdt, dsneutdd, snuda, snudz;
    Real enuc;

    Real rho, temp, abar, zbar;

    Array1D<Real, 1, NumSpec> y;

    deriva = false;

    evaluate_rates(state, rr);

    // Get the data from the state

    rho  = state.rho;
    temp = state.T;
    abar = state.abar;
    zbar = state.zbar;

    for (int i = 1; i <= NumSpec; ++i) {
        y(i) = state.xn[i-1] * aion_inv[i-1];
    }

    // Call the RHS to actually get dydt.

    bool for_jacobian_tderiv = false;
    rhs(y, rr, ydot, deriva, for_jacobian_tderiv);

    // Instantaneous energy generation rate -- this needs molar fractions

    ener_gener_rate(ydot, enuc);

    // Get the neutrino losses

    sneut5(temp, rho, abar, zbar, sneut, dsneutdt, dsneutdd, snuda, snudz);

    // Append the energy equation (this is erg/g/s)

    ydot(net_ienuc) = enuc - sneut;
}

// Analytical Jacobian

template<class MatrixType>
AMREX_GPU_HOST_DEVICE AMREX_INLINE
void actual_jac (burn_t& state, MatrixType& jac)
{
    Array1D<rate_t, 1, Rates::NumGroups> rr;

    bool deriva;

    Real b1, sneut, dsneutdt, dsneutdd, snuda, snudz;

    Real rho, temp, abar, zbar;
    Array1D<Real, 1, NumSpec> y;
    Array1D<Real, 1, neqs> yderivs;

    deriva = true;

    jac.zero();

    evaluate_rates(state, rr);

    // Get the data from the state

    rho  = state.rho;
    temp = state.T;
    abar = state.abar;
    zbar = state.zbar;

    for (int n = 1; n <= NumSpec; ++n) {
        y(n) = state.xn[n-1] * aion_inv[n-1];
    }

    // Species Jacobian elements with respect to other species

    dfdy_isotopes_aprox21(y, rr, jac);

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
    // calling the RHS using d(ratdum) / dT

    bool for_jacobian_tderiv = true;
    rhs(y, rr, yderivs, deriva, for_jacobian_tderiv);

    for (int i = 1; i <= NumSpec; ++i) {
        jac(i, net_ienuc) = temperature_to_energy_jacobian(state, yderivs(i));
    }

    Real jac_e_T;
    ener_gener_rate(yderivs, jac_e_T);
    jac_e_T -= dsneutdt;
    jac(net_ienuc, net_ienuc) = temperature_to_energy_jacobian(state, jac_e_T);
}

#endif
