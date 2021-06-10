#ifndef _vode_dvstep_H_
#define _vode_dvstep_H_

#include <vode_type.H>
#include <vode_dvset.H>
#include <vode_dvjust.H>
#include <vode_dvnlsd.H>

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
void advance_nordsieck (dvode_t& vstate)
{

    // Effectively multiplies the Nordsieck history
    // array by the Pascal triangle matrix.

    for (int k = vstate.NQ; k >= 1; --k) {
        for (int j = k; j <= vstate.NQ; ++j) {
            for (int i = 1; i <= VODE_NEQS; ++i) {
                vstate.yh(i,j) += vstate.yh(i,j+1);
            }
        }
    }

}


AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
void retract_nordsieck (dvode_t& vstate)
{

    // Undoes the Pascal triangle matrix multiplication
    // implemented in subroutine advance_nordsieck.

    for (int k = vstate.NQ; k >= 1; --k) {
        for (int j = k; j <= vstate.NQ; ++j) {
            for (int i = 1; i <= VODE_NEQS; ++i) {
                vstate.yh(i,j) -= vstate.yh(i,j+1);
            }
        }
    }

}


AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
int dvstep (burn_t& state, dvode_t& vstate)
{

    // dvstep performs one step of the integration of an initial value
    // problem for a system of ordinary differential equations.
    // dvstep calls subroutine dvnlsd for the solution of the nonlinear system
    // arising in the time step.  Thus it is independent of the problem
    // Jacobian structure and the type of nonlinear system solution method.
    // dvstep returns a completion flag kflag.

    // A return with kflag = -1 or -2 means either abs(H) = HMIN or 10
    // consecutive failures occurred. On a return with kflag negative,
    // the values of TN and the yh array are as of the beginning of the last
    // step, and H is the last step size attempted.

    // On a successful return, ETAMAX is reset and ACOR is scaled.

    // Parameter declarations
    const int KFC = -3;
    const int KFH = -7;
    const int MXNCF = 10;
    const Real ADDON = 1.0e-6_rt;
    const Real BIAS1 = 6.0e0_rt;
    const Real BIAS2 = 6.0e0_rt;
    const Real BIAS3 = 10.0e0_rt;
    const Real ETACF = 0.25e0_rt;
    const Real ETAMIN = 0.1e0_rt;
    const Real ETAMXF = 0.2e0_rt;
    const Real ETAMX2 = 10.0e0_rt;
    const Real ETAMX3 = 10.0e0_rt;
    const Real ONEPSM = 1.00001e0_rt;
    const Real THRESH = 1.5e0_rt;

    Real CNQUOT, DDN, DSM, DUP, TOLD;
    Real ETAQ, ETAQM1, ETAQP1, FLOTL, R;
    int NCF, NFLAG;

    ETAQ = 1.0_rt;
    ETAQM1 = 1.0_rt;

    int kflag = 0;
    TOLD = vstate.tn;
    NCF = 0;
    vstate.JCUR = 0;
    NFLAG = 0;

    // Take preliminary actions if the driver changed H. ETA must be
    // reset and the history array yh is rescaled.

    // If a change of order was dictated on the previous step, then
    // it is done here and appropriate adjustments in the history are made.

    if (vstate.NEWH != 0) {

        if (vstate.NEWQ < vstate.NQ) {
            dvjust(-1, vstate);
            vstate.NQ = vstate.NEWQ;
            vstate.L = vstate.NQ + 1;
            vstate.NQWAIT = vstate.L;
        }
        else if (vstate.NEWQ > vstate.NQ) {
            dvjust(1, vstate);
            vstate.NQ = vstate.NEWQ;
            vstate.L = vstate.NQ + 1;
            vstate.NQWAIT = vstate.L;
        }

        // Rescale the history array for a change in H by a factor of ETA.

        R = 1.0_rt;

        for (int j = 2; j <= vstate.L; ++j) {
            R *= vstate.ETA;
            for (int i = 1; i <= VODE_NEQS; ++i) {
                vstate.yh(i,j) *= R;
            }
        }

        vstate.H = vstate.HSCAL * vstate.ETA;
        vstate.HSCAL = vstate.H;
        vstate.RC = vstate.RC * vstate.ETA;

    }

    IArray1D pivot;

    // Compute the predicted values by effectively
    // multiplying the yh array by the Pascal triangle matrix.
    // dvset is called to calculate all integration coefficients.
    // RC is the ratio of new to old values of the coefficient H / EL(2) = h / l1.

    while (true) {

        vstate.tn += vstate.H;

        advance_nordsieck(vstate);

        dvset(vstate);

        vstate.RL1 = 1.0_rt / vstate.el(2);
        vstate.RC *= (vstate.RL1 / vstate.PRL1);
        vstate.PRL1 = vstate.RL1;

        // Save the value of y before calling the solver.
        // We will use this saved value to determine if any
        // constraints were violated in the update.

        Array1D<Real, 1, VODE_NEQS> y_save;
        for (int i = 1; i <= VODE_NEQS; ++i) {
            y_save(i) = vstate.y(i);
        }

        // Call the nonlinear system solver.

        Real ACNRM = dvnlsd(pivot, NFLAG, state, vstate);

        if (NFLAG != 0) {

            // The VNLS routine failed to achieve convergence (NFLAG != 0).
            // The yh array is retracted to its values before prediction.
            // The step size H is reduced and the step is retried, if possible.
            // Otherwise, an error exit is taken.

            NCF += 1;
            vstate.ETAMAX = 1.0_rt;
            vstate.tn = TOLD;

            retract_nordsieck(vstate);

            if (std::abs(vstate.H) <= HMIN * ONEPSM) {
                kflag = -2;
                return kflag;
            }

            if (NCF == MXNCF) {
                kflag = -2;
                return kflag;
            }

            vstate.ETA = ETACF;
            vstate.ETA = amrex::max(vstate.ETA, HMIN / std::abs(vstate.H));

            // Rescale the history array for a change in H by a factor of ETA.
            R = 1.0_rt;

            for (int j = 2; j <= vstate.L; ++j) {
                R *= vstate.ETA;
                for (int i = 1; i <= VODE_NEQS; ++i) {
                    vstate.yh(i,j) *= R;
                }
            }

            vstate.H = vstate.HSCAL * vstate.ETA;
            vstate.HSCAL = vstate.H;
            vstate.RC *= vstate.ETA;

            continue;
        }

        // We add to VODE some constraints. If these constraints are violated,
        // we treat it the same way we will treat an error test failure (below).

        bool valid_update = true;

#ifdef TRUE_SDC
        if (vstate.y(1) < 0.0_rt) {
            // density is negative
            valid_update = false;
        }

        if (vstate.y(NumSpec+2) < 0.0_rt) {
            // rho e is negative
            valid_update = false;
        }
#endif

#ifdef SIMPLIFIED_SDC
#if defined(SDC_EVOLVE_ENERGY)
	if (vstate.y(SEDEN+1) < 0.0_rt || vstate.y(SEINT+1) < 0.0_rt) {
	    valid_update = false;
	}

        Real rho_current = state.rho_orig + vstate.tn * state.ydot_a[SRHO];

        if (rho_current < 0.0_rt) {
            valid_update = false;
        }
#endif
#endif

        for (int i = 1; i <= NumSpec; ++i) {

#ifdef STRANG
            // Constrain abundances such that they don't change by more than a certain
            // factor in a given step (as long as their initial abundance was large enough
            // for us to worry about).

            if (std::abs(y_save(i)) >= atol_spec &&
                (std::abs(vstate.y(i)) > vode_increase_change_factor * std::abs(y_save(i)) ||
                 std::abs(vstate.y(i)) < vode_decrease_change_factor * std::abs(y_save(i)))) {
                valid_update = false;
            }

            // Constrain abundances such that they are not negative (within a tolerance)
            // or greater than one (within a tolerance).
            if (vstate.y(i) < -vode_failure_tolerance) {
                valid_update = false;
            }

            if (vstate.y(i) > 1.0_rt + vode_failure_tolerance) {
                valid_update = false;
            }

#endif

#ifdef SIMPLIFIED_SDC

            // these are basically the same checks as with Strang above, except
            // now we are evolving rhoX instead of X


            // this doesn't seem to work well with SDC

            //if (std::abs(y_save(SFS+i)) >= atol_spec * rho_current &&
            //    (std::abs(vstate.y(SFS+i)) > vode_increase_change_factor * std::abs(y_save(SFS+i)) ||
            //     std::abs(vstate.y(SFS+i)) < vode_decrease_change_factor * std::abs(y_save(SFS+i)))) {
            //    valid_update = false;
            //}

#if defined(SDC_EVOLVE_ENERGY)
            if (vstate.y(SFS+i) < -vode_failure_tolerance * rho_current) {
                valid_update = false;
            }

            if (vstate.y(SFS+i) > (1.0_rt + vode_failure_tolerance) * rho_current) {
                valid_update = false;
            }
#endif

#endif

#ifdef TRUE_SDC
            // as above, we want to make sure that the mass fractions stay bounded, but we are
            // evolving:
            //
            //  y(1) = density
            //  y(2:2+NumSpec-1) = rho X
            //  y(NumSpec+2) = rho e

            if (vstate.y(1+i) < -vode_failure_tolerance * std::abs(vstate.y(1))) {
                valid_update = false;
            }

            if (vstate.y(1+i) > (1.0_rt + vode_failure_tolerance) * std::abs(vstate.y(1))) {
                valid_update = false;
            }

            if (std::abs(y_save(1+i)) >= atol_spec * vstate.y(1) &&
                (std::abs(vstate.y(1+i)) > vode_increase_change_factor * std::abs(y_save(1+i)) ||
                 std::abs(vstate.y(1+i)) < vode_decrease_change_factor * std::abs(y_save(1+i)))) {
                valid_update = false;
            }

#endif


        }

        // The corrector has converged (NFLAG = 0). The local error test is
        // made and control passes to statement 500 if it fails.

        DSM = ACNRM / vstate.tq(2);
        if (DSM <= 1.0_rt && valid_update) {

            // After a successful step, update the yh and TAU arrays and decrement
            // NQWAIT. If NQWAIT is then 1 and NQ .lt. MAXORD, then ACOR is saved
            // for use in a possible order increase on the next step.
            // If ETAMAX = 1 (a failure occurred this step), keep NQWAIT >= 2.

            kflag = 0;
            vstate.NST += 1;
            for (int iback = 1; iback <= vstate.NQ; ++iback) {
                int i = vstate.L - iback;
                vstate.tau(i+1) = vstate.tau(i);
            }

            vstate.tau(1) = vstate.H;
            for (int j = 1; j <= vstate.L; ++j) {
                for (int i = 1; i <= VODE_NEQS; ++i) {
                    vstate.yh(i,j) += vstate.el(j) * vstate.acor(i);
                }
            }

            vstate.NQWAIT -= 1;
            if ((vstate.L != VODE_LMAX) && (vstate.NQWAIT == 1)) {
                for (int i = 1; i <= VODE_NEQS; ++i) {
                    vstate.yh(i,VODE_LMAX) = vstate.acor(i);
                }
                vstate.CONP = vstate.tq(5);
            }

            if (vstate.ETAMAX != 1.0_rt) {
                break;
            }

            if (vstate.NQWAIT < 2) {
                vstate.NQWAIT = 2;
            }

            vstate.NEWQ = vstate.NQ;
            vstate.NEWH = 0;
            vstate.ETA = 1.0_rt;
            vstate.ETAMAX = ETAMX3;
            if (vstate.NST <= 10) {
                vstate.ETAMAX = ETAMX2;
            }

            R = 1.0_rt / vstate.tq(2);
            for (int i = 1; i <= VODE_NEQS; ++i) {
                vstate.acor(i) *= R;
            }

            return kflag;

        }

        // The error test failed (or our constraints on the species failed).
        // kflag keeps track of multiple failures.
        // Restore tn and the yh array to their previous values, and prepare
        // to try the step again. Compute the optimum step size for the
        // same order. After repeated failures, H is forced to decrease
        // more rapidly.

        kflag -= 1;
        NFLAG = -2;
        vstate.tn = TOLD;

        retract_nordsieck(vstate);

        if (std::abs(vstate.H) <= HMIN * ONEPSM) {
            kflag = -1;
            return kflag;
        }

        vstate.ETAMAX = 1.0_rt;

        if (kflag > KFC) {

            // Compute ratio of new H to current H at the current order.
            FLOTL = vstate.L;
            vstate.ETA = 1.0_rt / (std::pow(BIAS2 * DSM, 1.0_rt / FLOTL) + ADDON);
            vstate.ETA = amrex::max(vstate.ETA, HMIN / std::abs(vstate.H), ETAMIN);
            if ((kflag <= -2) && (vstate.ETA > ETAMXF)) {
                vstate.ETA = ETAMXF;
            }

            // Rescale the history array for a change in H by a factor of ETA.
            R = 1.0_rt;

            for (int j = 2; j <= vstate.L; ++j) {
                R *= vstate.ETA;
                for (int i = 1; i <= VODE_NEQS; ++i) {
                    vstate.yh(i,j) *= R;
                }
            }

            vstate.H = vstate.HSCAL * vstate.ETA;
            vstate.HSCAL = vstate.H;
            vstate.RC = vstate.RC * vstate.ETA;

            continue;

        }

        // Control reaches this section if 3 or more consecutive failures
        // have occurred. It is assumed that the elements of the yh array
        // have accumulated errors of the wrong order. The order is reduced
        // by one, if possible. Then H is reduced by a factor of 0.1 and
        // the step is retried.  After a total of 7 consecutive failures,
        // an exit is taken with kflag = -1.

        if (kflag == KFH) {
            kflag = -1;
            return kflag;
        }

        if (vstate.NQ != 1) {
            vstate.ETA = amrex::max(ETAMIN, HMIN / std::abs(vstate.H));
            dvjust(-1, vstate);
            vstate.L = vstate.NQ;
            vstate.NQ -= 1;
            vstate.NQWAIT = vstate.L;

            // Rescale the history array for a change in H by a factor of ETA.
            R = 1.0_rt;

            for (int j = 2; j <= vstate.L; ++j) {
                R *= vstate.ETA;
                for (int i = 1; i <= VODE_NEQS; ++i) {
                    vstate.yh(i,j) *= R;
                }
            }

            vstate.H = vstate.HSCAL * vstate.ETA;
            vstate.HSCAL = vstate.H;
            vstate.RC = vstate.RC * vstate.ETA;

            continue;

        }

        vstate.ETA = amrex::max(ETAMIN, HMIN / std::abs(vstate.H));
        vstate.H *= vstate.ETA;
        vstate.HSCAL = vstate.H;
        vstate.tau(1) = vstate.H;
        rhs(vstate.tn, state, vstate, vstate.savf);
        vstate.NFE += 1;
        for (int i = 1; i <= VODE_NEQS; ++i) {
            vstate.yh(i,2) = vstate.H * vstate.savf(i);
        }

        vstate.NQWAIT = 10;

    }

    // If NQWAIT = 0, an increase or decrease in order by one is considered.
    // Factors ETAQ, ETAQM1, ETAQP1 are computed by which H could
    // be multiplied at order q, q-1, or q+1, respectively.
    // The largest of these is determined, and the new order and
    // step size set accordingly.
    // A change of H or NQ is made only if H increases by at least a
    // factor of THRESH.  If an order change is considered and rejected,
    // then NQWAIT is set to 2 (reconsider it after 2 steps).

    bool already_set_eta = false;

    // Compute ratio of new H to current H at the current order.
    FLOTL = vstate.L;
    ETAQ = 1.0_rt / (std::pow(BIAS2 * DSM, 1.0_rt / FLOTL) + ADDON);
    if (vstate.NQWAIT == 0) {
        vstate.NQWAIT = 2;
        ETAQM1 = 0.0_rt;
        if (vstate.NQ != 1) {
            // Compute ratio of new H to current H at the current order less one.
            DDN = 0.0_rt;
            for (int i = 1; i <= VODE_NEQS; ++i) {
                DDN += (vstate.yh(i,vstate.L) * vstate.ewt(i)) * (vstate.yh(i,vstate.L) * vstate.ewt(i));
            }
            DDN = std::sqrt(DDN / VODE_NEQS) / vstate.tq(1);
            ETAQM1 = 1.0_rt / (std::pow(BIAS1 * DDN, 1.0_rt / (FLOTL - 1.0_rt)) + ADDON);
        }

        ETAQP1 = 0.0_rt;
        if (vstate.L != VODE_LMAX) {
            // Compute ratio of new H to current H at current order plus one.
            CNQUOT = (vstate.tq(5) / vstate.CONP) * std::pow(vstate.H / vstate.tau(2), vstate.L);
            for (int i = 1; i <= VODE_NEQS; ++i) {
                vstate.savf(i) = vstate.acor(i) - CNQUOT * vstate.yh(i,VODE_LMAX);
            }
            DUP = 0.0_rt;
            for (int i = 1; i <= VODE_NEQS; ++i) {
                DUP += (vstate.savf(i) * vstate.ewt(i)) * (vstate.savf(i) * vstate.ewt(i));
            }
            DUP = std::sqrt(DUP / VODE_NEQS) / vstate.tq(3);
            ETAQP1 = 1.0_rt / (std::pow(BIAS3 * DUP, 1.0_rt / (FLOTL + 1.0_rt)) + ADDON);
        }

        if (ETAQ < ETAQP1) {
            if (ETAQP1 > ETAQM1) {
                vstate.ETA = ETAQP1;
                vstate.NEWQ = vstate.NQ + 1;
                for (int i = 1; i <= VODE_NEQS; ++i) {
                    vstate.yh(i,VODE_LMAX) = vstate.acor(i);
                }
            }
            else {
                vstate.ETA = ETAQM1;
                vstate.NEWQ = vstate.NQ - 1;
            }
            already_set_eta = true;
        }

        if (ETAQ < ETAQM1 && !already_set_eta) {
            vstate.ETA = ETAQM1;
            vstate.NEWQ = vstate.NQ - 1;
            already_set_eta = true;
        }

    }

    if (!already_set_eta) {
        vstate.ETA = ETAQ;
        vstate.NEWQ = vstate.NQ;
    }

    // Test tentative new H against THRESH and ETAMAX, and HMXI, then exit.
    if (vstate.ETA >= THRESH && vstate.ETAMAX != 1.0_rt) {
        vstate.ETA = amrex::min(vstate.ETA, vstate.ETAMAX);
        vstate.ETA = vstate.ETA / amrex::max(1.0_rt, std::abs(vstate.H) * vstate.HMXI * vstate.ETA);
        vstate.NEWH = 1;
        vstate.ETAMAX = ETAMX3;
        if (vstate.NST <= 10) {
            vstate.ETAMAX = ETAMX2;
        }
        R = 1.0_rt / vstate.tq(2);
        for (int i = 1; i <= VODE_NEQS; ++i) {
            vstate.acor(i) *= R;
        }
        return kflag;
    }

    vstate.NEWQ = vstate.NQ;
    vstate.NEWH = 0;
    vstate.ETA = 1.0_rt;
    vstate.ETAMAX = ETAMX3;
    if (vstate.NST <= 10) {
        vstate.ETAMAX = ETAMX2;
    }
    R = 1.0_rt / vstate.tq(2);
    for (int i = 1; i <= VODE_NEQS; ++i) {
        vstate.acor(i) *= R;
    }

    return kflag;

}

#endif
