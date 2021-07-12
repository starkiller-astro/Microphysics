#ifndef VODE_JAC_UTIL_SIMPLIFIED_SDC_H
#define VODE_JAC_UTIL_SIMPLIFIED_SDC_H

AMREX_GPU_HOST_DEVICE
void jac_to_vode(const Real time, burn_t& state,
                 JacNetArray2D& jac_react, dvode_t& vode_state,
                 RArray2D& jac);

AMREX_GPU_HOST_DEVICE
void rhs_to_vode(const Real time, burn_t& state,
                 YdotNetArray1D& ydot_react, dvode_t& vode_state,
                 RArray1D& ydot);
#endif
