#ifndef INTEGRATION_UTILS_H
#define INTEGRATION_UTILS_H

#include <burn_type.H>

///
/// Compute the numerical Jacobian of the reactive system, dydot/dy,
/// where ydot = {dX/dt, denuc/dt}, and y = {X, e}.
///
/// Note the analytic Jacobian usually works in terms of molar
/// fractions, Y, instead of mass fractions, X, but our integrators
/// always work in terms of X, so the numerical Jacobian is expressed
/// in terms of X.
///
AMREX_GPU_HOST_DEVICE
void numerical_jac(burn_t& state, JacNetArray2D& jac);

#endif
