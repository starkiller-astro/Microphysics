#ifndef _be_type_H_
#define _be_type_H_

#include <AMReX_REAL.H>
#include <AMReX_Array.H>

#include <ArrayUtilities.H>
#include <network.H>

#ifdef NETWORK_SOLVER
#include <actual_matrix.H>
#endif

#ifdef SIMPLIFIED_SDC
const int BE_NEQS = SVAR_EVOLVE;
#else
const int BE_NEQS = NumSpec + 1;
#endif

const int BE_SUCCESS = 0;
const int BE_NONCONVERGENCE = -100;
const int BE_LU_DECOMPOSITION_ERROR = -101;

typedef amrex::Array1D<int, 1, BE_NEQS> IArray1D;
typedef amrex::Array1D<Real, 1, BE_NEQS> RArray1D;
typedef ArrayUtil::MathArray2D<1, BE_NEQS, 1, BE_NEQS> RArray2D;

#endif
