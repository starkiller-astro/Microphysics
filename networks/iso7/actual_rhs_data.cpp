#include <actual_rhs.H>

namespace RateTable
{
    AMREX_GPU_MANAGED Array3D<Real, 1, Rates::NumRatesFR, 1, 2, 1, nrattab> rattab;
    AMREX_GPU_MANAGED Array3D<Real, 1, Rates::NumRatesFR, 1, 2, 1, nrattab> drattabdt;
    AMREX_GPU_MANAGED Array1D<Real, 1, nrattab> ttab;
}

void actual_rhs_init()
{
    rates_init();

    screening_init();

    set_up_screening_factors();

    if (use_tables)
    {
        amrex::Print() << "\nInitializing iso7 rate table\n";
        set_iso7rat();
    }
}
