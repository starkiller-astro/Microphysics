#ifndef rhs_utilities_H
#define rhs_utilities_H

#include <network_utilities.H>
#include <actual_network.H>

// Forward declarations

namespace RHS
{

template<int species, int rate>
AMREX_GPU_HOST_DEVICE AMREX_INLINE
constexpr int is_rate_used ();

template<int species, int j>
AMREX_GPU_HOST_DEVICE AMREX_INLINE
constexpr Real rhs_term (const burn_t& state, rate_fr_t const& rr);

template<int species, int N>
AMREX_GPU_HOST_DEVICE AMREX_INLINE
constexpr Array1D<Real, 1, N>
make_RHS_Array1D (const burn_t& state, rate_fr_t const& rr);

template<int species>
AMREX_GPU_HOST_DEVICE AMREX_INLINE
Real species_rhs_n (const burn_t& state, rate_fr_t const& rr);

AMREX_GPU_HOST_DEVICE AMREX_INLINE
Array1D<Real, 1, NumSpec> species_rhs (const burn_t& state, rate_fr_t const& rr);

} // namespace RHS



// Internal implementation functions

namespace RHS_impl {

// Calculate an integer factorial.
AMREX_GPU_HOST_DEVICE AMREX_INLINE
constexpr int factorial (int n)
{
    if (n <= 1) {
        return 1;
    }
    else {
        return n * factorial(n - 1);
    }
}

// Base case for recursion.
template<int species>
AMREX_GPU_HOST_DEVICE AMREX_INLINE
constexpr int add_nonzero_rates ()
{
    return 0;
}

// Recursively add nonzero rates.
template<int species, int rate, int... rates>
constexpr int add_nonzero_rates ()
{
    return RHS::is_rate_used<species, rate + 1>() + add_nonzero_rates<species, rates...>();
}

template<int species, int... rates>
AMREX_GPU_HOST_DEVICE AMREX_INLINE
constexpr int num_rhs_impl (std::integer_sequence<int, rates...>)
{
    return add_nonzero_rates<species, rates...>();
}

// Base case for recursion (no rates were found).
template<int species, int j>
AMREX_GPU_HOST_DEVICE AMREX_INLINE
constexpr int actual_rhs_term_impl (int count)
{
    return -1;
}

// If we have counted up j nonzero rates, return the
// current rate index. Otherwise, keep counting.
template<int species, int j, int rate, int... rates>
AMREX_GPU_HOST_DEVICE AMREX_INLINE
constexpr int actual_rhs_term_impl (int count)
{
    count += RHS::is_rate_used<species, rate + 1>();

    if (count == j) {
        return rate + 1;
    }
    else {
        return actual_rhs_term_impl<species, j, rates...>(count);
    }
}

// Return the rate corresponding to the j'th contribution to the RHS.
// We obtain this by looping through the rates, and counting each one
// with a nonzero contribution. We stop when we have hit j of them.
template<int species, int j, int... rates>
AMREX_GPU_HOST_DEVICE AMREX_INLINE
constexpr int rhs_term_impl (std::integer_sequence<int, rates...>)
{
    return actual_rhs_term_impl<species, j, rates...>(0);
}

// Implicitly construct an Array1D by expanding the integer sequence.
// Note that the integer sequence is zero-indexed but the terms are
// one-indexed.
template<int species, int... j>
AMREX_GPU_HOST_DEVICE AMREX_INLINE
constexpr Array1D<Real, 1, sizeof...(j)>
make_RHS_Array1D_impl (const burn_t& state, rate_fr_t const& rr, std::integer_sequence<int, j...>)
{
    return {{RHS::rhs_term<species, j+1>(state, rr)...}};
}

// Implicitly construct an Array1D by expanding the integer sequence.
// Note that the integer sequence is zero-indexed but the terms are
// one-indexed.
template<int... species>
AMREX_GPU_HOST_DEVICE AMREX_INLINE
Array1D<Real, 1, NumSpec> species_rhs_impl (const burn_t& state, rate_fr_t const& rr, std::integer_sequence<int, species...>)
{
    return {{RHS::species_rhs_n<species + 1>(state, rr)...}};
}

} // namespace RHS_impl



namespace RHS
{

// Determine if a rate is used in the RHS for a given species
// by seeing if its prefactor is nonzero.
template<int species, int rate>
AMREX_GPU_HOST_DEVICE AMREX_INLINE
constexpr int is_rate_used ()
{
    constexpr rhs_t rhs_data = RHS::rhs_data(rate);

    static_assert(species >= 1 && species <= NumSpec);
    static_assert(rate >= 1 && rate <= Rates::NumRates);

    if (rhs_data.species_A == species ||
        rhs_data.species_B == species ||
        rhs_data.species_C == species ||
        rhs_data.species_D == species ||
        rhs_data.species_E == species ||
        rhs_data.species_F == species) {
        return 1;
    } else {
        return 0;
    }
}

// Count number of nonzero RHS terms.
template<int species>
AMREX_GPU_HOST_DEVICE AMREX_INLINE
constexpr int num_rhs ()
{
    return RHS_impl::num_rhs_impl<species>(std::make_integer_sequence<int, Rates::NumRates>{});
}

// Calculate the j'th RHS term for a given species.
//
// The general form of a reaction is
// n_a A + n_b B + n_c C <-> n_d D + n_e E + n_f F
// for species A, B, C, D, E, and F, where n_a particles of A,
// n_b particles of B, and n_C particles of C are consumed in
// the forward reaction and produced in the reverse reaction.
//
// For a given species, such as species A, the forward reaction
// term is of the form
// -n_A * Y(A)**a * Y(B)**b * Y(C)**c * forward_rate,
// and the reverse reaction term is of the form
//  n_A * Y(D)**d * Y(E)**e * Y(F)**f * reverse_rate.
// Here a, b, and c are reaction-specific exponents which usually,
// but not always, are equal to n_a, n_b, and n_c respectively.
//
// For example, in C12 + He4 <-> O16, species A is C12, species B
// is He4, species D is O16 (the other species are unused). Then
// n_a = n_b = n_d = 1, and a = b = d = 1. In the triple alpha forward
// reaction we have A = He4, D = C12, n_a = 3, a = 3, and n_d = 1.
//
// We assume the reaction rates do not include the identical particle
// factor, so we account for that here by dividing the rate by n!
// Note that we use the exponent to determine this factorial term, not
// the number consumed, because there are some reactions with special handling
// like the Si28 + 7 * He4 <-> Ni56 reaction in iso7, where the number
// of He4 consumed is not directly related to the actual reaction rate.
// In every other case than that, the exponent should be equal to the
// number consumed/produced, so there would be no difference.
//
// If a given reaction uses fewer than three species, we infer
// this by calling its index -1 and then not accessing it
// in the multiplication.
template<int species, int j>
AMREX_GPU_HOST_DEVICE AMREX_INLINE
constexpr Real rhs_term (const burn_t& state, rate_fr_t const& rr)
{
    constexpr int rate = RHS_impl::rhs_term_impl<species, j>(std::make_integer_sequence<int, Rates::NumRates>{});

    constexpr rhs_t rhs_data = RHS::rhs_data(rate);

    // First, compute the Y * rate component of both the forward and
    // reverse reactions, which is the same regardless of which species
    // we're producing or consuming.

    Real forward_term = rr.forward_rates(rate);

    if constexpr (rhs_data.species_A >= 0) {
        Real Y_A = state.xn[rhs_data.species_A-1] * aion_inv[rhs_data.species_A-1];
        constexpr int identical_particle_factor = RHS_impl::factorial(rhs_data.exponent_A);
        forward_term *= std::pow(Y_A, rhs_data.exponent_A) /
                        static_cast<Real>(identical_particle_factor);
    }

    if constexpr (rhs_data.species_B >= 0) {
        Real Y_B = state.xn[rhs_data.species_B-1] * aion_inv[rhs_data.species_B-1];
        constexpr int identical_particle_factor = RHS_impl::factorial(rhs_data.exponent_B);
        forward_term *= std::pow(Y_B, rhs_data.exponent_B) /
                        static_cast<Real>(identical_particle_factor);
    }

    if constexpr (rhs_data.species_C >= 0) {
        Real Y_C = state.xn[rhs_data.species_C-1] * aion_inv[rhs_data.species_C-1];
        constexpr int identical_particle_factor = RHS_impl::factorial(rhs_data.exponent_C);
        forward_term *= std::pow(Y_C, rhs_data.exponent_C) /
                        static_cast<Real>(identical_particle_factor);
    }

    Real reverse_term = rr.reverse_rates(rate);

    if constexpr (rhs_data.species_D >= 0) {
        Real Y_D = state.xn[rhs_data.species_D-1] * aion_inv[rhs_data.species_D-1];
        constexpr int identical_particle_factor = RHS_impl::factorial(rhs_data.exponent_D);
        reverse_term *= std::pow(Y_D, rhs_data.exponent_D) /
                        static_cast<Real>(identical_particle_factor);
    }

    if constexpr (rhs_data.species_E >= 0) {
        Real Y_E = state.xn[rhs_data.species_E-1] * aion_inv[rhs_data.species_E-1];
        constexpr int identical_particle_factor = RHS_impl::factorial(rhs_data.exponent_E);
        reverse_term *= std::pow(Y_E, rhs_data.exponent_E) /
                        static_cast<Real>(identical_particle_factor);
    }

    if constexpr (rhs_data.species_F >= 0) {
        Real Y_F = state.xn[rhs_data.species_F-1] * aion_inv[rhs_data.species_F-1];
        constexpr int identical_particle_factor = RHS_impl::factorial(rhs_data.exponent_F);
        reverse_term *= std::pow(Y_F, rhs_data.exponent_F) /
                        static_cast<Real>(identical_particle_factor);
    }

    // Now compute the total contribution to this species.

    Real term = 0.0_rt;

    if constexpr (rhs_data.species_A == species) {
        term = rhs_data.number_A * (reverse_term - forward_term);
    }
    if constexpr (rhs_data.species_B == species) {
        term = rhs_data.number_B * (reverse_term - forward_term);
    }
    if constexpr (rhs_data.species_C == species) {
        term = rhs_data.number_C * (reverse_term - forward_term);
    }
    if constexpr (rhs_data.species_D == species) {
        term = rhs_data.number_D * (forward_term - reverse_term);
    }
    if constexpr (rhs_data.species_E == species) {
        term = rhs_data.number_E * (forward_term - reverse_term);
    }
    if constexpr (rhs_data.species_F == species) {
        term = rhs_data.number_F * (forward_term - reverse_term);
    }

    return term;
}

// Calculate the set of RHS terms.
template<int species, int N>
AMREX_GPU_HOST_DEVICE AMREX_INLINE
constexpr Array1D<Real, 1, N>
make_RHS_Array1D (const burn_t& state, rate_fr_t const& rr)
{
    return RHS_impl::make_RHS_Array1D_impl<species>(state, rr, std::make_integer_sequence<int, N>{});
}

// Calculate the RHS for a given species by constructing the array of terms
// and then summing them up.
template<int species>
AMREX_GPU_HOST_DEVICE AMREX_INLINE
Real species_rhs_n (const burn_t& state, rate_fr_t const& rr)
{
    constexpr int nrhs = num_rhs<species>();
    Array1D<Real, 1, nrhs> a = make_RHS_Array1D<species, nrhs>(state, rr);

    Real sum = 0.0_rt;
    for (int n = 1; n <= nrhs; ++n) {
        sum += a(n);
    }
    return sum;
}

// Calculate the array of RHS terms over all species.
AMREX_GPU_HOST_DEVICE AMREX_INLINE
Array1D<Real, 1, NumSpec> species_rhs (const burn_t& state, rate_t const& rr)
{
    rate_fr_t rr_fr;

    // Convert old-style rates to new-style
    for (int j = 1; j <= Rates::NumRatesFR; ++j) {
        rr_fr.forward_rates(j) = rr.rates(2 * j - 1);
        rr_fr.reverse_rates(j) = rr.rates(2 * j);
    }

    return RHS_impl::species_rhs_impl(state, rr_fr, std::make_integer_sequence<int, NumSpec>{});
}

// Calculate the energy generation rate, including neutrino losses.
AMREX_GPU_HOST_DEVICE AMREX_INLINE
Real ener_rhs (const burn_t& state, Array1D<Real, 1, NumSpec>& dydt)
{
    using namespace network;

    Real Xdot = 0.0_rt;

    // Sum the mass fraction time derivatives

    for (int i = 1; i <= NumSpec; ++i) {
        Xdot += dydt(i) * mion(i);
    }

    // This is basically e = m c**2

    Real dedt = Xdot * C::Legacy::enuc_conv2;

    // Get the neutrino losses

    Real sneut, dsneutdt, dsneutdd, snuda, snudz;
    sneut5(state.T, state.rho, state.abar, state.zbar, sneut, dsneutdt, dsneutdd, snuda, snudz);

    dedt -= sneut;

    return dedt;
}

// Calculate the Jacobian terms that are derivatives of one species
// with respect to another.
AMREX_GPU_HOST_DEVICE AMREX_INLINE
Array2D<Real, 1, NumSpec, 1, NumSpec> species_jac (const burn_t& state, rate_t const& rr)
{
    rate_fr_t rr_fr;

    // Convert old-style rates to new-style
    for (int j = 1; j <= Rates::NumRatesFR; ++j) {
        rr_fr.forward_rates(j) = rr.rates(2 * j - 1);
        rr_fr.reverse_rates(j) = rr.rates(2 * j);
    }

    Array2D<Real, 1, NumSpec, 1, NumSpec> jac{};

    for (int spec1 = 1; spec1 <= NumSpec; ++spec1) {

        for (int spec2 = 1; spec2 <= NumSpec; ++spec2) {

            // d(spec1) / d(spec2)

            for (int rate = 1; rate <= Rates::NumRatesFR; ++rate) {
                rhs_t rhs_data = RHS::rhs_data(rate);

                // Compute the forward term. It only has a contribution if
                // the species we're taking the derivative with respect to
                // (spec2) is one of (A, B, C). For the species that is spec2,
                // we take the derivative by multiplying by the current exponent
                // and then decrementing the exponent in the term.

                Real forward_term = 0.0_rt;

                if (spec2 == rhs_data.species_A || spec2 == rhs_data.species_B || spec2 == rhs_data.species_C) {

                    forward_term = rr_fr.forward_rates(rate);

                    if (rhs_data.species_A >= 0) {
                        Real Y_A = state.xn[rhs_data.species_A-1] * aion_inv[rhs_data.species_A-1];

                        int exponent = rhs_data.exponent_A;

                        int identical_particle_factor = RHS_impl::factorial(exponent);

                        if (spec2 == rhs_data.species_A) {
                            forward_term *= exponent;
                            --exponent;
                        }

                        forward_term *= std::pow(Y_A, exponent) /
                                        static_cast<Real>(identical_particle_factor);
                    }

                    if (rhs_data.species_B >= 0) {
                        Real Y_B = state.xn[rhs_data.species_B-1] * aion_inv[rhs_data.species_B-1];

                        int exponent = rhs_data.exponent_B;

                        int identical_particle_factor = RHS_impl::factorial(exponent);

                        if (spec2 == rhs_data.species_B) {
                            forward_term *= exponent;
                            --exponent;
                        }

                        forward_term *= std::pow(Y_B, exponent) /
                                        static_cast<Real>(identical_particle_factor);
                    }

                    if (rhs_data.species_C >= 0) {
                        Real Y_C = state.xn[rhs_data.species_C-1] * aion_inv[rhs_data.species_C-1];

                        int exponent = rhs_data.exponent_C;

                        int identical_particle_factor = RHS_impl::factorial(exponent);

                        if (spec2 == rhs_data.species_C) {
                            forward_term *= exponent;
                            --exponent;
                        }
                        forward_term *= std::pow(Y_C, exponent) /
                                        static_cast<Real>(identical_particle_factor);
                    }

                }

                // Do the same for the reverse term, which only has a contribution
                // if spec2 is one of (D, E, F).

                Real reverse_term = 0.0_rt;

                if (spec2 == rhs_data.species_D || spec2 == rhs_data.species_E || spec2 == rhs_data.species_F) {

                    reverse_term = rr_fr.reverse_rates(rate);

                    if (rhs_data.species_D >= 0) {
                        Real Y_D = state.xn[rhs_data.species_D-1] * aion_inv[rhs_data.species_D-1];

                        int exponent = rhs_data.exponent_D;

                        int identical_particle_factor = RHS_impl::factorial(exponent);

                        if (spec2 == rhs_data.species_D) {
                            reverse_term *= exponent;
                            --exponent;
                        }

                        reverse_term *= std::pow(Y_D, exponent) /
                                        static_cast<Real>(identical_particle_factor);
                    }

                    if (rhs_data.species_E >= 0) {
                        Real Y_E = state.xn[rhs_data.species_E-1] * aion_inv[rhs_data.species_E-1];

                        int exponent = rhs_data.exponent_E;

                        int identical_particle_factor = RHS_impl::factorial(exponent);

                        if (spec2 == rhs_data.species_E) {
                            reverse_term *= exponent;
                            --exponent;
                        }

                        reverse_term *= std::pow(Y_E, exponent) /
                                        static_cast<Real>(identical_particle_factor);
                    }

                    if (rhs_data.species_F >= 0) {
                        Real Y_F = state.xn[rhs_data.species_F-1] * aion_inv[rhs_data.species_F-1];

                        int exponent = rhs_data.exponent_F;

                        int identical_particle_factor = RHS_impl::factorial(exponent);

                        if (spec2 == rhs_data.species_F) {
                            reverse_term *= exponent;
                            --exponent;
                        }

                        reverse_term *= std::pow(Y_F, exponent) /
                                        static_cast<Real>(identical_particle_factor);
                    }

                }

                // Now compute the total contribution to this species.

                Real term = 0.0_rt;

                if (rhs_data.species_A == spec1) {
                    term = rhs_data.number_A * (reverse_term - forward_term);
                }
                if (rhs_data.species_B == spec1) {
                    term = rhs_data.number_B * (reverse_term - forward_term);
                }
                if (rhs_data.species_C == spec1) {
                    term = rhs_data.number_C * (reverse_term - forward_term);
                }
                if (rhs_data.species_D == spec1) {
                    term = rhs_data.number_D * (forward_term - reverse_term);
                }
                if (rhs_data.species_E == spec1) {
                    term = rhs_data.number_E * (forward_term - reverse_term);
                }
                if (rhs_data.species_F == spec1) {
                    term = rhs_data.number_F * (forward_term - reverse_term);
                }

                jac(spec1, spec2) += term;

            }

        }

    }

    return jac;
}


AMREX_GPU_HOST_DEVICE AMREX_INLINE
Array1D<Real, 1, 6> equilibrium_state (const rhs_t& rhs_data, const burn_t& state,
                                       Real forward_rate, Real reverse_rate)
{
    int n_A = rhs_data.number_A;
    int n_B = rhs_data.number_B;
    int n_C = rhs_data.number_C;
    int n_D = rhs_data.number_D;
    int n_E = rhs_data.number_E;
    int n_F = rhs_data.number_F;

    int e_A = rhs_data.exponent_A;
    int e_B = rhs_data.exponent_B;
    int e_C = rhs_data.exponent_C;
    int e_D = rhs_data.exponent_D;
    int e_E = rhs_data.exponent_E;
    int e_F = rhs_data.exponent_F;

    Real Y_A = 0.0_rt;
    if (rhs_data.species_A >= 0) {
        Y_A = state.xn[rhs_data.species_A-1] * aion_inv[rhs_data.species_A-1];
    }

    Real Y_B = 0.0_rt;
    if (rhs_data.species_B >= 0) {
        Y_B = state.xn[rhs_data.species_B-1] * aion_inv[rhs_data.species_B-1];
    }

    Real Y_C = 0.0_rt;
    if (rhs_data.species_C >= 0) {
        Y_C = state.xn[rhs_data.species_C-1] * aion_inv[rhs_data.species_C-1];
    }

    Real Y_D = 0.0_rt;
    if (rhs_data.species_D >= 0) {
        Y_D = state.xn[rhs_data.species_D-1] * aion_inv[rhs_data.species_D-1];
    }

    Real Y_E = 0.0_rt;
    if (rhs_data.species_E >= 0) {
        Y_E = state.xn[rhs_data.species_E-1] * aion_inv[rhs_data.species_E-1];
    }

    Real Y_F = 0.0_rt;
    if (rhs_data.species_F >= 0) {
        Y_F = state.xn[rhs_data.species_F-1] * aion_inv[rhs_data.species_F-1];
    }

    Real k_f = forward_rate;
    Real k_r = reverse_rate;

    Real Y_A_eq = 0.0_rt;
    Real Y_B_eq = 0.0_rt;
    Real Y_C_eq = 0.0_rt;
    Real Y_D_eq = 0.0_rt;
    Real Y_E_eq = 0.0_rt;
    Real Y_F_eq = 0.0_rt;

    // The general form of our reactions has forward reactions of the form:
    // N * (Y_A**e_A * Y_B**e_B * Y_C**e_C * k_f)
    // and reverse reactions of the form:
    // N * (Y_D**e_D * Y_B**e_E * Y_F**e_F * k_r)
    // where N is the number of particles consumed/produced, Y is the molar
    // fraction, and e is an integer exponent coefficient, k_f is the forward
    // rate, and k_r is the reverse rate.
    //
    // If the reaction is in equilibrium, we will satisfy
    // Y_A**e_A * Y_B**e_B * Y_C**e_C * k_f = Y_D**e_D * Y_B**e_E * Y_F**e_F * k_r
    // We expect that often this relation will not be satisfied exactly,
    // but sufficiently close that we will likely achieve equilibrium soon.
    //
    // So, given a state and a set of rates, we want to determine the corresponding
    // equilibrium solution. This involves solving the t -> infinity limit of the ODE.
    // For certain common values of the exponents this solution is easy to write down
    // analytically. We will solve for these cases specifically and ignore other cases.

    if (e_A == 1 && e_B == 1 && e_C == 0 && e_D == 1 && e_E == 0 && e_F == 0 &&
        n_A == 1 && n_B == 1 && n_C == 0 && n_D == 1 && n_E == 0 && n_F == 0)
    {
        // This reaction looks like
        // A + B --> D
        // and the corresponding ODE for species A is
        // dY_A / dt = -k_f * Y_A * Y_B + k_r * Y_D
        //
        // The conserved scalars are
        // c_1 = Y_A - Y_B
        // c_2 = Y_A + Y_D
        //
        // The species A ODE in terms of only Y_A is
        // dY_A / dt = a * Y_A**2 + b * Y_A + c
        // where
        // a = -k_f
        // b = c_1 * k_f - k_r
        // c = k_r * c_2
        //
        // The equilibrium solution for species A is
        // Y_A_eq = (-1 / 2a) * (b + sqrt(b * b - 4 * a * c))
        //
        // The equilibrium solutions for the other species are
        // Y_B_eq = Y_A_eq - c_1
        // Y_D_eq = c_2 - Y_A_eq

        Real c_1 = Y_A - Y_B;
        Real c_2 = Y_A + Y_D;

        Real a = -k_f;
        Real b = c_1 * k_f - k_r;
        Real c = k_r * c_2;

        Y_A_eq = (-0.5_rt / a) * (b + std::sqrt(b * b - 4.0_rt * a * c));
        Y_B_eq = Y_A_eq - c_1;
        Y_D_eq = c_2 - Y_A_eq;
    }

    Real X_A_eq = 0.0_rt;
    if (rhs_data.species_A >= 0) {
        X_A_eq = Y_A_eq * aion[rhs_data.species_A-1];
    }

    Real X_B_eq = 0.0_rt;
    if (rhs_data.species_B >= 0) {
        X_B_eq = Y_B_eq * aion[rhs_data.species_B-1];
    }

    Real X_C_eq = 0.0_rt;
    if (rhs_data.species_C >= 0) {
        X_C_eq = Y_C_eq * aion[rhs_data.species_C-1];
    }

    Real X_D_eq = 0.0_rt;
    if (rhs_data.species_D >= 0) {
        X_D_eq = Y_D_eq * aion[rhs_data.species_D-1];
    }

    Real X_E_eq = 0.0_rt;
    if (rhs_data.species_E >= 0) {
        X_E_eq = Y_E_eq * aion[rhs_data.species_E-1];
    }

    Real X_F_eq = 0.0_rt;
    if (rhs_data.species_F >= 0) {
        X_F_eq = Y_F_eq * aion[rhs_data.species_F-1];
    }

    Array1D<Real, 1, 6> eq_state = {X_A_eq, X_B_eq, X_C_eq, X_D_eq, X_E_eq, X_F_eq};

    return eq_state;
}

AMREX_GPU_HOST_DEVICE AMREX_INLINE
bool is_reaction_in_equilibrium (const rhs_t& rhs_data, const burn_t& state,
                                 const Array1D<Real, 1, 6>& equilibrium_state)
{
    // Check whether all species are within a given certain tolerance of the
    // corresponding equilibrium mass fractions.

    const Real equilibrium_tolerance = 0.005_rt;

    bool equilibrium = true;

    if (rhs_data.species_A >= 0) {
        Real X = state.xn[rhs_data.species_A - 1];
        Real X_eq = equilibrium_state(1);

        if (std::abs(X - X_eq) > equilibrium_tolerance * X || X < atol_spec) {
            equilibrium = false;
        }
    }

    if (rhs_data.species_B >= 0) {
        Real X = state.xn[rhs_data.species_B - 1];
        Real X_eq = equilibrium_state(2);

        if (std::abs(X - X_eq) > equilibrium_tolerance * X || X < atol_spec) {
            equilibrium = false;
        }
    }

    if (rhs_data.species_C >= 0) {
        Real X = state.xn[rhs_data.species_C - 1];
        Real X_eq = equilibrium_state(3);

        if (std::abs(X - X_eq) > equilibrium_tolerance * X || X < atol_spec) {
            equilibrium = false;
        }
    }

    if (rhs_data.species_D >= 0) {
        Real X = state.xn[rhs_data.species_D - 1];
        Real X_eq = equilibrium_state(4);

        if (std::abs(X - X_eq) > equilibrium_tolerance * X || X < atol_spec) {
            equilibrium = false;
        }
    }

    if (rhs_data.species_E >= 0) {
        Real X = state.xn[rhs_data.species_E - 1];
        Real X_eq = equilibrium_state(5);

        if (std::abs(X - X_eq) > equilibrium_tolerance * X || X < atol_spec) {
            equilibrium = false;
        }
    }

    if (rhs_data.species_F >= 0) {
        Real X = state.xn[rhs_data.species_F - 1];
        Real X_eq = equilibrium_state(6);

        if (std::abs(X - X_eq) > equilibrium_tolerance * X || X < atol_spec) {
            equilibrium = false;
        }
    }

    return equilibrium;
}

} // namespace RHS

#endif
