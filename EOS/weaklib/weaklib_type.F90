module weaklib_type_module

  use amrex_fort_module, only: rt => amrex_real
  use eos_type_module
  use PhysicalConstantsModule, only: BoltzmannConstantMKS
  use UnitsModule, only: Joule, Erg, MeV, Gram, AtomicMassUnit

  implicit none

  public

  real(rt), parameter :: BoltzmannConstantCGS = BoltzmannConstantMKS * Joule / Erg
  real(rt), parameter :: AtomicMassUnitCGS = AtomicMassUnit / Gram

  type :: weaklib_eos_t
     real(rt) :: density
     real(rt) :: temperature
     real(rt) :: electron_fraction

     real(rt) :: pressure
     real(rt) :: entropy_per_baryon
     real(rt) :: specific_internal_energy

     real(rt) :: electron_chemical_potential
     real(rt) :: proton_chemical_potential
     real(rt) :: neutron_chemical_potential

     real(rt) :: proton_mass_fraction
     real(rt) :: neutron_mass_fraction
     real(rt) :: alpha_mass_fraction
     real(rt) :: heavy_mass_fraction

     real(rt) :: gamma_one
     real(rt) :: speed_of_sound
  end type weaklib_eos_t

contains

  subroutine eos_to_weaklib(eos_state, weaklib_state)

    use amrex_constants_module, only: ZERO

    implicit none

    type (eos_t), intent(inout) :: eos_state
    type (weaklib_eos_t), intent(out) :: weaklib_state

    call convert_to_table_format(eos_state)

    weaklib_state % density = eos_state % rho
    weaklib_state % temperature = eos_state % T
    weaklib_state % electron_fraction = eos_state % y_e

    weaklib_state % pressure = eos_state % p
    weaklib_state % entropy_per_baryon = ZERO
    weaklib_state % specific_internal_energy = eos_state % e

    weaklib_state % electron_chemical_potential = ZERO
    weaklib_state % proton_chemical_potential = ZERO
    weaklib_state % neutron_chemical_potential = ZERO

    weaklib_state % proton_mass_fraction = ZERO
    weaklib_state % neutron_mass_fraction = ZERO
    weaklib_state % alpha_mass_fraction = ZERO
    weaklib_state % heavy_mass_fraction = ZERO

    weaklib_state % gamma_one = ZERO

  end subroutine eos_to_weaklib


  subroutine weaklib_to_eos(weaklib_state, eos_state)

    implicit none

    type (weaklib_eos_t), intent(in) :: weaklib_state

    ! eos_state is intent(inout) so we don't reset
    ! quantities that are calculated outside this EOS
    type (eos_t), intent(inout) :: eos_state

    eos_state % rho = weaklib_state % density
    eos_state % T   = weaklib_state % temperature
    eos_state % y_e  = weaklib_state % electron_fraction

    eos_state % p = weaklib_state % pressure
    eos_state % s = weaklib_state % entropy_per_baryon
    eos_state % e = weaklib_state % specific_internal_energy

    eos_state % electron_chemical_potential = weaklib_state % electron_chemical_potential
    eos_state % proton_chemical_potential = weaklib_state % proton_chemical_potential
    eos_state % neutron_chemical_potential = weaklib_state % neutron_chemical_potential

    eos_state % gam1 = weaklib_state % gamma_one

    call convert_from_table_format(eos_state)

    ! construct enthalpy
    eos_state % h = eos_state % e + eos_state % p/eos_state % rho

    ! construct sound speed using the first adiabatic index
    eos_state % cs = sqrt(eos_state % gam1 * eos_state % p / eos_state % rho)

    ! construct electron number density per unit volume
    eos_state % xne = eos_state % y_e * eos_state % rho / AtomicMassUnitCGS

  end subroutine weaklib_to_eos


  ! Unit Conversion Helpers:
  !
  ! the weaklib table interface uses
  ! entropy in units of k_B / baryon
  subroutine convert_to_table_format(state)

    implicit none

    type(eos_t), intent(inout) :: state

    state % s = state % s * (AtomicMassUnitCGS/BoltzmannConstantCGS)

  end subroutine convert_to_table_format


  subroutine convert_from_table_format(state)

    implicit none

    type(eos_t), intent(inout) :: state

    state % s = state % s / (AtomicMassUnitCGS/BoltzmannConstantCGS)

    state % electron_chemical_potential = state % electron_chemical_potential * MeV
    state % proton_chemical_potential = state % proton_chemical_potential * MeV
    state % neutron_chemical_potential = state % neutron_chemical_potential * MeV

  end subroutine convert_from_table_format

end module weaklib_type_module
