! This is a multi-gamma EOS.  Each species can have its own gamma, but
! otherwise, they all act as an ideal gas.

! Note: at the moment, it is not clear what the proper expression for
! a multi-gamma entropy should be, so do not rely on the entropy.

module actual_eos_module

  use amrex_error_module
  use amrex_constants_module
  use amrex_fort_module, only : rt => amrex_real

  use network, only: nspec, aion, zion
  use eos_type_module

  implicit none

  character (len=64), public :: eos_name = "multigamma"

  double precision, allocatable, save :: gammas(:)

#ifdef AMREX_USE_CUDA
  attributes(managed) :: gammas
#endif

contains


  function eos_supports_input_type(input) result(supported)

    use eos_type_module

    implicit none

    integer, intent(in) :: input
    logical :: supported

    if (input == eos_input_rt .or. &
        input == eos_input_rp .or. &
        input == eos_input_rh .or. &
        input == eos_input_re .or. &
        input == eos_input_tp .or. &
        input == eos_input_ph) then

       supported = .true.

    else

       supported = .false.

    endif

  end function eos_supports_input_type



  subroutine actual_eos_init

    use extern_probin_module, only: eos_gamma_default, &
                                    species_a_name, species_a_gamma, &
                                    species_b_name, species_b_gamma, &
                                    species_c_name, species_c_gamma
    use network, only: network_species_index

    implicit none

    integer :: idx

    ! Set the gammas for the species -- we have some runtime parameters
    ! that can override the default gammas for a few named species.
    allocate(gammas(nspec))
    gammas(:) = eos_gamma_default

#ifndef AMREX_USE_GPU
    print *, ""
    print *, "In actual_eos_init, species, gamma: "
    print *, "species a: ", trim(species_a_name), species_a_gamma
    print *, "species b: ", trim(species_b_name), species_b_gamma
    print *, "species c: ", trim(species_c_name), species_c_gamma
    print *, ""
#endif

    idx = network_species_index(species_a_name)
    if (idx > 0) gammas(idx) = species_a_gamma

    idx = network_species_index(species_b_name)
    if (idx > 0) gammas(idx) = species_b_gamma

    idx = network_species_index(species_c_name)
    if (idx > 0) gammas(idx) = species_c_gamma

  end subroutine actual_eos_init



  subroutine actual_eos(input, state)

    use fundamental_constants_module, only: k_B, n_A, hbar

    implicit none

    integer,      intent(in   ) :: input
    type (eos_t), intent(inout) :: state

    ! Local variables
    double precision :: sumY_gm1, sumYg_gm1
    double precision :: dens, temp

    ! Get the mass of a nucleon from Avogadro's number.
    double precision, parameter :: m_nucleon = ONE / n_A

    !$gpu

    ! Special gamma factors

    sumY_gm1  = sum(state % xn(:)/(aion(:)*(gammas(:)-ONE)))
    sumYg_gm1 = sum(state % xn(:)*gammas(:)/(aion(:)*(gammas(:)-ONE)))

    !-------------------------------------------------------------------------
    ! For all EOS input modes EXCEPT eos_input_rt, first compute dens
    ! and temp as needed from the inputs.
    !-------------------------------------------------------------------------

    select case (input)

    case (eos_input_rt)

       ! dens, temp and xmass are inputs

       ! We don't need to do anything here
       temp = state % T
       dens = state % rho


    case (eos_input_rh)

       ! dens, enthalpy, and xmass are inputs

       ! Solve for the temperature:
       ! h = e + p/rho = (p/rho)*[1 + 1/(gamma-1)] = (p/rho)*gamma/(gamma-1)
       dens = state % rho
       temp = (state % h * m_nucleon / k_B)/sumYg_gm1


    case (eos_input_tp)

       ! temp, pres, and xmass are inputs

       ! Solve for the density:
       ! p = rho k T / (abar m_nucleon)
       dens = state % p * state % abar * m_nucleon / (k_B * state % T)
       temp = state % T


    case (eos_input_rp)

       ! dens, pres, and xmass are inputs

       ! Solve for the temperature:
       ! p = rho k T / (mu m_nucleon)
       dens = state % rho
       temp = state % p * state % abar * m_nucleon / (k_B * state % rho)


    case (eos_input_re)

       ! dens, energy, and xmass are inputs

       ! Solve for the temperature
       ! e = k T / [(mu m_nucleon)*(gamma-1)]
       dens = state % rho
       temp = state % e * m_nucleon / (k_B * sumY_gm1)


    case (eos_input_ps)

       ! pressure entropy, and xmass are inputs
#ifndef AMREX_USE_GPU
       call amrex_error("eos_input_ps is not supported")
#endif

    case (eos_input_ph)

       ! pressure, enthalpy and xmass are inputs

       ! Solve for temperature and density
       dens = state % p * state % abar / state % h * sumYg_gm1
       temp = state % p * state % abar * m_nucleon / (k_B * dens)


    case (eos_input_th)

       ! temperature, enthalpy and xmass are inputs
#ifndef AMREX_USE_GPU
       call amrex_error("eos_input_th is not supported")
#endif

    case default

#ifndef AMREX_USE_GPU
       call amrex_error("EOS: invalid input.")
#endif

    end select

    !-------------------------------------------------------------------------
    ! Now we have the density and temperature (and mass fractions /
    ! mu), regardless of the inputs.
    !-------------------------------------------------------------------------

    state % T   = temp
    state % rho = dens

    ! Compute the pressure simply from the ideal gas law, and the
    ! specific internal energy using the gamma-law EOS relation.
    state % p = dens*k_B*temp/(state % abar*m_nucleon)
    state % e = k_B*temp*sumY_gm1/m_nucleon

    ! Enthalpy is h = e + p/rho
    state % h = state % e + state % p / dens

    ! entropy (per gram) -- this is wrong. Not sure what the expression
    ! is for a multigamma gas
    state % s = (k_B/(state%abar*m_nucleon))*(2.5_rt + &
         log( ( (state%abar*m_nucleon)**2.5/dens )*(k_B*temp)**1.5_rt / (TWO*M_PI*hbar*hbar)**1.5_rt ) )

    ! Compute the thermodynamic derivatives and specific heats
    state % dpdT = state % p / temp
    state % dpdr = state % p / dens
    state % dedT = state % e / temp
    state % dedr = ZERO
    state % dsdT = ZERO
    state % dsdr = ZERO
    state % dhdT = state % dedT + state % dpdT / dens
    state % dhdr = ZERO

    state % cv = state % dedT
    state % cp = state % h / state % T

    state % gam1 = state % cp / state % cv

    state % dpdr_e = state % dpdr - state % dpdT * state % dedr / state % dedT
    state % dpde   = state % dpdT / state % dedT


#ifdef EXTRA_THERMO
    ! These need to be worked out.
    state % dpdA = ZERO
    state % dedA = ZERO

    state % dpdZ = ZERO
    state % dedZ = ZERO

    ! Composition derivatives

    state % dpdX(:) = state % rho * k_B * state % T / (m_nucleon * aion(:))
    state % dedX(:) = k_B * state % T / (m_nucleon * aion(:) * (gammas(:) - ONE))

    state % dhdX(:) = state % dedX(:) + (state % p / state % rho**2 - state % dedr) &
         *  state % dpdX(:) / state % dpdr
#endif

    ! Sound speed
    state % cs = sqrt(state % gam1 * state % p / dens)

  end subroutine actual_eos

  subroutine actual_eos_finalize

    implicit none

    deallocate(gammas)

  end subroutine actual_eos_finalize

end module actual_eos_module
