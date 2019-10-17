! This is a constant gamma equation of state, using an ideal gas.
!
! The gas may either be completely ionized or completely neutral.
!
! The ratio of specific heats (gamma) is allowed to vary.  NOTE: the
! expression for entropy is only valid for an ideal MONATOMIC gas
! (gamma = 5/3).  

module actual_eos_module

  use amrex_error_module
  use amrex_fort_module, only : rt => amrex_real
  use amrex_constants_module
  use network, only: nspec, aion, aion_inv, zion
  use eos_type_module

  implicit none

  character (len=64), public :: eos_name = "gamma_law_general"  

  double precision, allocatable, save :: gamma_const

  logical, allocatable, save :: assume_neutral

  !$acc declare create(gamma_const, assume_neutral)

#ifdef AMREX_USE_CUDA
  attributes(managed) :: gamma_const, assume_neutral
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
        input == eos_input_ps .or. &
        input == eos_input_ph) then

       supported = .true.

    else

       supported = .false.

    endif

  end function eos_supports_input_type



  subroutine actual_eos_init

    use extern_probin_module, only: eos_gamma, eos_assume_neutral

    implicit none

    allocate(gamma_const)
    allocate(assume_neutral)
 
    ! constant ratio of specific heats
    if (eos_gamma .gt. 0.d0) then
       gamma_const = eos_gamma
    else
       call amrex_error("gamma_const cannot be < 0")
    end if

    assume_neutral = eos_assume_neutral

    !$acc update device(gamma_const, eos_assume_neutral)
    
  end subroutine actual_eos_init



  subroutine actual_eos(input, state)

    !$acc routine seq

    use fundamental_constants_module, only: k_B, n_A, hbar

    implicit none

    integer,      intent(in   ) :: input
    type (eos_t), intent(inout) :: state

    ! Get the mass of a nucleon from Avogadro's number.
    double precision, parameter :: m_nucleon = ONE / n_A
    double precision, parameter :: fac = ONE / (TWO*M_PI*hbar*hbar)**1.5d0

    double precision :: Tinv, rhoinv

    !$gpu

    ! Calculate mu.
    
    if (assume_neutral) then
       state % mu = state % abar
    else
       state % mu = ONE / sum( (ONE + zion(:)) * state % xn(:) * aion_inv(:) )
    endif
    
    !-------------------------------------------------------------------------
    ! For all EOS input modes EXCEPT eos_input_rt, first compute dens
    ! and temp as needed from the inputs.
    !-------------------------------------------------------------------------

    select case (input)

    case (eos_input_rt)

       ! dens, temp and xmass are inputs

       ! We don't need to do anything here
       continue

    case (eos_input_rh)

       ! dens, enthalpy, and xmass are inputs

       ! Solve for the temperature:
       ! h = e + p/rho = (p/rho)*[1 + 1/(gamma-1)] = (p/rho)*gamma/(gamma-1)

       state % T = (state % h * state % mu * m_nucleon / k_B)*(gamma_const - ONE)/gamma_const

    case (eos_input_tp)

       ! temp, pres, and xmass are inputs
       
       ! Solve for the density:
       ! p = rho k T / (mu m_nucleon)

       state % rho =  state % p * state % mu * m_nucleon / (k_B * state % T)

    case (eos_input_rp)

       ! dens, pres, and xmass are inputs

       ! Solve for the temperature:
       ! p = rho k T / (mu m_nucleon)

       state % T = state % p * state % mu * m_nucleon / (k_B * state % rho)

    case (eos_input_re)

       ! dens, energy, and xmass are inputs
       
       ! Solve for the temperature
       ! e = k T / [(mu m_nucleon)*(gamma-1)]

       state % T   = state % e * state % mu * m_nucleon * (gamma_const-ONE) / k_B

    case (eos_input_ps)

       ! pressure, entropy, and xmass are inputs
       
       ! Solve for the temperature
       ! Invert Sackur-Tetrode eqn (below) using 
       ! rho = p mu m_nucleon / (k T)

       state % T  = state % p**(TWO/FIVE) * &
                    ( TWO*M_PI*hbar*hbar/(state % mu*m_nucleon) )**(THREE/FIVE) * &
                    dexp(TWO*state % mu*m_nucleon*state % s/(FIVE*k_B) - ONE) / k_B

       ! Solve for the density
       ! rho = p mu m_nucleon / (k T)
       
       state % rho =  state % p * state % mu * m_nucleon / (k_B * state % T)

    case (eos_input_ph)

       ! pressure, enthalpy and xmass are inputs
       
       ! Solve for temperature and density

       state % rho = state % p / state % h * gamma_const / (gamma_const - ONE)
       state % T   = state % p * state % mu * m_nucleon / (k_B * state % rho)

    case (eos_input_th)

       ! temperature, enthalpy and xmass are inputs

       ! This system is underconstrained.

#ifndef AMREX_USE_GPU
       call amrex_error('EOS: eos_input_th is not a valid input for the gamma law EOS.')
#endif

    case default

#ifndef AMREX_USE_GPU
       call amrex_error('EOS: invalid input.')
#endif
       
    end select
    
    !-------------------------------------------------------------------------
    ! Now we have the density and temperature (and mass fractions /
    ! mu), regardless of the inputs.
    !-------------------------------------------------------------------------
           
    Tinv = ONE / state % T
    rhoinv = ONE / state % rho

    ! Compute the pressure simply from the ideal gas law, and the
    ! specific internal energy using the gamma-law EOS relation.
    state % p = state % rho * state % T * (k_B / (state % mu*m_nucleon))
    state % e = state % p/(gamma_const - ONE) * rhoinv

    ! enthalpy is h = e + p/rho
    state % h = state % e + state % p * rhoinv

    ! entropy (per gram) of an ideal monoatomic gas (the Sackur-Tetrode equation)
    ! NOTE: this expression is only valid for gamma = 5/3.
    state % s = (k_B/(state % mu*m_nucleon))*(2.5_rt + &
         log( ( (state % mu*m_nucleon)**2.5_rt * rhoinv )*(k_B * state % T)**1.5_rt * fac ) )

    ! Compute the thermodynamic derivatives and specific heats 
    state % dpdT = state % p * Tinv
    state % dpdr = state % p * rhoinv
    state % dedT = state % e * Tinv
    state % dedr = ZERO
    state % dsdT = 1.5_rt * (k_B / (state % mu * m_nucleon)) * Tinv
    state % dsdr = - (k_B / (state % mu * m_nucleon)) * rhoinv
    state % dhdT = state % dedT + state % dpdT * rhoinv
    state % dhdr = ZERO

    state % cv = state % dedT
    state % cp = gamma_const * state % cv

    state % gam1 = gamma_const

    state % dpdr_e = state % dpdr - state % dpdT * state % dedr * (ONE/state % dedT)
    state % dpde   = state % dpdT * (ONE/ state % dedT)

    ! sound speed
    state % cs = sqrt(gamma_const * state % p * rhoinv)

#ifdef EXTRA_THERMO
    state % dpdA = - state % p * (ONE/state % abar)
    state % dedA = - state % e * (ONE/state % abar)

    if (assume_neutral) then
      state % dpdZ = ZERO
      state % dedZ = ZERO
    else
      state % dpdZ = state % p * (ONE/(ONE + state % zbar))
      state % dedZ = state % e * (ONE/(ONE + state % zbar))
    endif
#endif

  end subroutine actual_eos

  subroutine actual_eos_finalize
    
    implicit none

    deallocate(gamma_const)
    deallocate(assume_neutral)
  
  end subroutine actual_eos_finalize

end module actual_eos_module
