! This is a constant gamma equation of state, using an ideal gas.
!
! This a simplified version of the more general eos_gamma_general.
!

module actual_eos_module

  use amrex_error_module
  use amrex_constants_module
  use eos_type_module

  implicit none

  character (len=64), public :: eos_name = "gamma_law"  
  
  double precision, allocatable, save :: gamma_const

#ifdef AMREX_USE_CUDA
  attributes(managed) :: gamma_const
#endif

contains


  function eos_supports_input_type(input) result(supported)

    use eos_type_module

    implicit none

    integer, intent(in) :: input
    logical :: supported

    if (input == eos_input_rt .or. &
        input == eos_input_rp .or. &
        input == eos_input_re) then

       supported = .true.

    else

       supported = .false.

    endif

  end function eos_supports_input_type



  subroutine actual_eos_init

    use extern_probin_module, only: eos_gamma

    implicit none

    allocate(gamma_const)
    
    ! constant ratio of specific heats
    if (eos_gamma .gt. 0.d0) then
       gamma_const = eos_gamma
    else
       gamma_const = FIVE3RD
    end if

  end subroutine actual_eos_init



  subroutine actual_eos(input, state)

    use fundamental_constants_module, only: k_B, n_A

    implicit none

    integer,      intent(in   ) :: input
    type (eos_t), intent(inout) :: state

    double precision, parameter :: R = k_B*n_A

    double precision :: poverrho

    !$gpu

    ! Calculate mu.

    ! xxxxxxx hack!
    ! The only difference between this EOS and gamma-law
    state % mu = one / state % aux(2)
    
    select case (input)

    case (eos_input_rt)

       ! dens, temp and xmass are inputs
       state % cv = R / (state % mu * (gamma_const-ONE)) 
       state % e = state % cv * state % T
       state % p = (gamma_const-ONE) * state % rho * state % e
       state % gam1 = gamma_const

    case (eos_input_rh)

       ! dens, enthalpy, and xmass are inputs
#if !(defined(ACC) || defined(CUDA))
       call amrex_error('EOS: eos_input_rh is not supported in this EOS.')
#endif
    case (eos_input_tp)

       ! temp, pres, and xmass are inputs
#if !(defined(ACC) || defined(CUDA))
       call amrex_error('EOS: eos_input_tp is not supported in this EOS.')
#endif
    case (eos_input_rp)

       ! dens, pres, and xmass are inputs

       poverrho = state % p / state % rho
       state % T = poverrho * state % mu * (ONE/R)
       state % e = poverrho * (ONE/(gamma_const-ONE))
       state % gam1 = gamma_const

    case (eos_input_re)

       ! dens, energy, and xmass are inputs

       poverrho = (gamma_const - ONE) * state % e

       state % p = poverrho * state % rho
       state % T = poverrho * state % mu * (ONE/R)
       state % gam1 = gamma_const
       
       ! sound speed
       state % cs = sqrt(gamma_const * poverrho)

       state % dpdr_e = poverrho
       state % dpde = (gamma_const-ONE) * state % rho

       ! Try to avoid the expensive log function.  Since we don't need entropy 
       ! in hydro solver, set it to an invalid but "nice" value for the plotfile.
       state % s = ONE  

    case (eos_input_ps)

       ! pressure entropy, and xmass are inputs

       call amrex_error('EOS: eos_input_ps is not supported in this EOS.')
       
    case (eos_input_ph)

       ! pressure, enthalpy and xmass are inputs
#if !(defined(ACC) || defined(CUDA))
       call amrex_error('EOS: eos_input_ph is not supported in this EOS.')
#endif
    case (eos_input_th)

       ! temperature, enthalpy and xmass are inputs

       ! This system is underconstrained.
#if !(defined(ACC) || defined(CUDA))       
       call amrex_error('EOS: eos_input_th is not a valid input for the gamma law EOS.')
#endif
    case default
#if !(defined(ACC) || defined(CUDA))       
       call amrex_error('EOS: invalid input.')
#endif       
    end select
    
  end subroutine actual_eos

  subroutine actual_eos_finalize                                                                    
                                                                                                    
    implicit none

    if (allocated(gamma_const)) then
       deallocate(gamma_const)
    endif
  end subroutine actual_eos_finalize

end module actual_eos_module
