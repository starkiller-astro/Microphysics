! This is the equation of state for zero-temperature white dwarf 
! matter composed of degenerate electrons:
! P = A ( x * (2x**2 - 3)(x**2 + 1)**1/2 + 3 sinh**-1(x) )
! 
! where rho = B x**3 and the constants are given by:
!
! A = pi m_e**4 c**5 / (3 h**3) = 6.0 x 10^22 dyne cm**-2
! B = 8 pi m_e**3 c**3 mu_e m_p  / (3 h**3) = 9.8 x 10^5 mu_e g cm**-3
!
! The equation of state comes from Chandrasekhar (1935), and the enthalpy
! is calculated by Hachisu (1986):
!
! h = (8A / B) (1 + x**2)**(1/2)
!
! The internal energy is calculated using the standard relation:
! 
! h = e + P / rho

module actual_eos_module

  use amrex_error_module
  use amrex_constants_module
  use network, only: nspec, aion, zion
  use eos_type_module

  implicit none

  character (len=64), public :: eos_name = "ztwd"
  
  double precision, private :: A, B, B2
  double precision, parameter, private :: iter_tol = 1.d-10
  integer,          parameter, private :: max_iter = 1000
  !$OMP THREADPRIVATE(B)

  private :: enthalpy, pressure

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
        input == eos_input_th .or. &
        input == eos_input_ps .or. &
        input == eos_input_ph) then

       supported = .true.

    else

       supported = .false.

    endif

  end function eos_supports_input_type
  
  

  subroutine actual_eos_init

    use fundamental_constants_module, only: m_e, m_p, c_light, hplanck

    implicit none
 
    A = M_PI * m_e**4 * c_light**5 / (THREE * hplanck**3)
    B2 = EIGHT * M_PI * m_e**3 * c_light**3 * m_p  / (THREE * hplanck**3)

  end subroutine actual_eos_init



  subroutine actual_eos(input, state)

    implicit none

    integer,      intent(in   ) :: input
    type (eos_t), intent(inout) :: state

    ! Local variables
    double precision :: dens, temp, enth, pres, eint, entr
    double precision :: x, dxdr

    dens = state % rho
    temp = state % T
    pres = state % p
    enth = state % h
    eint = state % e
    entr = state % s

    B = B2 * state % mu_e

    select case (input)

       !-------------------------------------------------------------------------
       ! Now do the calculations. In every case,
       ! make sure we have pressure, density, energy, and enthalpy.
       ! Relevant equations:
       ! rho = B x**3
       ! p   = A ( x * (2x**2 - 3)(x**2 + 1)**1/2 + 3 sinh**-1(x) )
       ! h   = (8A / B) * (1 + x**2)**1/2
       ! e   = h - p / rho
       !-------------------------------------------------------------------------

    case (eos_input_rh)

       ! dens, enthalpy, and xmass are inputs

       ! Solve for the pressure and energy:

       x = (dens / B)**THIRD
       pres = pressure(x)
       eint = enth - pres / dens


    case (eos_input_rt)

       ! dens, temp, and xmass are inputs

       ! Solve for the pressure, energy and enthalpy:

       x = (dens / B)**THIRD
       pres = pressure(x)
       enth = enthalpy(x)
       eint = enth - pres / dens


    case (eos_input_tp)

       ! temp, pres, and xmass are inputs

       ! Solve for the density, energy and enthalpy:

       call pres_iter(pres, dens)

       x = (dens / B)**THIRD
       enth = enthalpy(x)
       eint = enth - pres / dens


    case (eos_input_rp)

       ! dens, pres, and xmass are inputs

       ! Solve for the enthalpy and energy:

       x = (dens / B)**THIRD
       enth = enthalpy(x)
       eint = enth - pres / dens


    case (eos_input_re)

       ! dens, energy, and xmass are inputs

       ! Solve for the pressure and enthalpy:

       x = (dens / B)**THIRD
       pres = pressure(x)
       enth = enthalpy(x)


    case (eos_input_ps)

       ! pressure, entropy and xmass are inputs

       ! Solve for the density, energy and enthalpy:

       call pres_iter(pres, dens)

       x = (dens / B)**THIRD
       enth = enthalpy(x)
       eint = enth - pres / dens


    case (eos_input_ph)

       ! pressure, enthalpy and xmass are inputs

       ! Solve for the density and energy:

       x = ( ( (B * enth) / (EIGHT * A) )**2 - ONE )**HALF
       dens = B * x**3
       eint = enth - pres / dens


    case (eos_input_th)

       ! temperature, enthalpy and xmass are inputs

       ! Solve for the density, energy and pressure:

       x = ( ( (B * enth) / (EIGHT * A) )**2 - ONE )**HALF
       dens = B * x**3
       pres = pressure(x)
       eint = enth - pres / dens


    case default

       call amrex_error('EOS: invalid input.')

    end select

    !-------------------------------------------------------------------------
    ! Now we have all relevant quantities, regardless of the inputs.
    !-------------------------------------------------------------------------

    state % T   = temp
    state % rho = dens
    state % h   = enth
    state % s   = entr
    state % e   = eint
    state % p   = pres

    ! All temperature derivatives are zero since the gas is temperature-independent.

    state % dPdT = ZERO
    state % dhdT = ZERO
    state % dedT = ZERO
    state % dsdT = ZERO

    ! Density derivatives are computed using the chain rule, e.g. dPdr = dPdx * dxdr.

    x = (dens / B)**THIRD
    dxdr = THIRD * x / dens

    state % dPdr = dxdr * dpdx(x)
    state % dhdr = dxdr * dhdx(x)
    state % dedr = state % dhdr - state % dpdr / state % rho + state % p / (state % rho)**2
    state % dsdr = ZERO

    ! Heat capacities are zero: the gas properties don't change when the temperature changes.

    state % cv = ZERO
    state % cp = ZERO

    ! Adiabatic gamma_1 == d(log p) / d(log rho) |_s.

    state % gam1 = state % dpdr * (state % rho / state % p)

#ifdef EXTRA_THERMO
    ! Derivatives with respect to A and Z.

    state % dpdA = - state % p / state % abar
    state % dpdZ =   state % p / (ONE + state % zbar)

    state % dedA = - state % e / state % abar
    state % dedZ =   state % e / (ONE + state % zbar)
#endif

    ! Sound speed.

    state % cs = sqrt(state % dpdr)

  end subroutine actual_eos

  subroutine actual_eos_finalize

    implicit none

    ! Nothing to do here, yet.

  end subroutine actual_eos_finalize



  double precision function pressure(x)

    implicit none

    double precision, intent(in)  :: x

    pressure = A * ( x * (TWO * x**2 - THREE) * (x**2 + ONE)**HALF + THREE * asinh(x) )

  end function pressure



  double precision function enthalpy(x)

    implicit none

    double precision, intent(in)  :: x

    enthalpy = (EIGHT * A / B) * (ONE + x**2)**HALF

  end function enthalpy



  double precision function dpdx(x)

    implicit none

    double precision, intent(in) :: x

    dpdx = A * ( (TWO * x**2 - THREE)*(x**2 + ONE)**HALF + &
                 x * (4*x) * (x**2 + ONE)**HALF + &
                 x**2 * (TWO * x**2 - THREE) * (x**2 + ONE)**(-HALF) + &
                 THREE * (x**2 + ONE)**(-HALF) )

  end function dpdx



  double precision function dhdx(x)

    implicit none

    double precision, intent(in) :: x

    dhdx = enthalpy(x) * (x / (x**2 + ONE))

  end function dhdx



  subroutine pres_iter(pres, dens)

    implicit none

    double precision, intent(inout) :: pres, dens

    double precision :: x, dx
    integer          :: iter

    ! Starting guess for the iteration.

    x = ONE

    ! We are solving the equation:
    ! f(x) = p_want - p(x) = 0.
    ! Then we can use Newton's method, with dfdx = -dpdx.
    ! We iterate until the density change is close enough to zero.

    do iter = 1, max_iter

       dx = (pres - pressure(x)) / dpdx(x)

       x  = x + dx

       if ( abs(dx) / x .lt. iter_tol ) then
          exit
       endif

    enddo

    if (iter .eq. max_iter) then
       call amrex_error("EOS: pres_iter failed to converge.")
    endif

    dens = B * x**3

  end subroutine pres_iter



end module actual_eos_module
