module actual_rhs_module

  use amrex_constants_module
  use network
  use burn_type_module

  implicit none

contains

  subroutine actual_rhs_init()

    implicit none

  end subroutine actual_rhs_init


  subroutine actual_rhs(state)

    !$gpu

    implicit none

    type (burn_t)    :: state

    !$gpu

    state % ydot(iye)  = ZERO

  end subroutine actual_rhs


  subroutine actual_jac(state)

    implicit none

    type (burn_t)    :: state

    call set_jac_zero(state)

  end subroutine actual_jac

end module actual_rhs_module
