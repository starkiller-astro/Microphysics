module eos_override_module

  implicit none

  public eos_override

contains

  ! This is a user hook to override the details of the EOS state.

  subroutine eos_override(state)

    !$acc routine seq

    use extern_probin_module
    use eos_type_module, only: eos_t
    use actual_eos_module, only: eos_name

    implicit none

    type (eos_t) :: state

    !$gpu

    ! For the weaklib EOS, set the electron fraction from probin
    if (trim(eos_name) == "weaklib") then
       state % y_e = weaklib_electron_fraction
    endif

  end subroutine eos_override

end module eos_override_module
