module actual_network

  use amrex_fort_module, only : rt => amrex_real

  implicit none

  integer, parameter :: nspec = 1
  integer, parameter :: nspec_evolve = 0
  integer, parameter :: iye = 1

  integer, parameter :: naux  = 1
  integer, parameter :: ine = 1

  integer, parameter :: nrates = 0
  integer, parameter :: num_rate_groups = 0

  character (len=16), save :: aux_names(naux)
  character (len= 5), save :: short_aux_names(naux)

  character (len=16), save :: spec_names(nspec) 
  character (len= 5), save :: short_spec_names(nspec)

  double precision, allocatable :: aion(:), zion(:), nion(:)
  double precision, allocatable :: bion(:), mion(:), wion(:)

#ifdef AMREX_USE_CUDA
  attributes(managed) :: aion, zion, nion, bion, mion, wion
#endif

  !$acc declare create(aion, zion, bion, nion, mion, wion)

contains

  subroutine actual_network_init

    use amrex_constants_module, only: ZERO, ONE

    implicit none

    aux_names(1) = "Ne"
    short_aux_names(1) = "Ne"

    spec_names(iye)  = "electron-fraction"

    short_spec_names(iye)  = "Ye"

    allocate(aion(nspec))
    allocate(zion(nspec))
    allocate(nion(nspec))
    allocate(bion(nspec))
    allocate(mion(nspec))
    allocate(wion(nspec))
    
    aion(iye)  = ONE

    zion(iye)  = ONE

    ! Binding energies per nucleus in MeV
    bion(iye)  = ZERO

    ! Set the number of neutrons
    nion(:) = aion(:) - zion(:)

    ! Set the mass
    mion(iye) = ZERO

    ! Molar mass
    wion(iye) = ZERO

    !$acc update device(aion, zion, bion, nion, mion, wion)

  end subroutine actual_network_init


  subroutine actual_network_finalize

    implicit none

    if (allocated(aion)) then
       deallocate(aion)
    endif
    if (allocated(zion)) then
       deallocate(zion)
    endif
    if (allocated(nion)) then
       deallocate(nion)
    endif
    if (allocated(bion)) then
       deallocate(bion)
    endif
    if (allocated(mion)) then
       deallocate(mion)
    endif
    if (allocated(wion)) then
       deallocate(wion)
    endif

  end subroutine actual_network_finalize

end module actual_network
