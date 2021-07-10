module util_module

  use amrex_constants_module
  use amrex_fort_module, only : rt => amrex_real

  implicit none

contains

  subroutine get_xn(nz, xn_zone)

    use network, only: nspec, spec_names, network_species_index
    use extern_probin_module, only: primary_species_1, primary_species_2, primary_species_3
    use amrex_error_module

    integer, intent(in) :: nz
    real(rt), intent(  out) :: xn_zone(:,:)

    real(rt) :: Xp_min, Xp_max, dX
    integer :: n1, n2, n3, nprim
    integer :: is1, is2, is3
    integer :: k, n
    real(rt) :: excess

    ! get the primary species indices
    nprim = 0

    is1 = network_species_index(primary_species_1)
    if (is1 > 0) then
       nprim = nprim+1
    end if

    is2 = network_species_index(primary_species_2)
    if (is2 > 0) then
       nprim = nprim+1
    end if

    is3 = network_species_index(primary_species_3)
    if (is3 > 0) then
       nprim = nprim+1
    end if

    if (nprim == 0) then
       call amrex_error("ERROR: no primary species set")
    end if

    ! figure out how many zones to allocate to the each of the primary
    ! species and the extrema for the primary species
    if (nprim == 1) then
       n1 = nz
       n2 = 0
       n3 = 0

       Xp_min = 0.2_rt
       Xp_max = 0.9_rt

    else if (nprim == 2) then
       n1 = nz/2
       n2 = nz - n1
       n3 = 0

       Xp_min = 0.2_rt
       Xp_max = 0.8_rt

    else if (nprim == 3) then
       n1 = nz/3
       n2 = nz/3
       n3 = nz - n1 - n2

       Xp_min = 0.2_rt
       Xp_max = 0.7_rt

    end if

    do k = 1, nz
       xn_zone(:, k) = 0.0_rt

       if (k <= n1) then
          if (nprim >= 2) xn_zone(is2, k) = Xp_min/2
          if (nprim >= 3) xn_zone(is3, k) = Xp_min/2

          dX = (Xp_max - Xp_min) / max((n1 - 1), 1)
          xn_zone(is1, k) = Xp_min + (k - 1)*dX

       else if (nprim >= 2 .and. k <= n1 + n2) then
          xn_zone(is1, k) = Xp_min/2
          if (nprim >= 3) xn_zone(is3, k) = Xp_min/2

          dX = (Xp_max - Xp_min) / max((n2 - 1), 1)
          xn_zone(is2, k) = Xp_min + (k - (n1 + 1))*dX

       else
          xn_zone(is1, k) = Xp_min/2
          xn_zone(is2, k) = Xp_min/2

          dX = (Xp_max - Xp_min) / max((n3 - 1), 1)
          xn_zone(is3, k) = Xp_min + (k - (n1 + n2 + 1))*dX

       end if

       excess = ONE - sum(xn_zone(:, k))

       do n = 1, nspec
          if (n == is1 .or. &
              (n == is2 .and. nprim >= 2) .or. &
              (n == is3 .and. nprim >= 3)) cycle

          xn_zone(n, k) = excess / (nspec - nprim)

       end do

    end do

  end subroutine get_xn

  
  subroutine swap_f90_use_partition_factors() bind(C, name="swap_f90_use_partition_factors")
    use extern_probin_module, only: use_partition_factors

    if (use_partition_factors) then
       use_partition_factors = .false.
    else
       use_partition_factors = .true.
    endif
    
  end subroutine swap_f90_use_partition_factors

    
end module util_module
