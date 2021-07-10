module react_zones_module

  use variables
  use network
  use eos_type_module
  use eos_module
  use burn_type_module
  use amrex_fort_module, only : rt => amrex_real
  use amrex_constants_module
  use extern_probin_module
  use util_module
  use actual_rhs_module

  implicit none

contains

  subroutine init_state(lo, hi, &
                        state, s_lo, s_hi, npts) bind(C, name="init_state")

    integer, intent(in) :: lo(3), hi(3)
    integer, intent(in) :: s_lo(3), s_hi(3)
    real(rt), intent(inout) :: state(s_lo(1):s_hi(1), s_lo(2):s_hi(2), s_lo(3):s_hi(3), p % n_plot_comps)
    integer, intent(in) :: npts

    real(rt) :: dlogrho, dlogT, dmetal
    real(rt) :: temp_zone, dens_zone
    real(rt), allocatable :: xn_zone(:,:)

    integer :: ii, jj, kk
    real(rt) :: sum_X

    if (npts > 1) then
       dlogrho = (log10(dens_max) - log10(dens_min))/(npts - 1)
       dlogT   = (log10(temp_max) - log10(temp_min))/(npts - 1)
    else
       dlogrho = ZERO
       dlogT   = ZERO
    endif

    allocate(xn_zone(nspec, 0:npts-1))   ! this assumes that lo(3) = 0

    call get_xn(npts, xn_zone)

    ! normalize -- just in case
    do kk = lo(3), hi(3)
       sum_X = sum(xn_zone(:, kk))
       xn_zone(:, kk) = xn_zone(:, kk)/sum_X
    enddo

    do kk = lo(3), hi(3)   ! xn loop
       do jj = lo(2), hi(2)   ! T loop
          do ii = lo(1), hi(1)   ! rho loop

             state(ii, jj, kk, p % itemp) = 10.0_rt**(log10(temp_min) + dble(jj)*dlogT)
             state(ii, jj, kk, p % irho)  = 10.0_rt**(log10(dens_min) + dble(ii)*dlogrho)
             state(ii, jj, kk, p % ispec_old:p % ispec_old+nspec-1) = max(xn_zone(:, kk), 1.e-10_rt)

          enddo
       enddo
    enddo

  end subroutine init_state


  subroutine do_rhs  (lo, hi, &
                      state, s_lo, s_hi) bind(C, name="do_rhs")

    implicit none

    integer, intent(in) :: lo(3), hi(3)
    integer, intent(in) :: s_lo(3), s_hi(3)
    real(rt), intent(inout) :: state(s_lo(1):s_hi(1), s_lo(2):s_hi(2), s_lo(3):s_hi(3), p % n_plot_comps)

    type (burn_t)   :: burn_state
    type (eos_t)    :: eos_state
    integer         :: ii, jj, kk, j, i, n
    real(rt)        :: ydot(neqs), jac(neqs, neqs)

    !$gpu

    do ii = lo(1), hi(1)
       do jj = lo(2), hi(2)
          do kk = lo(3), hi(3)

             burn_state % rho = state(ii, jj, kk, p % irho)
             burn_state % T = state(ii, jj, kk, p % itemp)
             do j = 1, nspec
                burn_state % xn(j) = state(ii, jj, kk, p % ispec_old + j - 1)
             enddo

             call normalize_abundances_burn(burn_state)

             call burn_to_eos(burn_state, eos_state);
             call eos(eos_input_rt, eos_state);
             call eos_to_burn(eos_state, burn_state);

             ! the integrator doesn't actually care about the initial internal
             ! energy.
             burn_state % e = ZERO

             burn_state % i = ii
             burn_state % j = jj
             burn_state % k = kk

             burn_state % self_heat = .true.

             call actual_rhs(burn_state, ydot)
             call actual_jac(burn_state, jac)

             do j = 1, nspec
                state(ii, jj, kk, p % ispec + j - 1) = ydot(j)
             enddo

             state(ii, jj, kk, p % itemp_dot) = ydot(net_itemp)
             state(ii, jj, kk, p % ienuc_dot) = ydot(net_ienuc)

             n = 0
             do j = 1, neqs
             do i = 1, neqs
                state(ii, jj, kk, p % ijac + n) = jac(i, j)
                n = n + 1
             end do
             end do

          enddo
       enddo
    enddo

  end subroutine do_rhs

end module react_zones_module
