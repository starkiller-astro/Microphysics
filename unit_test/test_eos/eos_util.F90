subroutine do_eos(lo, hi, &
                  sp, s_lo, s_hi, npts) bind(C, name="do_eos")

  use variables
  use network
  use eos_type_module
  use eos_module
  use amrex_fort_module, only : rt => amrex_real
  use amrex_constants_module
  use extern_probin_module
  use actual_eos_module, only : eos_name

  implicit none

  integer, intent(in) :: lo(3), hi(3)
  integer, intent(in) :: s_lo(3), s_hi(3)
  real(rt), intent(inout) :: sp(s_lo(1):s_hi(1), s_lo(2):s_hi(2), s_lo(3):s_hi(3), p % n_plot_comps)
  integer, intent(in), value :: npts

  real(rt) :: dlogrho, dlogT, dmetal
  real(rt) :: metalicity, temp_zone, dens_zone
  real(rt) :: xn_zone(nspec)

  type(eos_t) :: eos_state
  type(eos_t) :: eos_state_reference

  integer :: ii, jj, kk, index_h1, index_he4

  !$gpu
  
  dlogrho = (log10(dens_max) - log10(dens_min))/(npts - 1)
  dlogT   = (log10(temp_max) - log10(temp_min))/(npts - 1)
  dmetal    = (metalicity_max  - ZERO)/(npts - 1)

  do kk = lo(3), hi(3)

     ! set the composition -- approximately solar
     metalicity = ZERO + dble(kk)*dmetal

     index_h1  = network_species_index("hydrogen-1")
     index_he4 = network_species_index("helium-4")

     if (index_h1 == -1 .or. index_he4 == -1) then
        xn_zone(:) = metalicity/(nspec)
     else
        xn_zone(:) = metalicity/(nspec - 2)   ! all but H, He
        xn_zone(index_h1)  = 0.75_rt - HALF*metalicity
        xn_zone(index_he4) = 0.25_rt - HALF*metalicity
     endif

     do jj = lo(2), hi(2)
        temp_zone = 10.0_rt**(log10(temp_min) + dble(jj)*dlogT)

        do ii = lo(1), hi(1)
           dens_zone = 10.0_rt**(log10(dens_min) + dble(ii)*dlogrho)

           eos_state % rho = dens_zone
           eos_state % T = temp_zone
           eos_state % xn(:) = xn_zone(:)

           ! store default state
           sp(ii, jj, kk, p % irho) = dens_zone
           sp(ii, jj, kk, p % itemp) = temp_zone
           sp(ii, jj, kk, p % ispec: p % ispec-1+nspec) = xn_zone(:)

           if (eos_supports_input_type(eos_input_rt)) then
              ! call EOS using rho, T
              call eos(eos_input_rt, eos_state)
           endif

           call copy_eos_t(eos_state_reference, eos_state)

           sp(ii, jj, kk, p % ih) = eos_state % h
           sp(ii, jj, kk, p % ie) = eos_state % e
           sp(ii, jj, kk, p % ip) = eos_state % p
           sp(ii, jj, kk, p % is) = eos_state % s

           if (eos_supports_input_type(eos_input_rh)) then
              ! call EOS using rho, h

              ! reset T to give it some work to do
              eos_state % T = 100.d0

              call eos(eos_input_rh, eos_state)

              sp(ii, jj, kk, p % ierr_T_eos_rh) = &
                   abs(eos_state % T - temp_zone)/temp_zone

              call copy_eos_t(eos_state, eos_state_reference)
           else
              sp(ii, jj, kk, p % ierr_T_eos_rh) = ZERO
           endif



           if (eos_supports_input_type(eos_input_tp)) then
              ! call EOS using T, p

              ! reset rho to give it some work to do
              eos_state % rho = 1.d0

              call eos(eos_input_tp, eos_state)

              sp(ii, jj, kk, p % ierr_rho_eos_tp) = &
                   abs(eos_state % rho - dens_zone)/dens_zone

              call copy_eos_t(eos_state, eos_state_reference)              
           else
              sp(ii, jj, kk, p % ierr_rho_eos_tp) = ZERO
           endif



           if (eos_supports_input_type(eos_input_rp)) then
              ! call EOS using r, p

              ! reset T to give it some work to do
              eos_state % T = 100.d0

              call eos(eos_input_rp, eos_state)

              sp(ii, jj, kk, p % ierr_T_eos_rp) = &
                   abs(eos_state % T - temp_zone)/temp_zone

              call copy_eos_t(eos_state, eos_state_reference)              
           else
              sp(ii, jj, kk, p % ierr_T_eos_rp) = ZERO
           endif



           if (eos_supports_input_type(eos_input_re)) then
              ! call EOS using r, e

              ! reset T to give it some work to do
              eos_state % T = 100.d0

              call eos(eos_input_re, eos_state)

              sp(ii, jj, kk, p % ierr_T_eos_re) = &
                   abs(eos_state % T - temp_zone)/temp_zone

              call copy_eos_t(eos_state, eos_state_reference)              
           else
              sp(ii, jj, kk, p % ierr_T_eos_re) = ZERO
           endif



           if (eos_supports_input_type(eos_input_ps)) then
              ! call EOS using p, s

              ! reset T and rho to give it some work to do
              eos_state % T = 100.d0
              eos_state % rho = 1.d0


              ! some EOSes don't have physically valid treatments
              ! of entropy throughout the entire rho-T plane
              if (eos_state%s > ZERO) then

                 call eos(eos_input_ps, eos_state)

                 ! store the thermodynamic state
                 sp(ii, jj, kk, p % ierr_T_eos_ps) = &
                      abs(eos_state % T - temp_zone)/temp_zone
                 sp(ii, jj, kk, p % ierr_rho_eos_ps) = &
                      abs(eos_state % rho - dens_zone)/dens_zone

              else
                 sp(ii, jj, kk, p % ierr_T_eos_ps) = ZERO
                 sp(ii, jj, kk, p % ierr_rho_eos_ps) = ZERO

              endif

              call copy_eos_t(eos_state, eos_state_reference)
           else
              sp(ii, jj, kk, p % ierr_T_eos_ps) = ZERO
              sp(ii, jj, kk, p % ierr_rho_eos_ps) = ZERO
           endif


           if (eos_supports_input_type(eos_input_ph)) then
              ! call EOS using p, h

              ! reset T and rho to give it some work to do
              eos_state % T = 100.d0
              eos_state % rho = 1.d0

              call eos(eos_input_ph, eos_state)

              sp(ii, jj, kk, p % ierr_T_eos_ph) = &
                   abs(eos_state % T - temp_zone)/temp_zone
              sp(ii, jj, kk, p % ierr_rho_eos_ph) = &
                   abs(eos_state % rho - dens_zone)/dens_zone

              call copy_eos_t(eos_state, eos_state_reference)
           else
              sp(ii, jj, kk, p % ierr_T_eos_ph) = ZERO
              sp(ii, jj, kk, p % ierr_rho_eos_ph) = ZERO
           endif



           if (eos_supports_input_type(eos_input_th)) then
              ! call EOS using T, h
              ! this doesn't work for all EOSes (where h doesn't depend on T)
              ! reset rho to give it some work to do -- for helmeos, h is not
              ! monotonic, so we only perturb rho slightly here
              eos_state % rho = 0.9 * eos_state % rho

              call eos(eos_input_th, eos_state)

              sp(ii, jj, kk, p % ierr_rho_eos_th) = &
                   abs(eos_state % rho - dens_zone)/dens_zone

              call copy_eos_t(eos_state, eos_state_reference)
           else
              sp(ii, jj, kk, p % ierr_rho_eos_th) = ZERO
           endif

        enddo
     enddo
  enddo

end subroutine do_eos
