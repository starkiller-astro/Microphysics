module actual_eos_module

  use amrex_error_module
  use amrex_constants_module, only: ZERO, HALF, TWO
  use amrex_fort_module, only : rt => amrex_real
  use eos_type_module
  use eos_aux_data_module

  implicit none

  character (len=64), public :: eos_name = "stellarcollapse"
  
  integer          :: max_newton = 100

  double precision :: ttol = 1.0d-8
  double precision :: dtol = 1.0d-8

  character(len=15) :: errfmt = '(3(e12.5,x))'

contains


  function eos_supports_input_type(input) result(supported)

    use eos_type_module

    implicit none

    integer, intent(in) :: input
    logical :: supported

    if (input == eos_input_rt .or. &
        input == eos_input_rp .or. &
        input == eos_input_re .or. &
        input == eos_input_tp) then

       supported = .true.

    else

       supported = .false.

    endif

  end function eos_supports_input_type

  
  ! EOS initialization routine 
  ! This reads in the HDF5 file containing the tabulated data
  subroutine actual_eos_init

    use amrex_paralleldescriptor_module, only: amrex_pd_ioprocessor
    use extern_probin_module, only: eos_file, use_energy_shift
    use network, only: network_species_index

    implicit none
 
    if (amrex_pd_ioprocessor()) print *, 'Reading HDF5 file', eos_file
    call read_stellarcollapse_file(eos_file,use_energy_shift)

  end subroutine actual_eos_init



  subroutine actual_eos(input, state)

    ! Stellar Collapse EOS
    ! 
    ! The stellarcollapse tables are indexed by log(density), 
    ! log(temperature), and electron fraction.  As such, the usual
    ! 'composition' variable passed to the EOS is the electron fraction.
    !
    ! Make sure you use a network that uses ye as a species!

    implicit none

    ! Input arguments
    integer,      intent(in   ) :: input
    type (eos_t), intent(inout) :: state

    ! Local variables and arrays
    double precision :: e_want, p_want, s_want, h_want
    double precision, parameter :: tol = 1.0d-8

    integer :: ierr

    ! Convert to the units used by the table.
    call convert_to_table_format(input, state)

    ierr = 0

    select case (input)

    !---------------------------------------------------------------------------
    ! dens, temp, and ye are inputs; 
    ! this is direct table interpolation, so nothing to do here
    !---------------------------------------------------------------------------
       
    case (eos_input_rt)

       !---------------------------------------------------------------------------
       ! dens, enthalpy, and xmass are inputs
       !---------------------------------------------------------------------------

       continue
       
    case (eos_input_rh)
       
       ! NOT CURRENTLY IMPLEMENTED
       call amrex_error("eos_input_th is not supported")

       !---------------------------------------------------------------------------
       ! temp, pres, and ye are inputs; iterate to find density
       !---------------------------------------------------------------------------

    case (eos_input_tp)

       ! Make sure the initial density guess is within table
       state % rho = max(mindens_tbl, min(maxdens_tbl, state % rho))

       ! We want to converge to the given pressure
       p_want = state % p

       call newton_iter(state, ierr, ipres, idens, p_want)

       if (ierr > 0) call amrex_error("Error in Newton iteration")




       !---------------------------------------------------------------------------
       ! dens, pres, and ye are inputs; iterate to find the temperature
       !---------------------------------------------------------------------------

    case (eos_input_rp)

       ! Make sure the initial temperature guess is within the table
       state % T = max(mintemp_tbl, min(maxtemp_tbl, state % T))

       ! We want to converge to the given pressure
       p_want = state % p

       call newton_iter(state, ierr, ipres, itemp, p_want)

       if (ierr > 0) call amrex_error("Error in Newton iteration")



       !---------------------------------------------------------------------------
       ! dens, energy, and ye are inputs; iterate to find temperature
       !---------------------------------------------------------------------------

    case (eos_input_re)

       ! Make sure the initial guess for temperature is within the table
       state % T = max(mintemp_tbl, min(state % T, maxtemp_tbl))

       ! We want to converge to the given energy
       e_want = state % e

       ! iterate to get the temperature
       call newton_iter(state, ierr, iener, itemp, e_want)

       if (ierr > 0) call amrex_error("Error in Newton iteration")



       !---------------------------------------------------------------------------
       ! pres, entropy, and xmass are inputs
       !---------------------------------------------------------------------------

    case (eos_input_ps)
       ! NOT CURRENTLY IMPLEMENTED
       call amrex_error("eos_input_ps is not supported")


       !---------------------------------------------------------------------------
       ! pres, enthalpy, and xmass are inputs
       !---------------------------------------------------------------------------

    case (eos_input_ph)
       ! NOT CURRENTLY IMPLEMENTED
       call amrex_error("eos_input_ph is not supported")


       !---------------------------------------------------------------------------
       ! temp, enthalpy, and xmass are inputs
       !---------------------------------------------------------------------------

    case (eos_input_th)
       ! NOT CURRENTLY IMPLEMENTED
       call amrex_error("eos_input_th is not supported")


       !---------------------------------------------------------------------------
       ! The EOS input doesn't match any of the available options.
       !---------------------------------------------------------------------------

    case default 

       call amrex_error("EOS: invalid input")

    end select

    ! Do a final lookup - by now we should have a consistent density and temperature
    call table_lookup(state)


    ! Convert back to hydro units from table units.
    ! Also builds some quantities like enthalpy.
    call convert_from_table_format(state)

  end subroutine actual_eos

  subroutine actual_eos_finalize

    implicit none

    ! Nothing to do here, yet.

  end subroutine actual_eos_finalize



  subroutine newton_iter(state, ierr, var, dvar, f_want)

    use interpolate_module

     implicit none

     type (eos_t),       intent(inout) :: state
     integer,            intent(in   ) :: var, dvar
     double precision,   intent(in   ) :: f_want
     integer,            intent(inout) :: ierr

     integer          :: iter, ivar
     double precision :: smallx, error, xnew, xtol
     double precision :: f, x, dfdx, df(3)

     logical :: converged, err
     character(len=128) :: errstring

     if (.not. (dvar .eq. itemp .or. dvar .eq. idens) ) then
       ierr = ierr_iter_var
       return
     endif

     converged = .false.
     err = .false.

     ! find out which table variable we are interpolating for
     select case(var)
     case (ipres)
        ivar = ilogpress
     case (iener)
        ivar = ilogenergy
     case (ientr)
        ivar = ientropy
     case default
        call amrex_error("newton_iter: don't know how to handle var",var)
     end select

     do iter = 1, max_newton

        ! If we're converged, exit the loop
        if (converged) return

        ! interpolate the table for var; df is dfdrho,dfdT,dfdye
        call tri_interpolate(state%rho,state%T,state%y_e, &
                             nrho,ntemp,nye, &
                             eos_logrho,eos_logtemp,eos_ye, &
                             eos_table(:,:,:,ivar), &
                             f, df, err)
        if (err) then
           write(errstring,trim(errfmt)) state%rho,state%T,state%y_e
           call amrex_error('newton iter: failure to interpolate',trim(errstring))
        endif

        ! Figure out what variable we're working with
        if (dvar .eq. itemp) then

          x = state % T
          ! note that we are in table units
          smallx = mintemp_tbl
          xtol = ttol

          dfdx = df(2)

        else ! dvar == density

          x = state % rho
          ! note that we are in table units
          smallx = mindens_tbl
          xtol = dtol

          dfdx = df(1)

        endif

        ! Now do the calculation for the next guess for T/rho

        xnew = x - (f - f_want) / dfdx

        ! Don't let the temperature/density change by more than a factor of two
        ! Note that because temperature/dens are logarithmic we want to do this via
        ! addition and not multiplication, which differs from how we do it in, say,
        ! the Helmholtz EOS

        xnew = max(x - dlog10(TWO), min(xnew, x + dlog10(TWO)))

        ! Don't let us freeze/evacuate

        xnew = max(smallx, xnew)

        ! Store the new temperature/density
        if (dvar .eq. itemp) then
          state % T    = xnew
        else
          state % rho  = xnew
        endif

        ! Compute the error from the last iteration
        error = abs( (xnew - x) / x )

        if (error .lt. xtol) converged = .true.
               
     enddo

     ! Call error if too many iterations were needed
     if (.not. converged) ierr = ierr_iter_conv

  end subroutine newton_iter



  function get_munu(rho,T,ye) result(munu)
    use interpolate_module 

    double precision, intent(in   ) :: rho, T, ye
    double precision                :: munu

    type(eos_t) :: state
    double precision :: derivs(3)
    logical :: err
    character(len=128) :: errstring

    ! convert our values to table format before interpolating
    state%rho = rho
    state%T = T
    state%y_e = ye
    call convert_to_table_format(eos_input_rt, state)

    ! look it up
    call tri_interpolate(state%rho,state%T,state%y_e,&
                         nrho,ntemp,nye, &
                         eos_logrho,eos_logtemp,eos_ye, &
                         eos_table(:,:,:,imunu),&
                         munu,derivs,err)

    ! check return
    if (err) then
       write(errstring,trim(errfmt)) state%rho,state%T,state%y_e
       call amrex_error('get_munu: tri-interpolate failure:',trim(errstring))
    endif
    
  end function get_munu


end module actual_eos_module
