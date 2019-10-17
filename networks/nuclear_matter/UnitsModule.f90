MODULE UnitsModule

  use amrex_fort_module, only: DP => amrex_real

  USE PhysicalConstantsModule, ONLY: &
    SpeedOfLightMKS, &
    GravitationalConstantMKS, &
    BoltzmannConstantMKS, &
    ElectronVoltMKS, &
    PlanckConstantMKS, &
    AvogadroConstantMKS

  IMPLICIT NONE
  PRIVATE

  LOGICAL, PUBLIC :: UnitsActive = .FALSE.

  REAL(DP), PUBLIC, PARAMETER :: &
    SpeedOfLight          = 1.0_DP, &
    GravitationalConstant = 1.0_DP, &
    BoltzmannConstant     = 1.0_DP

  ! --- Length ---

  REAL(DP), PUBLIC, PARAMETER :: &
    Meter      = 1.0_DP, &
    Centimeter = 1.0e-2_DP * Meter, &
    Kilometer  = 1.0e+3_DP * Meter

  ! --- Time ---

  REAL(DP), PUBLIC, PARAMETER :: &
    Second      = SpeedOfLightMKS / SpeedOfLight * Meter, &
    Millisecond = 1.0e-3_DP * Second, &
    Microsecond = 1.0e-6_DP * Second

  ! --- Mass ---

  REAL(DP), PUBLIC, PARAMETER :: &
    Kilogram  = GravitationalConstantMKS / GravitationalConstant &
                  * Meter**3 / Second**2, &
    Gram      = 1.0e-3_DP * Kilogram, &
    SolarMass = 1.98892e30_DP * Kilogram

  ! --- Other Units of Measure and Constants ---

  REAL(DP), PUBLIC, PARAMETER :: &
    Joule          = Kilogram * ( Meter / Second )**2, &
    Erg            = Gram * ( Centimeter / Second )**2, &
    Bethe          = 1.0e51_DP * Erg, &
    ElectronVolt   = ElectronVoltMKS * Joule, &
    MeV            = 1.0e6_DP * ElectronVolt, &
    Kelvin         = BoltzmannConstantMKS / BoltzmannConstant * Joule, &
    Newton         = Joule / Meter, &
    Dyne           = Erg / Centimeter, &
    PlanckConstant = PlanckConstantMKS * Joule * Second, &
    AtomicMassUnit = Gram / AvogadroConstantMKS


  ! --- Units Displayed During Execution and for IO ---

  CHARACTER(16), PRIVATE, PARAMETER :: &
    DisplayLabel_Null            = '', &
    DisplayLabel_Length          = 'km', &
    DisplayLabel_Time            = 'ms', &
    DisplayLabel_Mass            = 'M_sun', &
    DisplayLabel_MassDensity     = 'g/cm^3', &
    DisplayLabel_ParticleDensity = '1/cm^3', &
    DisplayLabel_Velocity        = 'km/s', &
    DisplayLabel_Momentum        = 'g cm/s', &
    DisplayLabel_MomentumDensity = 'g/cm^2/s', &
    DisplayLabel_Energy          = 'MeV', &
    DisplayLabel_EnergyGlobal    = 'B', &
    DisplayLabel_EnergyDensity   = 'erg/cm^3', &
    DisplayLabel_Pressure        = 'erg/cm^3', &
    DisplayLabel_Temperature     = 'K'

  REAL(DP), PRIVATE, PARAMETER :: &
    DisplayUnit_Length          = Kilometer, &
    DisplayUnit_Time            = Millisecond, &
    DisplayUnit_Mass            = SolarMass, &
    DisplayUnit_MassDensity     = Gram / Centimeter**3, &
    DisplayUnit_ParticleDensity = 1.0_DP / Centimeter**3, &
    DisplayUnit_Velocity        = Kilometer / Second, &
    DisplayUnit_Momentum        = Gram * Centimeter / Second, &
    DisplayUnit_MomentumDensity = Gram / Centimeter**2 / Second, &
    DisplayUnit_Energy          = MeV, &
    DisplayUnit_EnergyGlobal    = Bethe, &
    DisplayUnit_EnergyDensity   = Erg / Centimeter**3, &
    DisplayUnit_Pressure        = Erg / Centimeter**3, &
    DisplayUnit_Temperature     = Kelvin

  TYPE, PRIVATE :: UnitsDisplayType
    LOGICAL  :: &
      Active = .FALSE.
    CHARACTER(16) :: &
      LengthLabel          = DisplayLabel_Null, &
      TimeLabel            = DisplayLabel_Null, &
      MassLabel            = DisplayLabel_Null, &
      MassDensityLabel     = DisplayLabel_Null, &
      ParticleDensityLabel = DisplayLabel_Null, &
      VelocityLabel        = DisplayLabel_Null, &
      MomentumLabel        = DisplayLabel_Null, &
      MomentumDensityLabel = DisplayLabel_Null, &
      EnergyLabel          = DisplayLabel_Null, &
      EnergyGlobalLabel    = DisplayLabel_Null, &
      EnergyDensityLabel   = DisplayLabel_Null, &
      PressureLabel        = DisplayLabel_Null, &
      TemperatureLabel     = DisplayLabel_Null
    REAL(DP) :: &
      LengthUnit          = 1.0_DP, &
      TimeUnit            = 1.0_DP, &
      MassUnit            = 1.0_DP, &
      MassDensityUnit     = 1.0_DP, &
      ParticleDensityUnit = 1.0_DP, &
      VelocityUnit        = 1.0_DP, &
      MomentumUnit        = 1.0_DP, &
      MomentumDensityUnit = 1.0_DP, &
      EnergyUnit          = 1.0_DP, &
      EnergyGlobalUnit    = 1.0_DP, &
      EnergyDensityUnit   = 1.0_DP, &
      PressureUnit        = 1.0_DP, &
      TemperatureUnit     = 1.0_DP
  END type UnitsDisplayType

  TYPE(UnitsDisplayType), PUBLIC :: UnitsDisplay

  PUBLIC :: ActivateUnitsDisplay
  PUBLIC :: DescribeUnitsDisplay

CONTAINS


  SUBROUTINE ActivateUnitsDisplay

    UnitsActive = .TRUE.

    UnitsDisplay % Active = .TRUE.

    UnitsDisplay % LengthLabel          = DisplayLabel_Length
    UnitsDisplay % TimeLabel            = DisplayLabel_Time
    UnitsDisplay % MassLabel            = DisplayLabel_Mass
    UnitsDisplay % MassDensityLabel     = DisplayLabel_MassDensity
    UnitsDisplay % ParticleDensityLabel = DisplayLabel_ParticleDensity
    UnitsDisplay % VelocityLabel        = DisplayLabel_Velocity
    UnitsDisplay % MomentumLabel        = DisplayLabel_Momentum
    UnitsDisplay % MomentumDensityLabel = DisplayLabel_MomentumDensity
    UnitsDisplay % EnergyLabel          = DisplayLabel_Energy
    UnitsDisplay % EnergyGlobalLabel    = DisplayLabel_EnergyGlobal
    UnitsDisplay % EnergyDensityLabel   = DisplayLabel_EnergyDensity
    UnitsDisplay % PressureLabel        = DisplayLabel_Pressure
    UnitsDisplay % TemperatureLabel     = DisplayLabel_Temperature

    UnitsDisplay % LengthUnit          = DisplayUnit_Length
    UnitsDisplay % TimeUnit            = DisplayUnit_Time
    UnitsDisplay % MassUnit            = DisplayUnit_Mass
    UnitsDisplay % MassDensityUnit     = DisplayUnit_MassDensity
    UnitsDisplay % ParticleDensityUnit = DisplayUnit_ParticleDensity
    UnitsDisplay % VelocityUnit        = DisplayUnit_Velocity
    UnitsDisplay % MomentumUnit        = DisplayUnit_Momentum
    UnitsDisplay % MomentumDensityUnit = DisplayUnit_MomentumDensity
    UnitsDisplay % EnergyUnit          = DisplayUnit_Energy
    UnitsDisplay % EnergyGlobalUnit    = DisplayUnit_EnergyGlobal
    UnitsDisplay % EnergyDensityUnit   = DisplayUnit_EnergyDensity
    UnitsDisplay % PressureUnit        = DisplayUnit_Pressure
    UnitsDisplay % TemperatureUnit     = DisplayUnit_Temperature

  END SUBROUTINE ActivateUnitsDisplay


  SUBROUTINE DescribeUnitsDisplay

    WRITE(*,*)
    WRITE(*,'(A5,A25,L2)') &
      '', 'Units Activation Status =', UnitsDisplay % Active
    WRITE(*,'(A5,A27)') &
      '', '---------------------------'
    WRITE(*,*)
    WRITE(*,'(A7,A24,A)') &
      '', 'Lenght Units: ', &
      TRIM( UnitsDisplay % LengthLabel )
    WRITE(*,'(A7,A24,A)') &
      '', 'Time Units: ', &
      TRIM( UnitsDisplay % TimeLabel )
    WRITE(*,'(A7,A24,A)') &
      '', 'Mass Units: ', &
      TRIM( UnitsDisplay % MassLabel )
    WRITE(*,'(A7,A24,A)') &
      '', 'Mass Density Units: ', &
      TRIM( UnitsDisplay % MassDensityLabel )
    WRITE(*,'(A7,A24,A)') &
      '', 'Particle Density Units: ', &
      TRIM( UnitsDisplay % ParticleDensityLabel )
    WRITE(*,'(A7,A24,A)') &
      '', 'Velocity Units: ', &
      TRIM( UnitsDisplay % VelocityLabel )
    WRITE(*,'(A7,A24,A)') &
      '', 'Momentum Units: ', &
      TRIM( UnitsDisplay % MomentumLabel )
    WRITE(*,'(A7,A24,A)') &
      '', 'Momentum Density Units: ', &
      TRIM( UnitsDisplay % MomentumDensityLabel )
    WRITE(*,'(A7,A24,A)') &
      '', 'Energy Units: ', &
      TRIM( UnitsDisplay % EnergyLabel )
    WRITE(*,'(A7,A24,A)') &
      '', 'Energy Global Units: ', &
      TRIM( UnitsDisplay % EnergyGlobalLabel )
    WRITE(*,'(A7,A24,A)') &
      '', 'Energy Density Units: ', &
      TRIM( UnitsDisplay % EnergyDensityLabel )
    WRITE(*,'(A7,A24,A)') &
      '', 'Pressure Units: ', &
      TRIM( UnitsDisplay % PressureLabel )
    WRITE(*,'(A7,A24,A)') &
      '', 'Temperature Units: ', &
      TRIM( UnitsDisplay % TemperatureLabel )
    WRITE(*,*)

  END SUBROUTINE DescribeUnitsDisplay


END MODULE UnitsModule
