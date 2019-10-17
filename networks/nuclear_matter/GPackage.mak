F90sources += actual_network.F90

f90sources += PhysicalConstantsModule.f90
f90sources += UnitsModule.f90

ifneq ($(USE_REACT), FALSE)
  F90sources += actual_burner.F90
  F90sources += actual_rhs.F90
endif
