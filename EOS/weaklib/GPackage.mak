F90sources += actual_eos.F90
F90sources += weaklib_type.F90

# In weaklib/Distributions/Library
f90sources += wlKindModule.f90
f90sources += wlGridModule.f90
f90sources += wlThermoStateModule.f90
f90sources += wlDependentVariablesModule.f90
f90sources += wlInterpolationModule.f90
f90sources += wlIOModuleHDF.f90

# In weaklib/Distributions/EOSSource
f90sources += wlEquationOfStateTableModule.f90
f90sources += wlEOSIOModuleHDF.f90

# In weaklib/Distributions/OpacitySource
f90sources += wlOpacityFieldsModule.f90
f90sources += wlOpacityTableModule.f90
f90sources += wlOpacityTableIOModuleHDF.f90
