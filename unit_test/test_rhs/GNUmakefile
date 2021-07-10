PRECISION  = DOUBLE
PROFILE    = FALSE

DEBUG      = FALSE

DIM        = 3

COMP	   = gnu

USE_MPI    = FALSE
USE_OMP    = FALSE

USE_REACT = TRUE

EBASE = main

USE_CXX_EOS = TRUE

USE_CXX_REACTIONS ?= TRUE

ifeq ($(USE_CXX_REACTIONS),TRUE)
  DEFINES += -DCXX_REACTIONS
endif

# define the location of the CASTRO top directory
MICROPHYSICS_HOME  := ../..

# This sets the EOS directory in Castro/EOS -- note: gamma_law will not work,
# you'll need to use gamma_law_general
EOS_DIR     := helmholtz

# This sets the network directory in Castro/Networks
NETWORK_DIR := aprox13

CONDUCTIVITY_DIR := stellar

ifeq ($(USE_CUDA), TRUE)
  INTEGRATOR_DIR := VODE
endif

EXTERN_SEARCH += .

Bpack   := ./Make.package
Blocs   := .

include $(MICROPHYSICS_HOME)/unit_test/Make.unit_test

