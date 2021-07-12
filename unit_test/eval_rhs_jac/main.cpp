#include <iostream>
#include <cstring>
#include <vector>

#include <AMReX_ParmParse.H>
#include <AMReX_MultiFab.H>
using namespace amrex;

#include <extern_parameters.H>
#include <eos.H>
#include <network.H>
#include <burn_cell.H>
#include <unit_test.H>

int main(int argc, char *argv[]) {

  amrex::Initialize(argc, argv);

  ParmParse ppa("amr");

  std::string probin_file = "probin";

  ppa.query("probin_file", probin_file);

  std::cout << "probin = " << probin_file << std::endl;

  const int probin_file_length = probin_file.length();
  Vector<int> probin_file_name(probin_file_length);

  for (int i = 0; i < probin_file_length; i++)
    probin_file_name[i] = probin_file[i];

  init_unit_test(probin_file_name.dataPtr(), &probin_file_length);

  // C++ EOS initialization (must be done after Fortran eos_init and init_extern_parameters)
  eos_init(small_temp, small_dens);

  // C++ Network, RHS, screening, rates initialization
  network_init();

  burn_cell_c();

  amrex::Finalize();
}
