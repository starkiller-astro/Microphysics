#ifndef _variables_H_
#define _variables_H_

// A module to provide integer indices into the various storage arrays
// for accessing the different variables by name.

#include <network.H>
#include <burn_type.H>

enum plot_t
{
    p_irho = 0,
    p_itemp = 1,
    p_ispec = 2,
    p_ispec_old = p_ispec + NumSpec,
    p_itemp_dot = p_ispec_old + NumSpec,
    p_ienuc_dot = p_itemp_dot + 1,
    p_ijac = p_ienuc_dot + 1,
    n_plot_comps = p_ijac + 1 + neqs * neqs
};

#endif
