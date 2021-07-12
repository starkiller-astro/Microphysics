#!/usr/bin/env python3

from __future__ import print_function

import itertools
import os
import subprocess
import sys

executable = "main.Linux.gfortran.omp.exe"

link_files = ["xin.rprox",
              "gr0_3d.small",
              "helm_table.dat"]

link_files.append(executable)

# this dictionary holds the parameters we want to set.  For each key,
# we use a list to give all the possible values (even if there is only
# a single value).
aprox13_params = {
    "dens_min": [1.e4, 1.e6],
    "dens_max": [1.e8],
    "temp_min": [5.e7],
    "temp_max": [5.e9],
    "tmax": [1.e-3],
    "rtol_spec": [1.e-12, 1.e-8],
    "atol_spec": [1.e-12, 1.e-8],
    "jacobian": [1, 2]
}

aprox19_params = {
    "dens_min": [1.e5],
    "dens_max": [5.e8],
    "temp_min": [5.e7],
    "temp_max": [5.e8],
    "tmax": [1.e-9],
    "rtol_spec": [1.e-8, 1.e-6],
    "atol_spec": [1.e-8, 1.e-6],
    "jacobian": [1, 2],
    "call_eos_in_rhs": ["T", "F"]
}

rprox_params = {
    "dens_min": [1.e5],
    "dens_max": [2.e6],
    "temp_min": [1.e8],
    "temp_max": [1.e9],
    "tmax": [1.e-6],
    "rtol_spec": [1.e-12, 1.e-8],
    "atol_spec": [1.e-12, 1.e-8],
    "jacobian": [1, 2],
    "call_eos_in_rhs": ["T", "F"],
    "renormalize_abundances": ["T", "F"]
}


params = rprox_params

params_file = r"""
&PROBIN

  xin_file   = "xin.rprox"
  run_prefix = "react_rprox_"

  small_dens = 1.0

  @@test-params@@

/
"""

# time out for a run, in seconds
TIMEOUT = 600

def run(command, stdin=False, outfile=None):
    """ run a command in the unix shell """

    sin = None
    if stdin: sin = subprocess.PIPE
    p0 = subprocess.Popen(command, stdin=sin, stdout=subprocess.PIPE,
                          stderr=subprocess.STDOUT, shell=True)

    stdout0 = p0.communicate(timeout=TIMEOUT)
    if stdin: p0.stdin.close()
    rc = p0.returncode
    p0.stdout.close()

    if outfile is not None:
        try: cf = open(outfile, "w")
        except IOError:
            sys.exit("ERROR: unable to open file for writing: {}".format(outfile))
        else:
            for line in stdout0:
                if line is not None:
                    cf.write(line.decode("ascii"))
            cf.close()

    return stdout0[0], rc


def doit():

    # itertools.product() will produce lists with every possible
    # combination from the input lists.  Here we use as the input lists
    # the values from our dictionary.  This magic comes from:
    # http://stackoverflow.com/questions/3873654/combinations-from-dictionary-with-list-values-using-python

    combinations = [[{k: v} for (k, v) in zip(params.keys(), values)]
                    for values in itertools.product(*params.values())]


    run_no = 0

    top_dir = os.getcwd()

    outcomes = {}

    of = open("test-results.txt", "w")

    for c in combinations:

        cparams = {k: v for d in c for k, v in d.items()}

        # run this combination of test parameters

        # make the directory
        run_no += 1
        odir = "{:02d}".format(run_no)

        print("running case {}...".format(run_no))

        try:
            os.mkdir(odir)
        except:
            sys.exit("unable to create directory")

        # copy the executable and suport files

        for f in link_files:
            try:
                os.symlink(os.path.join(top_dir, f),
                           os.path.join(odir, os.path.basename(f)))
            except:
                sys.exit("unable to link file")

        # let's work in that directory
        os.chdir(odir)

        # write the input file
        infile = "inputs.{}".format(odir)
        with open(infile, "w") as f:
            for line in params_file.splitlines():
                if line.find("@@test-params@@") >= 0:
                    for k, v in cparams.items():
                        f.write("  {} = {}\n".format(k, v))
                else:
                    f.write("{}\n".format(line))

        # run
        command = "./{} {}".format(executable, infile)
        stdout, rc = run(command, outfile="run.out")

        os.chdir(top_dir)

        # log the result
        if rc == 0:
            result = "successful"
        else:
            result = "failed"

        outcomes[odir] = result

        # if we were successful, the last 4 lines are the summary of RHS evaluations
        stats = ""
        if rc == 0:
            stats = stdout.splitlines()[-4:]

        # output summary
        of.write("test {}: {}\n".format(odir, result))
        for k, v in sorted(cparams.items()):
            of.write("  {} = {}\n".format(k, v))
        of.write("\n")

        for line in stats:
            of.write("{}\n".format(line.decode("utf-8")))
        of.write("\n")
        of.flush()

    of.close()

    for k, v in sorted(outcomes.items()):
        print("{}: {}".format(k, v))

        

if __name__ == "__main__":
    doit()
