#
# Author: Tom L. Underwood
#
# This file provides a demonstration of how to use the 'fep' module. The file 'FEDDAT.000_200' in this directory
# is an output file created by DL_MONTE [1], from a transition-matrix grand-canonical Monte Carlo simulation 
# of CH4 modelled via the Trappe force field [2]. (See the 'CONTROL', 'CONFIG' and 'FIELD' files in this
# directory for further details - they are the input files used in the DL_MONTE simulation. Perhaps also see
# the DL_MONTE manual [1]). 'FEDDAT.000_200' is the dimensionless free energy profile vs. number of molecules in 
# the system generated by DL_MONTE, already in an appropriate format to be read by 'fep'.
#
# Here we first import the free energy profile from 'FEDDAT.000_200'. Then we use reweighting to add long-range
# corrections to the free energy profile: the DL_MONTE simulation did not incorporate the long-range contribution
# to the Lennard-Jones potential. This is output to a file 'fep_w_lrc.dat', which can be plotted in, e.g. gnuplot.
# Moreover the corresponding probability profile is output to 'pdf_w_lrc.dat'. 
# Then we reweight the free energy profile (including long-range corrections) to a chemical potential corresponding
# to coexistence, i.e. the chemical potential which yields an equal probability of the system being in the gas
# and liquid phases at the considered temperature. The coexistence free energy and probability profiles are then 
# output to the files 'fep_w_lrc_co.dat' and 'pdf_w_lrc_co.dat', and at the same time the determined chemical 
# potential at coexistence, as well as the liquid and gas densities at coexistence, are printed.
#
# The final coexistence liquid and gas densities should be in agreement with [2]: at 175K the gas density was found
# to be 0.048(2)g/mL and the liquid density was found to be 0.296(3)g/mL.
#
# [1] https://www.ccp5.ac.uk/DL_MONTE
# [2] B. Chen and J. I. Siepmann, J. Phys. Chem. B 103, 5370 (1999)
#



# For compatibility with both Python 2 and 3 print statements
from __future__ import print_function

# Amend the path so that the 'fep' can be found - assume it is 2 directories up
import sys
import os
path = os.path.join(  os.path.realpath( os.path.join('..', '..') )  )
sys.path.append( path )

import math

import fep



# Import the free energy profile from the simulation output
# 'op' is the order parameter array (number of molecules), and 'fe'
# is the free energy associated with each order parameter macrostate.
filename="FEDDAT.000_200"
op, fe = fep.from_file(filename)



# Simulation parameters required to add the long-range corrections
# These are taken from the CONTROL, CONFIG and FIELD input files

# System volume (Angstroms)
volume = 30*30*30
# Cut-off for Lennard-Jones potential (Angstroms)
rc = 9.0
# Lennard-Jones parameter epsilon/(k_BT): T=175.0K and epsilon =  15.3K
epsilonbeta = 15.3/175.0
# Lennard-Jones parameter sigma
sigma = 3.31



# It is useful for comparison with other results to consider the density (g/L) 
# as the order parameter instead of the number of molecules. Note that 0.0375445 
# here is a conversion factor particular to CH4, and depends on the molar mass of CH4
rho = op / (volume*0.0375445)



# Add long-range corrections to the free energy profile
fe = fep.reweight_add_lrc_lj(op, fe, epsilonbeta, sigma, rc, volume, 4)

# Output the long-range-corrected free energy profile vs. density of methane in g/mL.
# But shift the free energy profile so that its minimum is 0
fep.to_file( rho, fep.shift(fe), "fep_w_lrc.dat")

# Also output the long-range-corrected probability distribution vs. density
fep.to_file( rho, fep.pdf(fe), "pdf_w_lrc.dat")



# We now turn to reweighting the free energy profile to different chemical potentials



# Get the chemical potential (in units of k_BT) the the simulation was performed at.
# In CONTROL the thermodynamic activity z=exp(\mu/(k_BT)) is given as 0.0013. Get the
# chemical potential from that
mu = math.log(0.0013)

print("mu/k_BT from simulation = ", mu)



# Reweight to coexistence automatically. Be careful not to explore over too wide a chemical
# potential range - reweighting will fail if one reweights to chemical potentials far from
# the simulation value. Here I explore within 5% above and below the simulation chemical 
# potential. Here I choose to use the density as the order parameter, and define the gas
# and liquid phases as being comprised of a number of particles less than and greater than, 
# respectively, 160 (which is appropriate for our simulation volume).
# Note that with reweighting one MUST use the number of molecules as the order parameter.
mu_lbound = min(mu*1.05,mu*0.95)
mu_ubound = max(mu*1.05,mu*0.95)
print("mu_lbound/k_BT = ", mu_lbound)
print("mu_ubound/k_BT = ", mu_ubound)

mu_co, p_co, fe_co = fep.reweight_to_coexistence(op, fe, mu, mu_lbound, mu_ubound,  op_thresh=160, maxiter=50)



# Now 'mu_co' is the coexistence chemical potential (divided by k_BT) and 'fe_co' is the coexistence
# free energy profile. Also, 'p_co' is the gas probability at coexistence. This should be very close to
# 0.5 otherwise something has gone wrong
print("mu_co/k_BT from optimisation = ", mu_co)
print("p_co from optimisation = ", p_co)

# Export the coexistence free energy profile vs. density (g/mL). But shift the free energy
# profile so that its minimum is 0
fep.to_file( rho, fep.shift(fe_co), "fep_w_lrc_co.dat")

# Also output the coexistence probability distribution vs. density
fep.to_file( rho, fep.pdf(fe_co), "pdf_w_lrc_co.dat")



# Calculate and output the gas and liquid expeected number of molecules at coexistence, 
# again using 160 as molecules as the cut-off between the  liquid and gas phases. Convert
# these values to the gas and liquid densities, then output the densities.
op1, op2 = fep.expected_op(op, fe_co, op_thresh=160)
print( "coexistence gas density (g/mL) = ", op1/(volume*0.0375445) )
print( "coexistence liquid density (g/mL) = ", op2/(volume*0.0375445) )


