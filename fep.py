# Copyright (c) 2018 Tom L. Underwood
# 
# Permission is hereby granted, free of charge, to any person obtaining 
# a copy of this software and associated documentation files (the 
# "Software"), to deal in the Software without restriction, including 
# without limitation the rights to use, copy, modify, merge, publish, 
# distribute, sublicense, and/or sell copies of the Software, and to 
# permit persons to whom the Software is furnished to do so, subject to
# the following conditions:
# 
# The above copyright notice and this permission notice shall be 
# included in all copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, 
# EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF 
# MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND 
# NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS 
# BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN 
# ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN 
# CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE 
# SOFTWARE.


r"""
.. moduleauthor:: Tom L. Underwood


This module contains functions for performing simple minimpulations of 
free energy profiles.
By *free energy profile* here I mean a function :math:`F(S)=-\ln p(S)`,
where :math:`p(S)` is the probability of the system exhibiting some
*order parameter* :math:`S`. 
Note that the free energy profile :math:`F(S)` just defined is 
*dimensionless* and does not include the usual :math:`k_BT` factor.

Most of the functions here deal with thermodynamic ensembles 
in which the probability of the system being in some configuration
:math:`\sigma` is :math:`p_{\sigma}\propto\exp(-bS_{\sigma})`, where
:math:`b` is the thermodynamic 'force' conjugate to the order parameter
:math:`S`. For example, in the :math:`NVT` ensemble 
:math:`p_{\sigma}\propto\exp(-\beta E_{\sigma})`, where :math:`\beta`
is the thermodynamic beta and :math:`E` denotes the system energy. Hence 
the below functions can be used for free energy profiles over :math:`E`
obtained from :math:`NVT` simulations, with :math:`b=\beta`. 
The same is true for :math:`NPT` and :math:`\mu VT` simulations, where 
:math:`p_{\sigma}` is given by 
:math:`p_{\sigma}\propto\exp(-\beta E_{\sigma}+\beta PV_{\sigma})` and
:math:`p_{\sigma}\propto\exp(-\beta E_{\sigma}+\beta \mu N_{\sigma})`
respectively, where :math:`P` denotes the system pressure, :math:`V`
denotes the system volume, :math:`\mu` denotes the chemical potential,
and :math:`N` denotes the number of particles in the system. Moreover
in the :math:`NPT` ensemble one could alternatively consider free energy
profiles over :math:`V`, in which case :math:`b=\beta P`; and in the 
:math:`\mu VT` ensemble one could alternatively consider free energy
profiles over :math:`N`, in which case :math:`b=\beta \mu`.

Perhaps the most useful functions below perform single-histogram reweighting
over the quantity :math:`b`: the functions take the free energy
profile over :math:`S` at a given :math:`b`, and return the free energy
profile at a different :math:`b`.

Note:
   By *array* below we mean the ``numpy.array`` data type.

"""




import numpy as np
import warnings

import scipy.optimize




def from_file(filename):

    r""" Import a free energy profile from a file.
    
    The format of the file should be as follows. Each line corresponds to an
    order parameter, with the first token on the line being the order
    parameter, and the second token being the dimensionless free energy.
    The line may contain further tokens, but these are not used here.
    Moreover lines in the file beginning with '#' are treated as 
    comments and ignored.
    
    Args:
       filename (str): The name of the file to import.
    
    Returns:
       array: An array containing the order parameter values read from the file.
 
       array: An array containing the free energy values read from the file.

    """

    # TO DO: Perform some safety checks on the file? Or is the below sufficient?
    
    with open(filename,'r') as f:
        
        string = f.read()
        
    return from_str(string)
                



def from_str(string):

    r"""Import a free energy profile from a string.
    
    The format of the string should be as follows. Each line corresponds to
    an order parameter, with the first token on the line being the order 
    parameter, and the second token being the dimensionless free energy. 
    The line may contain further tokens, but these are not used here.
    Moreover lines beginning with '#' are treated as comments and
    ignored.
    
    Args:
       string (str): The string containing the free energy profile.

    Returns:
       array: An array containing the order parameter values read from the string.

       array: An array containing the free energy values read from the string.

    """

    # TO DO: Perform some safety checks on the read input?
    
    op = []
    fe = []
        
    lines = string.splitlines()

    for line in lines:

        # Keep only the content before the first comment character '#'
        line = line.split('#')[0]

        # If what's left isn't whitespace...
        if line.strip() != "":

            op.append( float(line.split()[0])  )
            fe.append( float(line.split()[1])  )

    return (np.array(op), np.array(fe))




def fep_str(op, fe):

    r"""Return a readable string representation of a free energy profile.

    The string has two columns. The first column contains the order parameters, while
    the second column contains the corresponding values of the free energy.

    Args:
       op (array): An array containing the order parameter values.
 
       fe (array): An array containing the free energy values.

    Returns:
       string: The string representation of the free energy profile.

    """

    assert len(op) == len(fe), "op and fe must be of same length"
    
    # TO DO: There is probably a more elegant way to do this.
    # Also, control of the format of the output is important; this
    # should be added to this function eventually

    string = ""

    for i in range(0,len(op)):

        string += str(op[i])+" "+str(fe[i])+"\n"

    return string




def to_file(op, fe, filename):

    r"""Export a free energy profile to a file.

    In the file the free energy profile is represented as two columns. The first 
    column contains the order parameters, while the second column contains the 
    corresponding values of the free energy.

    Args:
       op (array): An array containing the order parameter values.

       fe (array): An array containing the free energy values.
 
       filename (string):  The name of the file to export the free energy profile to.

    """

    # TO DO: Perform some safety checks on the file?
    # Also, control of the format of the output is important; this
    # should be added to this function eventually

    with open(filename,'w') as file:

        file.write( fep_str(op,fe) )




def pdf(fe):

    r"""Calculate the probability distribution for a free energy profile.

    Args:
       fe (array): An array containing the free energy values.

    Returns:
       array: An array of floats giving the probability corresponding to each
          element in `fe`.

    Note:
       The returned probability distribution is such that the sum over all elements is
       1 (which is not the same as the integral over the interpolated profile
       being 1!)

    """

    # Numerical issues will occur with very negative values of fe (it blows up). 
    # Hence we shift it so that the minimum is at 0.
    fe = shift(fe)

    normconst = np.sum( np.exp( -fe ) )

    return np.exp(-fe) / normconst




def shift(fe):

    r"""Shifts a free energy profile so that its minimum is at 0.

    Args:
       fe (array): An array containing the free energy values.

    Returns:
       array: The free energy values shifted so that the minimum value is 0.

    """
        
    return fe - fe.min()




def phase_probs(op, fe, op_thresh):

    r"""Returns the probabilities of two phases for a free energy profile.

    Phase 1 is defined to be configurations for which the order
    parameter is less than or equal to some threshold order parameter
    :math:`S_{thresh}`; and phase 2 corresponds to order parameters :math:`>S_{thresh}`.

    Args:
       op (array): An array containing the order parameter values.

       fe (array): An array containing the free energy values.

       op_thresh (float): The threshold order parameter :math:`S_{thresh}`
          defining the two phases.
        
    Returns:
       float: The probability of phase 1.

       float: The probability of phase 2.

    """

    assert len(op) == len(fe), "op and fe must be of same length"

    # Get the probability corresponding to each array element (i.e. pdf(fe)),
    # and sum over all those elements for which the order parameter is <= op_thresh.
    # Note that op > op_thresh achieves this (NOT op <= op_thresh)
    p1 = np.sum (  np.ma.masked_where(op > op_thresh, pdf(fe) ) )
    p2 = 1.0 - p1

    return p1, p2




def expected_op(op, fe, op_thresh):

    r"""Returns the expected order parameter of two phases for a free energy profile.

    Phase 1 is defined to be configurations for which the order
    parameter is less than or equal to some threshold order parameter
    :math:`S_{thresh}`; and phase 2 corresponds to order parameters :math:`>S_{thresh}`.

    Args:
       op (array): An array containing the order parameter values.

       fe (array): An array containing the free energy values.

       op_thresh (float): The threshold order parameter :math`S_{thresh}` 
          defining the two phases.
        
    Returns:
       float: The expected value of the order parameter for phase 1.

       float: The expected value of the order parameter for phase 2.

    Note:
       An expected value for a phase will be nan, and a ``RuntimeWarning`` will
       be thrown if the probability of the phase is 0 to within the precision 
       of the machine. In this case this function raises an additional warning
       to provide further context to the ``RuntimeWarning``.

    """

    assert len(op) == len(fe), "op and fe must be of same length"

    p1, p2 = phase_probs(op, fe, op_thresh)

    # op1 = (\sum_{i\in 1} op_i p_i) / (\sum_{i\in 1} p_i), where op_i is the order parameter
    # for array element i, p_i is the probability for element i, and \sum_{i\in 1}
    # is the sum over all p_i such that the op <= op_thresh. This is implemented
    # below.
    # Note that op < op_thresh is correct for op1 (NOT op <= op_thresh)
    op1 = np.sum (  np.ma.masked_where(op > op_thresh, pdf(fe) * op) ) / p1
    op2 = np.sum (  np.ma.masked_where(op <= op_thresh, pdf(fe) * op) ) / p2

    # The calculation of op1 and op2 above will encounter a divide-by-0 problem if p1 or
    # p2 is 0. The RuntimeWarning raised naturally in this case does not provide insight
    # into the source of the problem; hence I've raised another warning to provide further
    # information if it happens...
    if( p1==0 or p2==0 ):
        warnings.warn("'fep.expected_op' encountered p1=0 or p2=0. The function will return "+
                      "a 'nan' and may trigger a RuntimeWarning")

    return op1, op2




def reweight_add_lrc_lj(op, fe, epsilon, sigma, rc, volume, molsize, exclude=False):

    r"""Reweight a free energy profile over :math:`N` to add Lennard-Jones long-range corrections.

    This function reweights a free energy profile, assuming that the order parameter is
    the number of molecules in the system :math:`N` (as cwould be obtained from, e.g.
    a :math:`\mu VT` simulation), to add long-range corrections for Lennard-Jones interactions 
    between particles (assuming that the profile currently corresponds to a system where 
    they are absent).

    The long-range correction to the energy per Lennard-Jones particle is
    
    .. math::

       E_{lrc}={\frac{8\pi\rho\epsilon}{3}\Biggl[\Bigl(\frac{\sigma}{r_c}\Bigr)^3-\frac{1}{3}\Bigl(\frac{\sigma}{r_c}\Bigr)^9\Biggr]},

    where :math:`\epsilon` and :math:`\sigma` are the Lennard-Jones parameters, and :math:`\rho`
    is the density of Lennard-Jones particles. We assume the system is comprised of :math:`N` molecules
    comprised of :math:`n` atoms interacting via the Lennard-Jones potential, in which case
    :math:`\rho=Nn/V`, where :math:`V` is the volume of the system. 

    Here the free energy profile is assumed to pertain to a simulation where the
    volume is constant, the system is comprised of molecules with `molsize` atoms Lennard-Jones
    interaction centres, and long-range corrections have not been included. Moreover the 
    system is assumed to be homogenous (as is always the assumption when applying standard 
    long-range corrections).

    Args:
       op (array): An array containing the order parameter values.

       fe (array): An array containing the free energy values.

       epsilon (float): The value of the Lennard-Jones parameter :math:`\epsilon`, in units of 
          :math:`k_BT` (i.e. :math:`\epsilon/kT`).

       sigma (float): The value of the Lennard-Jones parameter :math:`\sigma`.

       rc (float): The cut-off distance for the Lennard-Jones potential.

       volume (float): The volume of the system.

       molsize (int): The number of Lennard-Jones interaction centres per molecule.

       exclude (boolean): If `true` then :math:`\rho` used in the above equation is chosen
          to be :math:`\rho=(Nn-n)/V` instead of :math:`Nn/V`. If the intramolecular
          contribution to the Lennard-Jones energy is *excluded* from the total energy, then
          :math:`\rho=(Nn-n)/V` corresponds to also excluding interactions with all images of
          the molecule with regards to the long-range corrections (though this is a weird thing
          to do).

    Returns:
       array: The free energies corresponding to `op`, but with long-range corrections added
          via reweighting

    Note: 
       By passing a negative value for `epsilon`, this function can be used to remove
       long-range corrections from a free energy profile corresponding to a simulation which
       incorporates long-range corrections.

    """

    # TO DO: Is the 'exclude' option in this function necessary?
    
    assert len(op) == len(fe), "op and fe must be of same length"
    
    tailconst = (8.0*np.pi/3.0) * ( (sigma/rc)**9/3.0 - (sigma/rc)**3 ) \
                * epsilon * sigma**3 / volume 
 
    nexcluded = 0

    if exclude:
        nexcluded = molsize

    # TO DO: Explain this equation better
    return fe + tailconst * molsize*op * ( molsize*op - np.ones(len(op))*nexcluded )




def reweight(op, fe, b_current, b_new):

    r"""Reweight a free energy profile to a new thermodynamic force.

    Args:
       op (array): An array containing the order parameter values.

       fe (array): An array containing the free energy values.

       b_current (float): The value of the force, which the current free energy 
          profile corresponds to.

       b_new (float): The value of the force to reweight the free energy profile to

    Returns:
       array: The free energies corresponding to `op`, but reweighted to the chemical
          potential `mu_new`

    """

    assert len(op) == len(fe), "op and fe must be of same length"
    
    return fe - (b_new - b_current) * op




def reweight_to_coexistence(op, fe, b_current, b_lbound, b_ubound, op_thresh, tol=1.48e-08, maxiter=50, warn_tol=0.05):

    r"""Reweight a free energy profile to a thermodynamic force which corresponds to
    coexistence.

    Here coexistence corresponds to the system being in phase 1 and phase 2 with
    equal probability, where phase 1 is defined as the set of configurations for
    which the order parameter is less than or equal to some threshold order parameter
    :math:`S_{thresh}`, and phase 2 corresponds to order parameters :math:`>S_{thresh}`.

    Brent's method is used to locate :math:`b_{co}`, the thermodynamic force corresponding
    to coexistence. This is achieved by determining the minimum in :math:`|p_1(b)-0.5|`, where 
    :math:`p_1(b)` denotes the phase 1 probability at thermodynamic force :math:`b`. The 
    function ``scipy.optimize.brent`` is used to perform the optimisation.

    Args:
       op (array): An array containing the order parameter values.

       fe (array): An array containing the free energy values.

       b_current (float): The thermodynamic force :math:`b` which the free energy 
          profile `fe` corresponds to.

       b_lbound (float): Lower bound on :math:`b` used in the search for coexistence.

       b_ubound (float): Upper bound on :math:`b` used in the search for coexistence.

       op_thresh (float): The threshold order parameter :math:`S_{thresh}` defining the two phases.

       tol (float, optional): Convergence threshold for Brent's method.
          The optimisation is stopped if subsequent iterations yield a change in the objective function
          of less than `tol`. Thus `tol` is effectively the precision in to which :math:`b_{co}` is
          to be determined; :math:`b_{co}` corresponds to the probability of phase 1 being 0.5 to a 
          precision of approximately `tol`.

       maxiter (int, optional): Maximum number of iterations to use in Brent's method.

       warn_tol (float, optional): A warning is raised if the probability of phase 1 at the :math:`b_co`
          determined by this function is greater than `warn_tol` away from 0.5.

    Returns:
       float: The thermodynamic force :math:`b_{co}` deemed to correspond to coexistence.

       float: The probability of phase 1 at :math:`b_{co}`. Note that
          this should be very close to 0.5, otherwise the optimisation has not gone
          according to plan for some reason.
    
       array: The free energy profile at :math:`b_{co}`.

    Note:
       The optimisation method assumes that `b_lbound` and `b_ubound` bracket
       :math:`b_{co}`. If this is not the case then a warning is raised. 

       A warning is raised if the :math:`b_co` determined by this function is greater than
       `warn_tol` away from 0.5.

       A warning is raised if the maximum number of iterations in the optimisation procedure is
       reached.
    """

    assert len(op) == len(fe), "op and fe must be of same length"
    assert b_lbound < b_ubound, "b_lbound must be < b_ubound"

    # Function which returns the absolute difference in the probability of phase 1,
    # relative to 0.5, for a given chemical potential - calculated by 
    # reweighting self to that chemical potential
    # Note that the tolerance amounts to the precision to which p1 is close to 0.5??
    def objective(b): 
            
        p1, p2 = phase_probs(op, reweight(op, fe, b_current, b), op_thresh) 
        return abs(p1 - 0.5)

    # Check that b_co is bracketed by b_lbound and b_ubound
    p1_lbound, p2_tmp = phase_probs(op, reweight(op, fe, b_current, b_lbound), op_thresh)
    p1_ubound, p2_tmp = phase_probs(op, reweight(op, fe, b_current, b_ubound), op_thresh) 
    if (p1_lbound-0.5) * (p1_ubound-0.5) > 0:
        warnings.warn("reweight_to_coexistence: b_lbound and b_ubound do not bracket b_co")

    # The Brent method determines where the objective function is 0 (which corresponds
    # to coexistence)
    b_opt, obj_opt, niters, fcalls = scipy.optimize.brent(objective, brack = (b_lbound,b_ubound),
                                                          tol=tol, full_output=True, maxiter=maxiter)

    if niters >= maxiter:
        warnings.warn("reweight_to_coexistence: maximum number of iterations reached!")

    # Now reweight to the optimum b to retrieve the free energy profile
    fe_opt = reweight(op, fe, b_current, b_opt)
    p1_opt, p2_opt = phase_probs(op, fe_opt, op_thresh) 

    # Check that p1_opt is close to 0.5, as it should be.
    if abs(p1_opt - 0.5) > warn_tol:
        warnings.warn("reweight_to_coexistence: coexistence probability not close to 0.5!")

    return b_opt, p1_opt, fe_opt
