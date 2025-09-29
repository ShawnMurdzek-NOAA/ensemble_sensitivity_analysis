"""
Ensemble Sensitivity Analysis (ESA)

Object for handling ESA calculations

shawn.s.murdzek@noaa.gov
"""

#---------------------------------------------------------------------------------------------------
# Import Modules
#---------------------------------------------------------------------------------------------------

import numpy as np
import scipy.stats as ss


#---------------------------------------------------------------------------------------------------
# Contents
#---------------------------------------------------------------------------------------------------

class ens_data():
    """
    Class to handle ESA with processed ensemble output

    Parameters
    ----------
    state : np.array
        Ensemble state at the initial time (i.e., the independent variable in the ESA regression). 
        Dimensions: (Nens, Nx)
    resp : np.array
        Value of the response function for each ensemble member. Dimensions: (Nens)
    x : np.array
        X location for model fields. Dimensions: (Nx)
    y : np.array
        Y location for model fields. Dimensions: (Nx)
    z : np.array
        Z location for model fields. Dimensions: (Nx)
    state_meta : dictionary, optional
        Metadata for the ensemble state at the initial time
    resp_meta : dictionary, optional
        Metadata for the response function

    """

    def __init__(self, state, resp, x, y, z, state_meta={}, resp_meta={}):

        self.state = state
        self.resp = resp
        self.x = x
        self.y = y
        self.z = z
        self.state_meta = state_meta
        self.resp_meta = resp_meta

        # Determine size of model state and number of ensemble members
        self.Nens, self.Nx = np.shape(state)
    
    
    def compute_esa(self):
        """
        Compute ensemble sensitivity and associated p value for each state variable
        
        Returns
        -------
        None.
        
        Notes
        -----

        Source: Hill et al. (2020, MWR) eqn (1)
        
        Note that ddof is not required here because the denominator used when computing the 
        covariance and variance (e.g., N or N-1) cancels when computing the regression slope.

        """
        
        esa = np.zeros([self.Nx])
        pval = np.zeros([self.Nx])
        for i in range(self.Nx):
            out = ss.linregress(self.state[:, i], self.resp)
            esa[i] = out[0]
            pval[i] = out[3]
        
        self.esa = esa
        self.pval = pval
    
    
    def compute_variance_diff(self, ob_var, ddof=1):
        """
        Compute estimated reduction in response function variance from assimilating an 
        observation at a particular location.

        Parameters
        ----------
        ob_var : float
            Observation error variance
        ddof : integer, optional
            Delta degrees of freedom. Passed to the cov and var functions.
            Denominator of covariance and variance computation is N - ddof.
            For sample statistics, ddof should be 1.

        Returns
        -------
        None.
        
        Notes
        -----
        
        Source: Hill et al. (2020, MWR) eqn (3)

        """
        
        var_diff = np.zeros([self.Nx])
        for i in range(self.Nx):
            cov = np.cov(self.state[:, i], self.resp, ddof=ddof)[0, 1]
            var = np.var(self.state[:, i], ddof=ddof)
            var_diff[i] = - (cov * cov) / (var + ob_var)
        
        self.var_diff = var_diff


"""
End esa.py
"""