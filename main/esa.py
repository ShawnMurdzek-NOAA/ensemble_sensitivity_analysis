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
import xarray as xr


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
    full_shape : list-like, optional
        Shape of state matrix when unraveled 
    state_meta : dictionary, optional
        Metadata for the ensemble state at the initial time
    resp_meta : dictionary, optional
        Metadata for the response function

    """

    def __init__(self, state, resp, x, y, z, full_shape=(), state_meta={}, resp_meta={}):

        self.state = state
        self.resp = resp
        self.x = x
        self.y = y
        self.z = z
        self.full_shape = full_shape
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
    
    
    def save_to_netcdf(self, fname, fields=None, unravel=True, attrs={}):
        """
        Save output to a netCDF file

        Parameters
        ----------
        fname : string
            NetCDF file name
        fields : None or list of strings
            Fields to write out. Setting to None writes all computed fields
        unravel : boolean
            Option to unravel 1D fields prior to writing
        attrs : dictionary
            Additional attributes to add to the netCDF file

        Returns
        -------
        None.

        """
        
        # Populate fields if None
        if fields is None:
            fields = []
            all_fields = ['esa', 'pval', 'var_diff']
            for f in all_fields:
                if hasattr(self, f):
                    fields.append(f)
        
        # Set unravel to False if full_shape is not defined
        if len(self.full_shape) == 0:
            print('Warning: ens_data.save_to_netcdf: full_shape not defined. Setting unravel = False')
        
        # Create dataset
        if unravel:
            ds = self._create_dataset_unravel(fields)
        else:
            ds = self._create_dataset_ravel(fields)
            
        # Add additional attributes
        ds.attrs.update(attrs)
            
        # Save to netCDF
        ds.to_netcdf(fname)
        
    
    def _create_dataset_unravel(self, fields):
        """
        Unravel all fields and create a dataset

        Parameters
        ----------
        fields : list
            Fields to include

        Returns
        -------
        ds : xr.dataset
            Dataset with unraveled fields

        """
        
        # Create coords dictionary
        coords = {}
        for f in ['x', 'y', 'z']:
            coords[f] = (['i', 'j', 'k'], np.reshape(getattr(self, f), self.full_shape))
        
        # Create data dictionary
        data = {}
        for f in fields:
            data[f] = (['i', 'j', 'k'], np.reshape(getattr(self, f), self.full_shape))
        
        ds = xr.Dataset(data_vars=data, 
                        coords=coords, 
                        attrs=self._combine_metadata())
    
        return ds
    
    
    def _create_dataset_ravel(self, fields):
        """
        Create a dataset, but do not unravel the fields

        Parameters
        ----------
        fields : list
            Fields to include

        Returns
        -------
        ds : xr.dataset
            Dataset with raveled fields

        """
        
        # Create coords dictionary
        coords = {}
        for f in ['x', 'y', 'z']:
            coords[f] = ('i', getattr(self, f))
        
        # Create data dictionary
        data = {}
        for f in fields:
            data[f] = ('i', getattr(self, f))
        
        ds = xr.Dataset(data_vars=data, 
                        coords=coords, 
                        attrs=self._combine_metadata())
    
        return ds
    
    
    def _combine_metadata(self):
        """
        Combine state_meta and resp_meta into a single dictionary

        Returns
        -------
        meta : dictionary
            state_meta and resp_meta combined into one dictionary

        """
        
        meta = {}
        for k in self.state_meta.keys():
            meta[f"state_{k}"] = self.state_meta[k]
        for k in self.resp_meta.keys():
            meta[f"response_{k}"] = self.resp_meta[k]
            
        return meta


"""
End esa.py
"""