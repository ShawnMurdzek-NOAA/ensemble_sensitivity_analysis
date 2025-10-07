"""
Tests for esa.py

shawn.s.murdzek@noaa.gov
"""

#---------------------------------------------------------------------------------------------------
# Import Modules
#---------------------------------------------------------------------------------------------------

# Add top-level directory to PYTHONPATH
import sys
import os
path = '/'.join(os.getcwd().split('/')[:-1])
sys.path.append(path)

import pytest
import numpy as np
import copy
import xarray as xr

from main import ens_io


#---------------------------------------------------------------------------------------------------
# Contents
#---------------------------------------------------------------------------------------------------

class TestESA():
    
    @pytest.fixture(scope='class')
    def sample_ens(self):
        state_fnames = [f"./data/wrf/mem00{n}/wrfout.2009-04-15_20:45:00.TEST.nc" for n in range(1, 4)]
        resp_fnames = [f"./data/wrf/mem00{n}/wrfout.2009-04-15_22:00:00.TEST.nc" for n in range(1, 4)]
        
        state_param = {'var':'T2', 'subset':False}
        resp_param = {'var':'PSFC', 'subset':False, 'reduction':'max'}
        
        ens_obj = ens_io.read_parse_wrf(state_fnames,
                                        resp_fnames,
                                        state_param,
                                        resp_param,
                                        horiz_coord='idx',
                                        vert_coord='idx',
                                        verbose=0)
        
        return ens_obj
    
    
    @pytest.fixture(scope='class')
    def sample_ens_3d(self):
        state_fnames = [f"./data/wrf/mem00{n}/wrfout.2009-04-15_20:45:00.TEST.nc" for n in range(1, 4)]
        resp_fnames = [f"./data/wrf/mem00{n}/wrfout.2009-04-15_22:00:00.TEST.nc" for n in range(1, 4)]
        
        state_param = {'var':'T', 'subset':False}
        resp_param = {'var':'PSFC', 'subset':False, 'reduction':'max'}
        
        ens_obj = ens_io.read_parse_wrf(state_fnames,
                                        resp_fnames,
                                        state_param,
                                        resp_param,
                                        horiz_coord='idx',
                                        vert_coord='idx',
                                        verbose=0)
        
        return ens_obj
    
    
    def test_esa(self, sample_ens):
        """
        Test ESA calculation
        """
        
        ens_obj = copy.deepcopy(sample_ens)
        
        ens_obj.compute_esa()
        
        # Compute ESA an alternate way
        # Note that the ddof keyword does not really matter, it just needs to be consistent for
        # the cov and var calculations
        esa = (np.cov(ens_obj.state[:, 0], ens_obj.resp, ddof=1)[0, 1] / 
               np.var(ens_obj.state[:, 0], ddof=1))
        
        assert hasattr(ens_obj, 'esa')
        assert hasattr(ens_obj, 'pval')
        assert len(ens_obj.esa) == ens_obj.Nx
        assert len(ens_obj.pval) == ens_obj.Nx
        assert np.isclose(ens_obj.esa[0], esa)
    
    
    def test_variance_diff(self, sample_ens):
        """
        Test calculation of estimated variance difference
        """
        
        ens_obj = copy.deepcopy(sample_ens)
        
        ens_obj.compute_variance_diff(1)
        
        assert hasattr(ens_obj, 'var_diff')
        assert len(ens_obj.var_diff) == ens_obj.Nx
    
    
    def test_create_dataset_unravel_2D(self, sample_ens):
        """
        Test _create_dataset_unravel using a 2D atmospheric state
        """
        
        # Replaces ESA field with T2, then unravel
        ens_obj = copy.deepcopy(sample_ens)
        ens_obj.esa = ens_obj.state[0, :]
        out_ds = ens_obj._create_dataset_unravel(['esa'])
        
        # Load actual values for T2
        true_ds = xr.open_dataset('./data/wrf/mem001/wrfout.2009-04-15_20:45:00.TEST.nc')
        
        assert np.all(np.isclose(out_ds['esa'].values, true_ds['T2'].values))
    
    
    def test_create_dataset_unravel_3D(self, sample_ens_3d):
        """
        Test _create_dataset_unravel using a 3D atmospheric state
        """
        
        # Replaces ESA field with T, then unravel
        ens_obj = copy.deepcopy(sample_ens_3d)
        ens_obj.esa = ens_obj.state[0, :]
        out_ds = ens_obj._create_dataset_unravel(['esa'])
        
        # Load actual values for T
        true_ds = xr.open_dataset('./data/wrf/mem001/wrfout.2009-04-15_20:45:00.TEST.nc')
    
        assert np.all(np.isclose(out_ds['esa'].values, true_ds['T'].values))
    
    
    def test_save_to_netcdf(self, sample_ens):
        """
        Test save_to_netcdf using a 2D atmospheric state
        """
        
        # Replaces ESA field with T2
        ens_obj = copy.deepcopy(sample_ens)
        ens_obj.esa = ens_obj.state[0, :]
        
        # Load actual values for T2
        true_ds = xr.open_dataset('./data/wrf/mem001/wrfout.2009-04-15_20:45:00.TEST.nc')
        
        # Test writing to netCDF with unravel=True
        fname = 'test.nc'
        ens_obj.save_to_netcdf(fname, fields=['esa'], unravel=True)
        test_ds = xr.open_dataset(fname)
        assert np.all(np.isclose(test_ds['esa'].values, true_ds['T2'].values))
        os.remove(fname)
        
        # Test writing to netCDF with unravel=False
        fname = 'test.nc'
        ens_obj.save_to_netcdf(fname, fields=['esa'], unravel=False)
        test_ds = xr.open_dataset(fname)
        assert np.all(np.isclose(test_ds['esa'].values, ens_obj.esa))
        os.remove(fname)
        

"""
End test_esa.py
"""      