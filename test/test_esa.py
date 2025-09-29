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
        

"""
End test_esa.py
"""      