"""
Tests for ens_io.py

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
import xarray as xr
import numpy as np

from main import ens_io


#---------------------------------------------------------------------------------------------------
# Contents
#---------------------------------------------------------------------------------------------------

class TestGeneral():
    
    def test_reduce_field(self):
        """
        Test various field reduction methods
        """
        
        x = np.random.random((30, 40))
        
        def gt(x):
            return np.sum(x > 0.5)
        
        def lt(x):
            return np.sum(x < 0.5)
        
        for s, f in zip(['max', 'min', 'mean', 'sum', 'npts_gt_thres', 'npts_lt_thres'], 
                        [np.amax, np.amin, np.mean, np.sum, gt, lt]):
            val = ens_io.reduce_field(x, reduction=s, kw={'thres':0.5})
            assert not hasattr(val, "__len__")  # Ensure that val is a scalar
            assert np.isclose(val, f(x))
        

class TestEnsWRFIO():

    @pytest.fixture(scope='class')
    def sample_fnames(self):
        state_fnames = [f"./data/wrf/mem00{n}/wrfout.2009-04-15_20:45:00.TEST.nc" for n in range(1, 4)]
        resp_fnames = [f"./data/wrf/mem00{n}/wrfout.2009-04-15_22:00:00.TEST.nc" for n in range(1, 4)]
        return state_fnames, resp_fnames
    
    
    def test_plain_read(self, sample_fnames):
        """
        Test a basic ensemble read with no subsetting and horiz_coord and vert_coord set to 'idx'
        """
        
        state_param = {'var':'T2', 'subset':False}
        resp_param = {'var':'PSFC', 'subset':False, 'reduction':'max'}
        
        ens_obj = ens_io.read_parse_wrf(sample_fnames[0],
                                        sample_fnames[1],
                                        state_param,
                                        resp_param,
                                        horiz_coord='idx',
                                        vert_coord='idx',
                                        verbose=0)
        
        # Compare to ensemble members opened using xarray
        state_ds1 = xr.open_dataset(sample_fnames[0][0])
        resp_ds1 = xr.open_dataset(sample_fnames[1][0])
        
        # Check state array and response function value
        assert np.all(np.isclose(np.ravel(state_ds1[state_param['var']][0, :].values), 
                                 ens_obj.state[0]))
        assert np.isclose(np.amax(resp_ds1[resp_param['var']].values), 
                                  ens_obj.resp[0])
        
        # Check coordinates
        for coord, name in zip([ens_obj.x, ens_obj.y],
                               ['west_east', 'south_north']):
            assert len(coord.shape) == 1
            assert coord.size == ens_obj.Nx
            assert coord.min() == 0
            assert coord.max() == np.amax(state_ds1[name].values)
        
        # Check other attributes
        assert ens_obj.Nens == len(sample_fnames[0])
        assert ens_obj.resp_meta['reduction'] == resp_param['reduction']
        
        
    def test_read_ens_wrapper(self, sample_fnames):
        """
        Test the read_ens wrapper
        """
        
        state_param = {'var':'T2', 'subset':False}
        resp_param = {'var':'PSFC', 'subset':False, 'reduction':'max'}
        
        ens_obj = ens_io.read_parse_wrf(sample_fnames[0],
                                        sample_fnames[1],
                                        state_param,
                                        resp_param,
                                        horiz_coord='idx',
                                        vert_coord='idx',
                                        verbose=0)
        
        # Compare to read_ens wrapper
        ens_obj2 = ens_io.read_ens(sample_fnames[0],
                                   sample_fnames[1],
                                   state_param,
                                   resp_param,
                                   horiz_coord='idx',
                                   vert_coord='idx',
                                   verbose=0,
                                   ftype='wrf')
        
        for i in range(len(ens_obj.state)):
            assert np.all(np.isclose(ens_obj.state[i], ens_obj2.state[i]))
            assert np.isclose(ens_obj.resp[i], ens_obj2.resp[i])
            
    
    def test_horiz_coord_xy(self, sample_fnames):
        """
        Test a ensemble read with horiz_coord set to 'xy'
        """
        
        state_param = {'var':'T2', 'subset':False}
        resp_param = {'var':'PSFC', 'subset':False, 'reduction':'max'}
        
        ens_obj = ens_io.read_parse_wrf(sample_fnames[0],
                                        sample_fnames[1],
                                        state_param,
                                        resp_param,
                                        horiz_coord='xy',
                                        vert_coord='idx',
                                        verbose=0)
        
        state_ds = xr.open_dataset(sample_fnames[0][0])
        xmax = (state_ds['west_east'].size - 1) * state_ds.attrs['DX'] / 2
        ymax = (state_ds['south_north'].size - 1) * state_ds.attrs['DX'] / 2
        assert ens_obj.x.max() == xmax
        assert ens_obj.x.min() == -xmax
        assert ens_obj.y.max() == ymax
        assert ens_obj.y.min() == -ymax
    
    
    def test_subset_idx(self, sample_fnames):
        """
        Test subsetting feature when vert_coord and horiz_coord are 'idx'
        """
        
        state_param = {'var':'T', 'subset':True,
                       'xlim':[10, 30], 'ylim':[5, 30], 'zlim':[5, 7]}
        resp_param = {'var':'PSFC', 'subset':True, 'reduction':'max',
                      'xlim':[20, 40], 'ylim':[15, 35]}
        
        ens_obj = ens_io.read_parse_wrf(sample_fnames[0],
                                        sample_fnames[1],
                                        state_param,
                                        resp_param,
                                        horiz_coord='idx',
                                        vert_coord='idx',
                                        verbose=0)
        
        # Compare to ensemble members opened using xarray
        state_ds = xr.open_dataset(sample_fnames[0][0])
        state_T = state_ds['T'][0,
                                state_param['zlim'][0]:(state_param['zlim'][-1]+1),
                                state_param['ylim'][0]:(state_param['ylim'][-1]+1),
                                state_param['xlim'][0]:(state_param['xlim'][-1]+1)].values
        state_T = np.ravel(state_T)
        
        resp_ds = xr.open_dataset(sample_fnames[1][0])
        resp = resp_ds['PSFC'][0,
                               resp_param['ylim'][0]:resp_param['ylim'][-1],
                               resp_param['xlim'][0]:resp_param['xlim'][-1]].values
        resp = np.amax(resp)
        
        assert len(state_T) == ens_obj.Nx
        assert np.all(np.isclose(state_T, ens_obj.state[0, :]))
        assert np.isclose(resp, ens_obj.resp[0])
        

"""
End test_ens_io.py
"""