"""
Functions for ESA Ensemble I/O

The computation of the response function is also handled here

shawn.s.murdzek@noaa.gov
"""

#---------------------------------------------------------------------------------------------------
# Import Modules
#---------------------------------------------------------------------------------------------------

import xarray as xr
import numpy as np


#---------------------------------------------------------------------------------------------------
# Contents
#---------------------------------------------------------------------------------------------------

class ens_data():
    """
    Class to handle processed ensemble output

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
    

def read_parse_wrf(state_fnames,
                   resp_fnames,
                   state_param,
                   resp_param,
                   horiz_coord='idx',
                   vert_coord='idx',
                   verbose=0):
    """
    Read and parse WRF netCDF input

    Parameters
    ----------
    state_fnames : list
        NetCDF files containing WRF fields at the initial time. Each entry is a different ensemble 
        member
    resp_fnames : list
        NetCDF files containing WRF fields at the response time. Each entry is a different ensemble 
        member
    state_param : dictionary
        Dictionary of specifications for the ensemble state. 
        Required keys: 'var' and 'subset'
    resp_param : dictionary
        Dictionary of specifications fo the ensemble response function.
        Required keys: 'var', 'reduction', and 'subset'
    horiz_coord : string
        Horizontal coordinates options: 'idx', 'xy'
    vert_coord : string
        Horizontal coordinates options: 'idx'
    verbose : int, optional
        Verbosity level

    Returns
    -------
    ens_data object
        Ensemble output for ESA

    """

    check_read_parse_inputs(state_fnames, resp_fnames, state_param, resp_param)

    # Read in ensemble state and response function
    if verbose > 0: print('  Reading WRF information')
    state_ls = []
    resp = np.zeros(len(resp_fnames))
    for i, (fnames, param) in enumerate(zip([resp_fnames, state_fnames], [resp_param, state_param])):
        for j, f in enumerate(fnames):
            ds = xr.open_dataset(f)
            
            # Read in entire field and remove time dimension. Field should be 2D or 3D
            field = ds[param['var']].values[0, :]
            field_meta = ds[param['var']].attrs
            ndim = len(field.shape)
                
            # Determine length of each dimension
            if ndim == 2:
                ny, nx = field.shape
            elif ndim == 3:
                nz, ny, nx = field.shape
            else:
                raise ValueError(f"State field should be 2D or 3D (ndim = {ndim})")
            
            # Extract horizontal coordinates
            if horiz_coord == 'idx':
                x = np.arange(0, nx, dtype=int)
                y = np.arange(0, ny, dtype=int)
            elif horiz_coord == 'xy':
                dx = ds.attrs['DX']
                if nx == ds['west_east'].size:
                    x = dx * ds['west_east'].values
                    x = x - 0.5*x[-1]
                elif nx == ds['west_east_stag'].size:
                    x = dx * ds['west_east_stag'].values
                    x = x - 0.5*(x[-1] + dx)
                else:
                    raise ValueError(f"nx ({nx}) does not match size of x coordinate")
                dy = ds.attrs['DY']
                if ny == ds['south_north'].size:
                    y = dy * ds['south_north'].values
                    y = y - 0.5*y[-1]
                elif ny == ds['south_north_stag'].size:
                    y = dy * ds['south_north_stag'].values
                    y = y - 0.5*(y[-1] + dy)
                else:
                    raise ValueError(f"ny ({ny}) does not match size of y coordinate")
            else:
                raise ValueError(f"Invalid option for horiz_coord: {horiz_coord}")
          
            # Extract vertical coordinates
            if ndim == 3:
                if vert_coord == 'idx':
                    z = np.arange(0, nz, dtype=int)
                else:
                    raise ValueError(f"Invalid option for vert_coord: {vert_coord}")
            else:
                z = np.zeros(1, dtype=int)
               
            # Perform subsetting
            # Do vertical subsetting first, then horizontal subsetting
            if param['subset']:
                if ndim == 3:
                    zind = np.where(np.logical_and(z >= param['zlim'][0],
                                                   z <= param['zlim'][1]))[0]
                    z = z[zind]
                    field = field[zind[0]:zind[-1]+1, :]
                if horiz_coord in ['idx', 'xy']:
                    xind = np.where(np.logical_and(x >= param['xlim'][0],
                                                   x <= param['xlim'][1]))[0]
                    yind = np.where(np.logical_and(y >= param['ylim'][0],
                                                   y <= param['ylim'][1]))[0]
                    x = x[xind]
                    y = y[yind]
                    if ndim == 2:
                        field = field[yind[0]:yind[-1]+1, xind[0]:xind[-1]+1]
                    elif ndim == 3:
                        field = field[:, yind[0]:yind[-1]+1, xind[0]:xind[-1]+1]
                else:
                    raise ValueError(f"Cannot perform subsetting with horiz_coord {horiz_coord}")
            
            # Perform reduction
            if 'reduction' in param.keys():
                resp[j] = reduce_field(field, reduction=param['reduction'])
                resp_meta = field_meta
                resp_meta['reduction'] = param['reduction']
            
            # Turn state array and coordinates into 1D arrays
            else:
                y, z, x  = np.meshgrid(y, z, x)
                x = np.ravel(x)
                y = np.ravel(y)
                z = np.ravel(z)
                state_ls.append(np.ravel(field))
                state_meta = field_meta
    
    # Convert state lists into an array
    state = np.array(state_ls)

    return ens_data(state, resp, x, y, z, state_meta, resp_meta)


def reduce_field(field, reduction='max', kw={}):
    """
    Reduce a 2D or 3D field into a single value

    Parameters
    ----------
    field : np.array
        Model field
    reduction : string, optional
        Reduction method. The default is 'max'.
    kw : dictionary, optional
        Additional keyword arguments used for the reduction

    Returns
    -------
    val : float
        Result of reduction

    """
    
    if reduction == 'max':
        val = np.amax(field)
    elif reduction == 'min':
        val = np.amin(field)
    elif reduction == 'mean':
        val = np.mean(field)
    elif reduction == 'sum':
        val = np.sum(field)
    elif reduction == 'npts_gt_thres':
        val = np.sum(field > kw['thres'])
    elif reduction == 'npts_lt_thres':
        val = np.sum(field < kw['thres'])
    
    return val


def check_read_parse_inputs(state_fnames, resp_fnames, state_param, resp_param):
    """
    Check inputs for the various read_parse functions in this file
    
    Parameters
    ----------
    state_fnames : list
        NetCDF files containing WRF fields at the initial time. Each entry is a different ensemble 
        member
    resp_fnames : list
        NetCDF files containing WRF fields at the response time. Each entry is a different ensemble 
        member
    state_param : dictionary
        Dictionary of specifications for the ensemble state. 
        Required keys: 'var' and 'subset'
    resp_param : dictionary
        Dictionary of specifications fo the ensemble response function.
        Required keys: 'var', 'reduction', and 'subset'
    
    Returns
    -------
    None
    
    """
    
    # Check that both ensemble sizes are the same
    if len(state_fnames) != len(resp_fnames):
        print('In main.ens_io.read_parse_wrf')
        print(f"length of state_fnames = {len(state_fnames)}")
        print(f"length of resp_fnames = {len(resp_fnames)}")
        raise ValueError("state_fnames and resp_fnames must have the same size.")
    
    # Check that the param dictionaries have the required keys
    req_key_state_param = ['var', 'subset']
    req_key_resp_param = ['var', 'reduction', 'subset']
    for k in req_key_state_param:
        if k not in state_param.keys():
            raise ValueError(f"Key {k} is missing from state_param")
    for k in req_key_resp_param:
        if k not in resp_param.keys():
            raise ValueError(f"Key {k} is missing from resp_param")
            
    return None


def read_ens(state_fnames, 
             resp_fnames,
             state_param,
             resp_param,
             horiz_coord='idx',
             vert_coord='idx',
             verbose=0, 
             fix_fname=None, 
             ftype='wrf'):
    """
    Read ensemble output

    Parameters
    ----------
    state_fnames : list
        Files containing model fields at the initial time. Each entry is a different ensemble 
        member
    resp_fnames : list
        Files containing model fields at the response time. Each entry is a different ensemble 
        member
    state_param : dictionary
        Dictionary of specifications for the ensemble state. 
        Required keys: 'var' and 'subset'
    resp_param : dictionary
        Dictionary of specifications fo the ensemble response function.
        Required keys: 'var', 'reduction', and 'subset'
    horiz_coord : string, optional
        Horizontal coordinates options
    vert_coord : string, optional
        Horizontal coordinates options
    verbose : int, optional
        Verbosity level
    fix_fname : string, optional
        File containing grid or mesh information. Only needed for MPAS output
    ftype : string, optional
        Input file type. Options: 'wrf'
    
    Returns
    -------
    ens_data object
        Ensemble output

    """

    if ftype == 'wrf':
        ens_obj = read_parse_wrf(state_fnames,
                                 resp_fnames,
                                 state_param,
                                 resp_param,
                                 horiz_coord='idx',
                                 vert_coord='idx',
                                 verbose=0)
    else:
        raise ValueError(f"ftype {ftype} is not recognized")

    return ens_obj


"""
End ens_io.py
"""