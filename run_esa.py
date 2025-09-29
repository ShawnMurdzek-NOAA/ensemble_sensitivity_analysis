"""
Main Driver for the Python-Based ESA

Command-Line Arguments
----------------------
sys.argv[1] : YAML input file

shawn.s.murdzek@noaa.gov
"""

#---------------------------------------------------------------------------------------------------
# Import Modules
#---------------------------------------------------------------------------------------------------

import sys
import datetime as dt
import yaml
import numpy as np
import copy

from main import ens_io


#---------------------------------------------------------------------------------------------------
# Functions
#---------------------------------------------------------------------------------------------------

def read_param(fname):
    """
    Read input YAML file and reformat plotting options in input parameters.

    Parameters
    ----------
    param : dictionary
        Input parameters
    
    Returns
    -------
    param : dictionary
        Input parameters

    """

    # Read input parameters
    with open(fname, 'r') as fptr:
        param = yaml.safe_load(fptr)
    
    return param


def read_ensemble(param):
    """
    Read ensemble data

    Parameters
    ----------
    param : dictionary
        Input parameters
    
    Returns
    -------
    ens_obj : esa.ens_data object
        Ensemble data

    """

    state_fnames = [param['ens']['state_path'].format(num=n) for n in range(1, param['ens']['nmem'] + 1)]
    resp_fnames = [param['ens']['resp_path'].format(num=n) for n in range(1, param['ens']['nmem'] + 1)]
    ens_obj = ens_io.read_ens(state_fnames,
                              resp_fnames,
                              param['ens']['state'],
                              param['ens']['response'],
                              horiz_coord=param['ens']['horiz_coord'],
                              vert_coord=param['ens']['vert_coord'],
                              verbose=param['ens']['verbose'],
                              fix_fname=param['ens']['fix_file'],
                              ftype=param['ens']['type'])

    return ens_obj


def save_output(ens_obj, param):
    """
    Wrapper function for saving ESA output

    Parameters
    ----------
    ens_obj : esa.ens_data object
        Ensemble data with ESA fields
    param : dictionary
        Input parameters

    Returns
    -------
    None.

    """
    
    attrs = {'horiz_coord': param['ens']['horiz_coord'],
             'vert_coord': param['ens']['vert_coord']}
    
    ens_obj.save_to_netcdf(param['out']['nc_fname'], attrs=attrs)


def print_max(ens_obj, fields=None):
    """
    Print location of max ESA values

    Parameters
    ----------
    ens_obj : esa.ens_data object
        DESCRIPTION.
    fields : list or None, optional
        List of fields to find the max of. If None, all fields are used

    Returns
    -------
    None.

    """
    
    # Populate fields if None
    if fields is None:
        fields = []
        all_fields = ['esa', 'var_diff']
        for f in all_fields:
            if hasattr(ens_obj, f):
                fields.append(f)
    
    # Create a second fields list
    fields2 = copy.deepcopy(fields)
    additional_fields2 = ['pval']
    for f in additional_fields2:
        if hasattr(ens_obj, f) and (f not in fields2):
            fields2.append(f)
    
    # Print location of maxima
    print()
    print(40*'-')
    for f in fields:
        idx = np.argmax(np.abs(getattr(ens_obj, f)))
        print(f"Abs max {f} location: x = {ens_obj.x[idx]}, y = {ens_obj.y[idx]}, z = {ens_obj.z[idx]}")
        s = ''
        for f in fields2:
            s = s + f"{f} = {getattr(ens_obj, f)[idx]:.3e} "
        print(s)
        print()
        


#---------------------------------------------------------------------------------------------------
# Main Code
#---------------------------------------------------------------------------------------------------

if __name__ == '__main__':

    start = dt.datetime.now()
    print('\n-----------------------------------------------')
    print("Starting ESA Program")

    # Read input YAML file
    param = read_param(sys.argv[1])

    # Read ensemble data
    print('\nReading ensemble data')
    ens_obj = read_ensemble(param)
    
    # Compute ESA
    print('Computing ESA')
    ens_obj.compute_esa()
    if param['var_diff']['use']:
        print('Computing variance difference')
        ens_obj.compute_variance_diff(param['var_diff']['ob_var'])
        
    # Save output
    print('Saving output')
    save_output(ens_obj, param)
    
    # Print max values
    print_max(ens_obj)

    print(f'\ntotal elapsed time = {(dt.datetime.now() - start).total_seconds()} s')


"""
End run_esa.py
"""