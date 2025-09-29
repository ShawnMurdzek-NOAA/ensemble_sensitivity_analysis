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
    ens_obj : ens_io.ens_data object
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

    print(f'\ntotal elapsed time = {(dt.datetime.now() - start).total_seconds()} s')


"""
End run_esa.py
"""