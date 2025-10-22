"""
Find Local Extrema in ESA fields

shawn.murdzek@colorado.edu
"""

#---------------------------------------------------------------------------------------------------
# Import Modules
#---------------------------------------------------------------------------------------------------

import datetime as dt
import sys
import argparse
import numpy as np
import xarray as xr
import pandas as pd
from skimage.feature import peak_local_max
import copy


#---------------------------------------------------------------------------------------------------
# Main Program
#---------------------------------------------------------------------------------------------------

def parse_in_args(argv):
    """
    Parse input arguments

    Parameters
    ----------
    argv : list
        Command-line arguments from sys.argv[1:]
    
    Returns
    -------
    Parsed input arguments

    """

    parser = argparse.ArgumentParser(description='Find local extrema in ESA fields from a netCDF file \
                                                  output by the \
                                                  ensemble_sensitivity_analysis program.')
    
    # Positional arguments
    parser.add_argument('in_file', 
                        help='Input netCDF file created by run_esa.py',
                        type=str)

    # Optional arguments
    parser.add_argument('-o',
                        dest='out_file',
                        default='extrema.csv',
                        help='Output CSV file name.',
                        type=str)
    
    parser.add_argument('-f',
                        dest='field',
                        default='esa',
                        help='Field to find extrema of',
                        type=str)

    parser.add_argument('-n',
                        dest='n',
                        default='3',
                        help='Number of local extrema to find',
                        type=int)

    parser.add_argument('-t',
                        dest='type',
                        default='minmax',
                        help="Type of local extrema. Options: 'max', 'min', 'minmax'",
                        type=str)
    
    parser.add_argument('--pval_thres',
                        dest='pval_thres',
                        default=-1,
                        help='Option to mask out values where pval > pval_thres prior to finding \
                              local extrema. Set to -1 to not use.',
                        type=float)

    parser.add_argument('--min_dist',
                        dest='min_dist',
                        default=10,
                        help='Minimum distance (in grid boxes) allowed between local extrema',
                        type=int)

    parser.add_argument('--thres_rel',
                        dest='thres_rel',
                        default=0,
                        help='Relative minimum intensity of local extrema values. Actual threshold is \
                              calculated as (field global extreme value) * thres_rel.',
                        type=float)

    return parser.parse_args(argv)


def find_extrema(ds, param):
    """
    Find local extrema for the desired ESA field
    """

    # Extract field
    if param.field in ds:
        f = copy.deepcopy(ds[param.field].values)
    else:
        raise KeyError(f"Field {f} is not in the input netCDF file")
    
    # Change field to account for extrema type
    if param.type == 'max':
        None
    elif param.type == 'min':
        f = -1 * f
    elif param.type == 'minmax':
        f = np.abs(f)
    else:
        raise ValueError(f"Extrema type {param.type} is not recognized")

    # Mask out large pvals
    if param.pval_thres > 0:
        if 'pval' in ds:
            f[ds['pval'] > param.pval_thres] = np.amin(f)
        else:
            print("Warning: 'pval' field not found. Skipping pval_thres masking")

    # Find extrema
    loc = peak_local_max(f, num_peaks=param.n, min_distance=param.min_dist, 
                         threshold_rel=param.thres_rel, exclude_border=False)

    return loc


def output_extrema(ds, loc, param):
    """
    Write extrema to a CSV file
    """

    # Extract indices
    out_dict = {'i':loc[:, 0],
                'j':loc[:, 1],
                'k':loc[:, 2]}

    # Extract other fields
    all_vars = ['x', 'y', 'z'] + [k for k in ds]
    for v in all_vars:
        out_dict[v] = np.zeros(loc.shape[0])
        for i in range(loc.shape[0]):
            out_dict[v][i] = ds[v][loc[i, 0], loc[i, 1], loc[i, 2]].values

    # Write metadata to CSV
    with open(param.out_file, 'w') as fptr:
        fptr.write(f"field = {param.field}, extrema type = {param.type}, pval_thres = {param.pval_thres}\n")

    # Write to CSV
    out_df = pd.DataFrame(out_dict)
    out_df.to_csv(param.out_file, mode='a')

    return out_df


if __name__ == '__main__':
    
    start = dt.datetime.now()
    print('Starting find_ESA_field_local_extrema.py')
    print(f"Time = {start.strftime('%Y%m%d %H:%M:%S')}\n")

    # Read in parameters and netCDF file
    param = parse_in_args(sys.argv[1:])
    print('Reading in ESA output')
    esa_ds = xr.open_dataset(param.in_file)
    
    # Find extrema
    print('Finding local extrema')
    extrema_loc = find_extrema(esa_ds, param)

    # Write extrema to output file
    print('Outputting results')
    out = output_extrema(esa_ds, extrema_loc, param)

    print('\nProgram finished!')
    print(f"Elapsed time = {(dt.datetime.now() - start).total_seconds()} s")


"""
End find_ESA_field_local_extrema.py
"""
