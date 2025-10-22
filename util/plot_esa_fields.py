"""
Plot ESA fields

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
import matplotlib.pyplot as plt
import pandas as pd


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

    parser = argparse.ArgumentParser(description='Plot fields from a netCDF file output by the \
                                                  ensemble_sensitivity_analysis program.')
    
    # Positional arguments
    parser.add_argument('in_file', 
                        help='Input netCDF file created by run_esa.py',
                        type=str)

    # Optional arguments
    parser.add_argument('-o',
                        dest='out_file',
                        default='out_{k}.png',
                        help='Output image file name. Include {k} placeholder for vertical index',
                        type=str)
    
    parser.add_argument('-f',
                        dest='field',
                        default='esa',
                        help='Field to plot',
                        type=str)
    
    parser.add_argument('--pval_thres',
                        dest='pval_thres',
                        default=0.05,
                        help='Option to hatch areas where pval < pval_thres. Set to -1 to not use.',
                        type=float)
    
    parser.add_argument('--extrema_csv',
                        dest='extrema_csv',
                        default='none',
                        help="CSV file containing extrema values to be plotted. Set to 'none' to not use.",
                        type=str)
    
    return parser.parse_args(argv)


def read_extrema_csv(fname):
    """
    Read CSV holding extrema values
    """

    df = pd.read_csv(fname, skiprows=1)

    return df


def plot_cartesian(ds, param):
    """
    Plot ESA fields when horiz_coord is 'xy' or 'idx'
    """
   
    # Read in CSV with extrema
    if param.extrema_csv != 'none':
        extrema_df = read_extrema_csv(param.extrema_csv)

    # Make plot for each vertical level
    for k in range(ds[param.field].shape[0]):
    
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8, 8))
        plt.subplots_adjust(left=0.1, bottom=0, right=0.98, top=0.82)
        
        # Plot ESA-related field
        x = ds['x'][k, :, :].values
        y = ds['y'][k, :, :].values
        f = ds[param.field][k, :, :].values
        #vmax = np.percentile(f, 0.99)
        #vmin = np.percentile(f, 0.01)
        cax = ax.pcolormesh(x, y, f, cmap='plasma')
        
        # Add p-value hatching
        h_str = ''
        if param.pval_thres > 0:
            pval = ds['pval'][k, :, :].values
            nlvl = [0, param.pval_thres, 1]
            ax.contour(x, y, pval, nlvl, colors='k', linewidths=0.5)
            ax.contourf(x, y, pval, nlvl, colors='none', hatches=['\\\\', None])
            h_str = f"Hatching: ESA p-value < {param.pval_thres}"
        
        # Add local extrema
        e_str=''
        if param.extrema_csv != 'none':
            df = extrema_df.loc[extrema_df['i'] == k, ['x', 'y']]
            if len(df) > 0:
                ax.plot(df['x'], df['y'], c='c', marker='o', lw=0, ms=8)
                e_str = "Cyan dots: Local extrema"

        # Add annotations
        ax.set_aspect('equal')
        z_str = f"avg z = {np.mean(ds['z'][k, :, :].values)}"
        state_str = f"state = {ds.attrs['state_description']}"
        resp_str = f"response = {ds.attrs['response_reduction']} {ds.attrs['response_description']}"
        plt.suptitle(f"{state_str}, {z_str}\n{resp_str}\n{h_str}\n{e_str}", size=16)
        cbar = plt.colorbar(cax, ax=ax, orientation='horizontal', pad=0.075)
        cbar.set_label(param.field, size=12)
        
        plt.savefig(param.out_file.format(k=k))
        plt.close()
    
    return None


if __name__ == '__main__':
    
    start = dt.datetime.now()
    print('Starting plot_esa_fields.py')
    print(f"Time = {start.strftime('%Y%m%d %H:%M:%S')}")

    # Read in parameters and netCDF file
    param = parse_in_args(sys.argv[1:])
    esa_ds = xr.open_dataset(param.in_file)
    
    # Make plot
    if esa_ds.attrs['horiz_coord'] in ['xy', 'idx']:
        plot_cartesian(esa_ds, param)
    else:
        raise ValueError(f"horiz_coord option {esa_ds.attrs['horiz_coord']} is not supported")

    print('Program finished!')
    print(f"Elapsed time = {(dt.datetime.now() - start).total_seconds()} s")


"""
End plot_esa_fields.py
"""
