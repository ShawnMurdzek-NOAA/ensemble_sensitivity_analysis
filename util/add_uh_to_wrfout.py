"""
Add Updraft Helicity to wrfout Files

shawn.s.murdzek@noaa.gov
"""

#---------------------------------------------------------------------------------------------------
# Import Modules
#---------------------------------------------------------------------------------------------------

import datetime as dt
import sys
import argparse
import netCDF4 as nc
import wrf
import numpy as np


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

    parser = argparse.ArgumentParser(description='Add updraft helicity to wrfout netCDF file')

    # Positional arguments
    parser.add_argument('in_file',
                        help='Input wrfout netCDF file',
                        type=str)

    # Optional arguments
    parser.add_argument('-n',
                        dest='name',
                        default='UH25',
                        help='Name of updraft helicity field',
                        type=str)

    parser.add_argument('--bottom',
                        dest='bottom',
                        default=2000,
                        help='Layer bottom for updraft helicity calculation (m)',
                        type=float)

    parser.add_argument('--top',
                        dest='top',
                        default=5000,
                        help='Layer top for updraft helicity calculation (m)',
                        type=float)

    return parser.parse_args(argv)


def compute_uh_field(fptr, param):
    """
    Compute updraft helicity and save to netCDF4 file pointer
    """

    uh = wrf.getvar(fptr, 'updraft_helicity', bottom=param.bottom, top=param.top)
    uh_nc = fptr.createVariable(param.name, 'f4', ('Time', 'south_north', 'west_east'))
    uh_nc.units = 'm2/s2'
    uh_nc.long_name = f"{param.bottom:.1f} - {param.top:.1f} m updraft helicity"
    uh_nc[:] = uh.values[np.newaxis, :, :]

    return fptr


if __name__ == '__main__':

    start = dt.datetime.now()
    print('Starting add_uh_to_wrfout.py')
    print(f"Time = {start.strftime('%Y%m%d %H:%M:%S')}\n")

    # Read in parameters and open netCDF file
    param = parse_in_args(sys.argv[1:])
    wrf_fptr = nc.Dataset(param.in_file, 'r+')

    # Compute UH
    print('Computing updraft helicity')
    wrf_fptr = compute_uh_field(wrf_fptr, param)
    wrf_fptr.close()

    print('\nProgram finished!')
    print(f"Elapsed time = {(dt.datetime.now() - start).total_seconds()} s")


"""
End add_uh_to_wrfout.py
"""
