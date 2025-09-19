"""
Create WRF Test Dataset

shawn.s.murdzek@noaa.gov
"""

#---------------------------------------------------------------------------------------------------
# Import Modules
#---------------------------------------------------------------------------------------------------

import xarray as xr


#---------------------------------------------------------------------------------------------------
# Inputs
#---------------------------------------------------------------------------------------------------

# Number of ensemble members
nens = 3

# Hybrid levels to keep
nlvl = 45

# Timestamp for WRF files
time =  '2009-04-15_20:45:00'

# Path to WRF output files 
# (include {n} placeholder for member number and {t} placeholder for time)
path = '/work/ahouston/smurdzek/WRF_ens_skeb_tests/psi1e-4_t1e-5_ztau1800_exp1.83/skeb{n}/run/wrfout_d01_{t}'

# Fields to save
fields = ['XLAT', 'XLONG', 'W', 'T', 'T2', 'PSFC', 'REFL_10CM', 'RAINNC']


#---------------------------------------------------------------------------------------------------
# Program
#---------------------------------------------------------------------------------------------------

# Save WRF fields
for n in range(1, nens+1):
    print(f'Creating test data for member {n}')
    ds = xr.open_dataset(path.format(n=n, t=time))

    # Only keep 4 hybrid levels
    ds = ds.sel(bottom_top=slice(0, nlvl))
    ds = ds.sel(bottom_top_stag=slice(0, nlvl+1))

    # Save desired fields
    out = {}
    for f in fields:
        out[f] = ds[f]
    out_ds = xr.Dataset(out, attrs=ds.attrs)

    # Save to netCDF
    out_ds.to_netcdf(f"./mem{n:03d}/wrfout.{time}.TEST.nc")


"""
End create_test_wrf_nc.py
"""
