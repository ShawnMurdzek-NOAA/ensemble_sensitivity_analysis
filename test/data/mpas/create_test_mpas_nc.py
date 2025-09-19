"""
Create MPAS Test Dataset

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
nlvl = 4

# Timestamp for MPAS files
time =  '2024-05-08_16.00.00'

# MPAS mesh info file
info_fname = '/mnt/lfs6/BMC/wrfruc/murdzek/test_ens_data/mpas/south3.5km.invariant.nc_L60_GEFS'

# Mesh info fields to save
info_fields = ['latCell', 'lonCell', 'zgrid', 'ter']

# Path to MPAS atmospheric output files 
# (include {n} placeholder for member number and {t} placeholder for time)
path = '/mnt/lfs6/BMC/wrfruc/murdzek/test_ens_data/mpas/mem{n:03d}/mpasout.{t}.nc'

# Atmospheric fields to save
fields = ['qv', 'theta', 'surface_pressure', 't2m', 'refl10cm_max']


#---------------------------------------------------------------------------------------------------
# Program
#---------------------------------------------------------------------------------------------------

# Save MPAS mesh info
print('Creating netCDF file with mesh info')
ds = xr.open_dataset(info_fname)
ds = ds.sel(nVertLevels=slice(0, nlvl))
ds = ds.sel(nVertLevelsP1=slice(0, nlvl+1))
out_info = {}
for f in info_fields:
    out_info[f] = ds[f]
out_info_ds = xr.Dataset(out_info, attrs=ds.attrs)
out_info_ds.to_netcdf("./invariant_TEST.nc")

# Save MPAS atmospheric fields
for n in range(1, nens+1):
    print(f'Creating test data for member {n:03d}')
    ds = xr.open_dataset(path.format(n=n, t=time))

    # Only keep 4 hybrid levels
    ds = ds.sel(nVertLevels=slice(0, nlvl))

    # Save desired fields
    out = {}
    for f in fields:
        out[f] = ds[f]
    out_ds = xr.Dataset(out, attrs=ds.attrs)

    # Save to netCDF
    out_ds.to_netcdf(f"./mem{n:03d}/mpasout.{time}.TEST.nc")


"""
End create_test_mpas_nc.py
"""
