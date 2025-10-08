# Input YAML File Description

The inputs for `run_.py` come from a single YAML file. An example YAML file can be found in `tests/sample.yml`. This README includes descriptions of the various fields in an input YAML file.

## Ensemble Information (ens section)

- **type**: Type of ensemble output being used. Options:
  - `wrf`: WRF netCDF output.
- **state_path**: Path to the files containing the ensemble state (i.e., predictor in the ESA regression). Must include a {num} placeholder for the ensemble member number.
- **resp_path**: Path to the files used to compute the response function (i.e., predictand in the ESA regression). Must include a {num} placeholder for the ensemble member number.
- **fix_file**: Fix file needed to read the files in `state_path` and `resp_path`. Only required for MPAS netCDF output.
- **nmem**: Number of ensemble members.
- **verbose**: Integer representing the level of verbosity. The higher the integer, the more output is printed.
- **horiz_coord**: Desired horizontal coordinate. Options:
  - `idx`: Index (e.g., 0, 1, 2, ...).
  - `xy`: Meters from the center of the domain. Only available for idealized WRF runs.
- **vert_coord**: Desired vertical coordinate. Options:
  - `idx`: Index (e.g., 0, 1, 2, ...).
- **state**: Options related to the ensemble state:
  - **var**: Model variable used in ESA.
  - **subset**: Option to subset the model domain prior to performing ESA. See options in the "subsetting" section below.
- **response**: Options related to the response function:
  - **var**: Model variable used to compute the response function.
  - **reduction**: Reduction applied to the model variable used to compute the response function. The goal is to have a single number for each ensemble member. Options:
    - `max`: Maximum
    - `min`: Minimum
    - `mean`: Mean
    - `sum`: Summation
  - **subset**: Option to subset the model domain prior to computing the response function. See options in the "subsetting" section below.

### Subsetting Options

If `subset: True` in either the `state` or `response` section of the YAML, then the following parameters must also be set:

- **xlim**: List with two entries that correspond to the minimum and maximum x values of the desired domain. Units are consistent with the `horiz_coord` option.
- **ylim**: List with two entries that correspond to the minimum and maximum y values of the desired domain. Units are consistent with the `horiz_coord` option.
- **zlim**: List with two entries that correspond to the minimum and maximum z values of the desired domain. Units are consistent with the `vert_coord` option.

## Option to Compute Estimated Variance Differences (var\_diff section)

- **use**: Option to compute the estimated response function variance reduction from assimilating an observation at each model cell in the ensemble state.
- **ob_var**: Assumed observation error variance. Should be consistent with the ensemble state variable (`ens.state.var`)

## Output Options (out section)

- **nc_fname**: Output netCDF file name. May include:
  - `esa`: ESA value computed for each model cell in the ensemble state.
  - `pval`: P-value for the null hypothesis that the ESA regression slope is 0.
  - `var_diff`: Estimated response function variance reduction from assimilating an observation at the given model cell. 
  - `x`, `y`, `z`: Model coordinates using the same conventions specified by `horiz_coord` and `vert_coord`
