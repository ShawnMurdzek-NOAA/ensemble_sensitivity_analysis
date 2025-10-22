# Ensemble Sensitivity Analysis (ESA) in Python

Program for running Ensemble Sensitivity Analysis (ESA) in Python.

## Contents

- `main/`: Main code for computing ESA.
- `util/`: Utilities that are separate from the ESA program (e.g., plotting scripts).
- `test/`: Automated unit tests as well as a simple test case.

## Quick Start Guide

Start by downloading the code from GitHub:

`git clone https://github.com/ShawnMurdzek-NOAA/ensemble_sensitivity_analysis.git`

Next, configure the required Python environment. If conda is enabled, a new environment can be created by running the following, with `{ENV_PREFIX}` replaced with the desired install location for the new environment:

```
cd ensemble_sensitivity_analysis
conda env create -f environment.yml --prefix {ENV_PREFIX}
conda activate {ENV_PREFIX}
```

The program requires a single YAML input file. An example is provided here: `test/sample.yml`. Assuming that the Python environment is configured correctly (see above), the test case can be run using the following command:

`python run_esa.py ./test/sample.yml`

This test case uses the following inputs:
- `test/data/wrf/memXXX/wrfout.2009-04-15_20:45:00.TEST.nc`: Idealized WRF ensemble files used to determine the ensemble state.
- `test/data/wrf/memXXX/wrfout.2009-04-15_22:00:00.TEST.nc`: Idealized WRF ensemble files used to compute the response function.

If the program runs successfully, the following files will be created:
- `test/test_out.nc`: ESA output file

## Utility Programs

The `./util/` directory contains a number of independent programs that may be helpful for probing output from the main ESA program (`run_esa.py`). These include:

- `plot_esa_fields.py`: Create 2D horizontal cross sections of fields output from the ESA program.
- `find_ESA_field_local_extrema.py`: Find local extrema (e.g., min or max) in fields output from the ESA program.

For additional information regarding any of the utility programs, load the ESA program Python environment and run the following:

```
cd ./util
python <program name> -h
```

## ESA Overview

ESA is the linear regression slope between an a single model cell and variable at one time (i.e., the predictor or "state") to some response function derived from the model at a later time (i.e., the predictand or "response"). The ensemble provides multiple samples of the state and response, which allows for the computation of linear regression. In this program, the state consists of a 2D or 3D field with many model cells (e.g., 2-m temperature), whereas the response will be a single value for each ensemble (e.g., max reflectivity). The linear regression slope is computed separately for each model cell, resulting in a 2D or 3D field of regression slopes. This array of regression slopes is referred to as `esa` in the netCDF file produced by the program.

## References

- [Ancell and Hakim (2007)](https://doi.org/10.1175/2007MWR1904.1)
- [Hill et al. (2020)](https://doi.org/10.1175/MWR-D-20-0015.1)
- [Arseneau and Ancell (2023)](https://doi.org/10.1175/MWR-D-22-0352.1) [Table 2 has a nice list of example response functions]
