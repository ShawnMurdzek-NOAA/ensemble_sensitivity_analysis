# Ensemble Sensitivity Analysis (ESA) in Python

Program for running Ensemble Sensitivity Analysis (ESA) in Python.

## Contents

- `main/`: Main code for computing ESA.
- `tests/`: Automated unit tests as well as a simple test case.

## Quick Start Guide

Start by downloading the code from GitHub:

`git clone https://github.com/ShawnMurdzek-NOAA/ensemble_sensitivity_analysis.git`

Next, configure the required Python environment. If conda is enabled, a new environment can be created by running the following, with `{ENV_PREFIX}` replaced with the desired install location for the new environment:

```
cd ensemble_sensitivity_analysis
conda env create -f environment.yml --prefix {ENV_PREFIX}
conda activate {ENV_PREFIX}
```

The program requires a single YAML input file. An example is provided here: `tests/sample.yml`. Assuming that the Python environment is configured correctly (see above), the test case can be run using the following command:

`python run_esa.py ./tests/sample.yml`

This test case uses the following inputs:
- `tests/data/wrf/memXXX/wrfout.2009-04-15_20:45:00.TEST.nc`: Idealized WRF ensemble files used to determine the ensemble state.
- `tests/data/wrf/memXXX/wrfout.2009-04-15_22:00:00.TEST.nc`: Idealized WRF ensemble files used to compute the response function.

If the program runs successfully, the following files will be created:
- `test.nc`: ESA output file

## References

- [Ancell and Hakim (2007)](https://doi.org/10.1175/2007MWR1904.1)
- [Hill et al. (2020)](https://doi.org/10.1175/MWR-D-20-0015.1)
- [Arseneau and Ancell (2023)](https://doi.org/10.1175/MWR-D-22-0352.1) [Table 2 has a nice list of example response functions]
