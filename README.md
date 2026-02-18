# eerie-data-viewer-workflows

Functions and workflows for preparing data for visualization for the EERIE project (https://eerie-project.eu/) data viewer. The input
data is read from the EERIE intake catalogues and, in the case of observations,
from netCDF files.

IMPORTANT NOTE: The EERIE data is published in almost real time and is continuously updated.
It can contain issues like inconsistent units or other encoding errors. The scripts do correct some of
the issues and homogenize variable names, but some issues may persist or arise as the data is updated.
If you aim to use this script and EERIE cloud data please review the results critically. Do not consider
this "production ready".

The present code is also changing and evolving. Please report any bugs you may find in the issue tracker.

The following is a schematic description of the main workflow:

- Open the intake catalogue (can be “cloud” or “levante”).
- Loop over the variables.
- Open the entry in the catalogue as an xarray dataset.
- Rename the variable if needed together with coordinates.
- Loop over the analysis dimensions (time_filter, period). For each of them compute the product, regrid to a common grid using conservative regridding, add the analysis dimensions as dimensions to the xarray dataset and append it to a list.
- Merge all the products into a single dataset and save it to netCDF using dask.
- A separate script converts the netCDFs to zarr and uploads it to the storage, defining the chunks so there is a single chunk for each map or time series.

## Installation and configuration

It is recommended to clone the project and create a conda environment with the environment.yml file. Then
the project root can be simply added to the PYTHONPATH.

```commandline
git clone git@github.com:eerie-project/eerie-viewer-workflows.git
cd eerie-viewer-workflows
conda env create -f environment.yml -n eerieview
conda activate eerieview
export PYTHONPATH="$PWD"
cd scripts
python get_climatologies.py
```

The scripts load environment variables automatically using `python-dotenv` (`load_dotenv()`), so you can define
them in a `.env` file in the project root. Exported shell variables also work.

Environment variables used by the workflows:

```
S3_ENDPOINT_URL=""         # Object storage endpoint URL (S3-compatible)
S3_KEY=""                  # Object storage access key
S3_SECRET=""               # Object storage secret key
S3_BUCKET="eerie"          # Bucket name for reading/writing zarr data
ZARR_DESTINATION_PREFIX="" # Optional path prefix inside the bucket (e.g. "test", "prod"); empty disables prefix
PRODUCTSDIR=""             # Base directory for generated products (decadal/, time_series/, misc/)
OBSDIR=""                  # Directory containing downloaded observation inputs
DOWNLOADIR=""              # Base download directory (used by AVISO download/processing scripts)
DIAGSDIR=""                # Directory for diagnostics outputs (e.g. monthly EKE intermediates)
CDO_LOCATION=""            # Currently it is not used. Path to CDO binary (required by regridding workflows)
```

Example `.env`:

```dotenv
S3_ENDPOINT_URL="https://your-s3-endpoint"
S3_KEY="..."
S3_SECRET="..."
S3_BUCKET="eerie"
ZARR_DESTINATION_PREFIX="test"
PRODUCTSDIR="/path/to/products"
OBSDIR="/path/to/obs"
DOWNLOADIR="/path/to/downloads"
DIAGSDIR="/path/to/diagnostics"
CDO_LOCATION="/usr/bin/cdo"
```

## Main scripts available

The scripts/entrypoints are in the scripts folder. For them to run the root directory needs to be in the
PYTHONPATH. The following are the main scripts:

- get_climatologies: It computes the decadal products from the EERIE data, which are climatologies but also trends.
- get_obs_climatologies: Computes climatologies and trends for observations. Currently ERA5 and AVISO data are supported. These are not read from intake catalogues but need to be present as files in the disk.
- download_era5.py: Script to download ERA5 files.
- get_monthly_eke.py: Computes monthly Eddy Kinetic Energy from daily sea level data from EERIE models.
- get_aviso_monthly_variables.py: Computes monthly data from the AVISO observations.
- get_time_series.py: Computes regionally averaged time series from EERIE data.
- get_obs_time_series.py: Computes regionally averaged time series from the observations.
- get_global_temp_time_series.py: Computes global mean temperature time series products.
- upload_to_zarr.py: `get_climatologies.py` and related scripts generate netCDF files. This script merges and uploads them to object storage as zarr datasets.
- plot_stripes.py: This script is used for Quality Control. It generates figures by systematically reading all the fields from the zarr files, in order to inspect them looking for gaps or suspicious patterns.

## Workflow for developers/contributors

For best experience create a new conda environment (e.g. DEVELOP) with Python 3.11:

```
conda create -n DEVELOP -c conda-forge python=3.11
conda activate DEVELOP
```

Before pushing to GitHub, run the following commands:

1. Update conda environment: `make conda-env-update`
1. Install this package: `pip install -e .`
1. Run quality assurance checks: `make qa`
1. Run tests: `make unit-tests`
1. Run the static type checker: `make type-check`

## License

```
Copyright 2025, European Union.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
```
