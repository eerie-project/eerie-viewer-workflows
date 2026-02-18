import importlib
import json
import os
from pathlib import Path
from time import perf_counter

import pandas
import xarray
import zarr
from dask.diagnostics import ProgressBar
from dotenv import load_dotenv

from eerieview.constants import AVISO_VARIABLES, OCEAN_VARIABLES
from eerieview.data_processing import fix_360_longitudes
from eerieview.logger import get_logger
from eerieview.zarr import get_filesystem

load_dotenv()
logger = get_logger(__name__)

member2shortmeber = {
    "icon-esm-er-eerie-control-1950": "icon",
    "ifs-fesom2-sr-eerie-control-1950": "ifs-fesom2",
    "ifs-fesom2-sr-hist-1950": "ifs-fesom2",
    "icon-esm-er-hist-1950": "icon",
    "ifs-nemo-er-hist-1950": "ifs-nemo-er",
    "HadGEM3-GC5-EERIE-N216-ORCA025-eerie-historical": "hadgem3-mediumres",
    "HadGEM3-GC5-EERIE-N640-ORCA12-eerie-historical": "hadgem3-hires",
    "HadGEM3-GC5-EERIE-N96-ORCA1-eerie-historical": "hadgem3-lowres",
    "ifs-amip-tco1279-hist": "ifs-amip-tco1279-hist",
    "ifs-amip-tco1279-hist-c-0-a-lr20": "ifs-amip-tco1279-hist-c-0-a-lr20",
    "ifs-amip-tco399-hist-c-0-a-lr20": "ifs-amip-tco399-hist-c-0-a-lr20",
    "ifs-amip-tco399-hist-c-lr20-a-0": "ifs-amip-tco399-hist-c-lr20-a-0",
    "ifs-amip-tco399-hist": "ifs-amip-tco399-hist",
    "icon-esm-er-highres-future-ssp245": "icon",
    "ifs-fesom2-sr-highres-future-ssp245": "ifs-fesom2",
    "ifs-nemo-er-highres-future-ssp245": "ifs-nemo-er",
    "HadGEM3-GC5-EERIE-N216-ORCA025-eerie-ssp245": "hadgem3-mediumres",
    "HadGEM3-GC5-EERIE-N640-ORCA12-eerie-ssp245": "hadgem3-hires",
    "HadGEM3-GC5-EERIE-N96-ORCA1-eerie-ssp245": "hadgem3-lowres",
}


def get_merged_dataset(ifiles, chunks, drop_member: bool = False):
    """Open multiple NetCDF files, align variables, and merge into one dataset."""
    to_merge = [
        xarray.open_dataset(f)
        .drop_vars(
            [
                "height2m",
                "height10m",
                "height_2",
                "lev",
                "latitude_longitude",
                "lon_bnds",
                "lat_bnds",
            ],
            errors="ignore",
        )
        .chunk(chunks)
        for f in ifiles
    ]
    if drop_member:
        to_merge = [ds.drop_vars("member") for ds in to_merge]
    dataset = xarray.merge(to_merge, join="outer")
    return dataset


def get_encoding(ds: xarray.Dataset, chunks: dict[str, int]):
    """Build Zarr encoding with float32 dtype and variable-specific chunk layout."""
    encoding = {}
    for v in ds.data_vars:
        # Match chunks to dimension order of the variable
        var_chunks = tuple(chunks.get(d, -1) for d in ds[v].dims)
        encoding[v] = dict(dtype="float32", chunks=var_chunks)
    return encoding


def shorten_members(dataset):
    """Map verbose member IDs to shorter names used by downstream consumers."""
    dataset["member"] = dataset["member"].to_index().map(member2shortmeber)
    assert not dataset.member.isnull().any()
    return dataset


def get_zarr_url(bucket: str, base_path: str, default_prefix: str | None = None) -> str:
    """Build a normalized S3 URL and optionally inject a configurable path prefix.

    `ZARR_DESTINATION_PREFIX` controls the optional folder that sits between the
    bucket name and the dataset path. If the variable is unset, `default_prefix`
    is used. Set `ZARR_DESTINATION_PREFIX` to an empty string to disable prefixes.
    """
    prefix = os.getenv("ZARR_DESTINATION_PREFIX")
    if prefix is None:
        prefix = default_prefix
    parts = [p.strip("/") for p in [prefix, base_path] if p and p.strip("/")]
    return f"s3://{bucket}/{'/'.join(parts)}"


def log_upload_config(
    upload_name: str, zarr_url: str, ifiles: list[str], chunks: dict[str, int]
) -> None:
    """Log the key configuration for one upload operation."""
    logger.info(
        f"[{upload_name}] target={zarr_url} inputs={len(ifiles)} chunks={chunks}"
    )
    logger.info(f"[{upload_name}] input_files={ifiles}")


def log_dataset_state(upload_name: str, dataset: xarray.Dataset, stage: str) -> None:
    """Log dataset shape and variable count at a processing stage."""
    dims_str = ", ".join(f"{dim}={size}" for dim, size in dataset.sizes.items())
    logger.info(
        f"[{upload_name}] {stage}: vars={len(dataset.data_vars)} dims=({dims_str})"
    )


def write_dataset_to_zarr(
    dataset: xarray.Dataset,
    zarr_url: str,
    encoding: dict[str, dict],
    fs,
    upload_name: str,
) -> None:
    """Write a prepared dataset to Zarr while logging lifecycle and duration."""
    store = zarr.storage.FSStore(zarr_url, fs=fs)
    if fs.exists(zarr_url):
        logger.info(f"[{upload_name}] clearing existing store at {zarr_url}")
        fs.rm(zarr_url, recursive=True)
    start = perf_counter()
    logger.info(f"[{upload_name}] writing dataset to {zarr_url}")
    with ProgressBar():
        dataset.to_zarr(
            store=store, zarr_format=2, consolidated=True, encoding=encoding, mode="w"
        )
    elapsed = perf_counter() - start
    logger.info(f"[{upload_name}] upload complete in {elapsed:.1f}s")


def upload_eerie_climatologies(
    variables: list[str], product: str = "clim", experiment: str = "control", grid="025"
):
    """Upload EERIE climatology/trend products for a given experiment to Zarr."""
    idir = Path(os.environ["PRODUCTSDIR"], "decadal")
    bucket = os.environ["S3_BUCKET"]
    # Keep legacy "test" behavior by default, but allow overriding through
    # ZARR_DESTINATION_PREFIX (for example: "prod", "staging", or empty).
    zarr_url = get_zarr_url(
        bucket, f"decadal/{experiment}_EERIE_{product}.zarr", default_prefix="test"
    )
    ifiles = [
        f"{idir}/{varname}_{experiment}_EERIE_{product}.nc" for varname in variables
    ]
    if grid == "025":
        latchunk, lonchunk = 721, 1440
    elif grid == "125":
        latchunk, lonchunk = 1440, 2880
    else:
        raise RuntimeError(f"Unsupported {grid=}")
    chunks = dict(member=1, period=-1, time_filter=1, lat=latchunk, lon=lonchunk)
    upload_name = f"eerie-climatologies:{experiment}:{product}:{grid}"
    log_upload_config(upload_name, zarr_url, ifiles, chunks)
    dataset = get_merged_dataset(ifiles, chunks)
    log_dataset_state(upload_name, dataset, "merged")
    dataset = set_cmor_metadata(dataset, product)
    dataset = shorten_members(dataset)
    log_dataset_state(upload_name, dataset, "metadata-applied")
    encoding = get_encoding(dataset, chunks)
    fs = get_filesystem()
    write_dataset_to_zarr(dataset, zarr_url, encoding, fs, upload_name)


def get_obs_file(idir: Path, varname: str, product: str, region_set: str | None = None):
    """Return the expected observation file path for a variable and product."""
    if varname in AVISO_VARIABLES:
        source = "aviso"
    else:
        source = "era5"
    if region_set is None:
        obs_file = f"{idir}/{varname}_{source}_{product}.nc"
    else:
        obs_file = f"{idir}/{varname}_{source}_{region_set}_{product}.nc"
    return obs_file


def upload_obs_climatologies(variables: list[str], product="clim"):
    """Upload observed climatology/trend products to Zarr."""
    logger.info(f"Uploading obs for {product=}")
    idir = Path(os.environ["PRODUCTSDIR"], "decadal")
    bucket = os.environ["S3_BUCKET"]
    zarr_url = get_zarr_url(bucket, f"decadal/obs_{product}.zarr")
    ifiles = [get_obs_file(idir, varname, product) for varname in variables]
    chunks = dict(period=1, time_filter=1, lat=721, lon=1440)
    upload_name = f"obs-climatologies:{product}"
    log_upload_config(upload_name, zarr_url, ifiles, chunks)
    dataset = get_merged_dataset(ifiles, chunks)
    log_dataset_state(upload_name, dataset, "merged")
    dataset = dataset.drop_vars(["height2m", "height10m", "height_2"], errors="ignore")
    dataset = set_cmor_metadata(dataset, product)
    log_dataset_state(upload_name, dataset, "metadata-applied")
    encoding = get_encoding(dataset, chunks)
    fs = get_filesystem()
    write_dataset_to_zarr(dataset, zarr_url, encoding, fs, upload_name)


def upload_eerie_time_series(variables: list[str], experiment: str, region_set: str):
    """Upload EERIE regional time-series products for one experiment."""
    logger.info(f"Uploading EERIE time series for {experiment=} {region_set=}")
    idir = Path(os.environ["PRODUCTSDIR"], "time_series")
    bucket = os.environ["S3_BUCKET"]
    zarr_url = get_zarr_url(
        bucket, f"time_series/{experiment}_EERIE_{region_set}_ts.zarr"
    )
    ifiles = [
        f"{idir}/{varname}_{experiment}_EERIE_{region_set}_ts.nc"
        for varname in variables
    ]
    chunks = dict(time_filter=1, time=-1, region=1)
    upload_name = f"eerie-time-series:{experiment}:{region_set}"
    log_upload_config(upload_name, zarr_url, ifiles, chunks)
    dataset = get_merged_dataset(ifiles, chunks)
    log_dataset_state(upload_name, dataset, "merged")
    dataset = dataset.drop_vars(["height2m", "height10m", "height_3"], errors="ignore")
    dataset = set_cmor_metadata(dataset, "ts")
    if experiment != "hist-amip":
        dataset = shorten_members(dataset)
    log_dataset_state(upload_name, dataset, "metadata-applied")
    encoding = get_encoding(dataset, chunks)
    fs = get_filesystem()
    write_dataset_to_zarr(dataset, zarr_url, encoding, fs, upload_name)


def upload_obs_time_series(variables: list[str], region_set: str):
    """Upload observed regional time-series products to Zarr."""
    logger.info(f"Uploading obs time series for {region_set=}")
    idir = Path(os.environ["PRODUCTSDIR"], "time_series")
    bucket = os.environ["S3_BUCKET"]
    zarr_url = get_zarr_url(bucket, f"time_series/obs_{region_set}_ts.zarr")
    ifiles = [
        get_obs_file(idir, varname, "ts", region_set=region_set)
        for varname in variables
    ]
    chunks = dict(time_filter=1, time=-1, region=1)
    upload_name = f"obs-time-series:{region_set}"
    log_upload_config(upload_name, zarr_url, ifiles, chunks)
    dataset = get_merged_dataset(ifiles, chunks, drop_member=True)
    log_dataset_state(upload_name, dataset, "merged")
    dataset = dataset.drop_vars(["height2m", "height10m"], errors="ignore")
    dataset = set_cmor_metadata(dataset, "ts")
    dataset = dataset.chunk(chunks)
    log_dataset_state(upload_name, dataset, "metadata-applied")
    encoding = get_encoding(dataset, chunks)
    fs = get_filesystem()
    write_dataset_to_zarr(dataset, zarr_url, encoding, fs, upload_name)


def upload_eddy_rich_zarr():
    """Upload the single-level EDDY-rich ocean velocity sample dataset."""
    variables = ["uo", "vo"]
    ifile = Path(
        os.environ["PRODUCTSDIR"],
        "misc",
        "icon-esm-er.hist-1950_u_v_ocean_197001_19700212_weekly.nc",
    )
    bucket = os.environ["S3_BUCKET"]
    zarr_url = get_zarr_url(
        bucket, "misc/icon-esm-er.hist-1950_u_v_ocean_19700101.zarr"
    )
    upload_name = "eddy-rich:single-level"
    log_upload_config(upload_name, zarr_url, [str(ifile)], dict(lat=-1, lon=-1))
    dataset = xarray.open_dataset(ifile).squeeze().rename(u="uo", v="vo")
    dataset = fix_360_longitudes(dataset)
    dataset = dataset.drop_vars(["depth", "time"], errors="ignore")
    dataset = set_cmor_metadata(dataset, "clim")
    chunks = dict(lat=-1, lon=-1)
    log_dataset_state(upload_name, dataset, "metadata-applied")
    encoding = get_encoding(dataset, chunks)
    fs = get_filesystem()
    write_dataset_to_zarr(dataset, zarr_url, encoding, fs, upload_name)


def upload_eddy_rich_zarr_5lev():
    """Upload the 5-level EDDY-rich ocean velocity sample dataset."""
    variables = ["uo", "vo"]
    ifile = Path(
        os.environ["PRODUCTSDIR"],
        "misc",
        "icon-esm-er.hist-1950_u_v_ocean_19700101_5lev.nc",
    )
    bucket = os.environ["S3_BUCKET"]
    zarr_url = get_zarr_url(
        bucket, "misc/icon-esm-er.hist-1950_u_v_ocean_19700101_5lev.zarr"
    )
    upload_name = "eddy-rich:five-level"
    log_upload_config(
        upload_name, zarr_url, [str(ifile)], dict(depth=1, lat=-1, lon=-1)
    )
    dataset = xarray.open_dataset(ifile).squeeze().rename(u="uo", v="vo")
    dataset = dataset.drop_vars(["time"], errors="ignore")
    dataset = set_cmor_metadata(dataset, "clim")
    chunks = dict(depth=1, lat=-1, lon=-1)
    log_dataset_state(upload_name, dataset, "metadata-applied")
    encoding = get_encoding(dataset, chunks)
    fs = get_filesystem()
    write_dataset_to_zarr(dataset, zarr_url, encoding, fs, upload_name)


def get_variable_cmor_metadata(varname: str):
    """Read CMOR metadata for one variable from the bundled realm table."""
    realm = "Omon" if varname in OCEAN_VARIABLES else "Amon"
    cmor_json = Path(
        str(importlib.resources.files("eerieview")),
        f"resources/EERIE_{realm}.json",
    )
    with open(cmor_json, "r") as fileobj:
        table = json.load(fileobj)["variable_entry"]
    df = pandas.DataFrame.from_dict(table, orient="index")
    return df.loc[varname]


def set_cmor_metadata(dataset: xarray.Dataset, product) -> xarray.Dataset:
    """Populate CMOR-like metadata fields used by the viewer in each variable."""
    for varname in dataset.data_vars:
        varname_noanom = str(varname).replace("_anom", "").replace("_pvalue", "")
        if varname_noanom == "eke":
            attrs = dict(
                standard_name="eddy_kinetic_energy",
                long_name="Eddy Kinetic Energy",
                units="m2/s2",
            )
        else:
            attrs = get_variable_cmor_metadata(varname_noanom)
        for attrname in ["long_name", "standard_name", "units"]:
            attrval = attrs[attrname]
            if attrname == "units":
                # Viewer plots use human-readable units for these variables.
                if varname_noanom in ["tas", "tasmin", "tasmax", "tos"]:
                    attrval = "degC"
                if varname_noanom == "pr":
                    attrval = "mm day-1"
            if "anom" in str(varname):
                # Tag anomaly variables while preserving CMOR base naming.
                if attrname == "standard_name":
                    attrval += "_anomaly"
                if attrname == "long_name":
                    attrval += " Anomaly"
            dataset[varname].attrs[attrname] = attrval
        if product == "trend":
            dataset[varname].attrs["standard_name"] += "_trend"
            dataset[varname].attrs["long_name"] += " Trend"
    return dataset


def upload_time_series(
    variables: list[str], variables_amip: list[str], region_set: str
):
    """Upload all observed and modeled time-series datasets for one region set."""
    upload_obs_time_series(variables, region_set)
    upload_eerie_time_series(variables, "hist", region_set)
    upload_eerie_time_series(variables_amip, "hist-amip", region_set)
    upload_eerie_time_series(variables, "control", region_set)
    upload_eerie_time_series(variables, "future", region_set)


def main():
    """Entry point for batch upload of decadal and time-series EERIE products."""
    # Uncomment/select subsets while iterating locally to avoid full re-uploads.
    # Destination endpoint/bucket are controlled by S3_ENDPOINT_URL and S3_BUCKET.
    # Optional path prefix is controlled by ZARR_DESTINATION_PREFIX.
    variables = [
        "sfcWind",
        "uas",
        "vas",
        "tas",
        "pr",
        "tos",
        "clt",
        "tasmax",
        "tasmin",
        "zos",
        "eke",
    ]
    variables_amip = [v for v in variables if v not in ["zos", "eke", "so"]]
    for product in ["clim", "trend"]:
        upload_obs_climatologies(variables, product=product)
        for experiment in ["future", "hist", "control", "hist-amip"]:
            if experiment == "hist-amip":
                variables_exp = variables_amip
            else:
                variables_exp = variables
            logger.info(f"Uploading {product=} for {experiment=}")
            upload_eerie_climatologies(
                variables_exp, product=product, experiment=experiment, grid="025"
            )
    upload_time_series(variables, variables_amip, "IPCC")
    upload_time_series(variables, variables_amip, "EDDY")


if __name__ == "__main__":
    main()
