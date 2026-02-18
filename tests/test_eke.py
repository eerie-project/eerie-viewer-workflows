import numpy as np
import xarray as xr

from eerieview.eke import remove_smooth_climatology


def test_remove_smooth_climatology_chunked(tmp_path):
    """
    Test that remove_smooth_climatology handles temporally chunked data without
    producing unexpected NaNs.
    """
    # Create a synthetic dataset with daily data for 25 years
    years = 25
    times = xr.date_range("2000-01-01", periods=years * 365, freq="D")
    # Reduced size to speed up tests
    lats = np.linspace(-10, 10, 20)
    lons = np.linspace(0, 20, 20)

    # Fill with 1.0
    data = np.ones((len(times), len(lats), len(lons)))
    da = xr.DataArray(
        data,
        coords={"time": times, "lat": lats, "lon": lons},
        dims=("time", "lat", "lon"),
        name="zos",
    )

    # Chunk it in time - this was the trigger for the bug
    da_chunked = da.chunk({"time": 365, "lat": -1, "lon": -1})

    clim_file = tmp_path / "test_clim.zarr"

    # Run the function
    detrended = remove_smooth_climatology(da_chunked, clim_file)

    # Verify the output is still dask-backed
    assert detrended.chunks is not None

    # Compute the result for a middle year
    middle_year_data = detrended.sel(time="2012").compute()

    # The important part is that it's NOT NaN
    assert not np.isnan(middle_year_data).any()
    np.testing.assert_allclose(middle_year_data.values, 0.0, atol=1e-7)

    # Check if climatology file exists and is valid
    assert clim_file.exists()
    clim_ds = xr.open_zarr(clim_file)
    assert "zos" in clim_ds.data_vars
    assert not np.isnan(clim_ds.zos.sel(time="2012")).any()


def test_remove_smooth_climatology_existing_file(tmp_path):
    """Test that the function correctly reads an existing climatology file."""
    times = xr.date_range("2000-01-01", periods=30, freq="D")
    lats = [0]
    lons = [0]
    da = xr.DataArray(
        np.ones((30, 1, 1)),
        coords={"time": times, "lat": lats, "lon": lons},
        dims=("time", "lat", "lon"),
        name="zos",
    )

    clim_file = tmp_path / "existing_clim.zarr"
    # Create a fake climatology file
    clim_da = da * 0.5
    clim_da.to_dataset().to_zarr(
        clim_file
    )  # No special encoding here so small dims are OK

    # Calling the function should now use the existing file
    detrended = remove_smooth_climatology(da, clim_file)
    result = detrended.compute()

    # 1.0 - 0.5 = 0.5
    np.testing.assert_allclose(result.values, 0.5)
