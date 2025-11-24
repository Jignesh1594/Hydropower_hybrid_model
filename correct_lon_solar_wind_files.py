import xarray as xr
import logging
import argparse

"""
This code is used to convert longitude values from 0–360 to –180–180 for the solar and wind capacity-factor files. 
First, convert the wind speed and downward irradiance data into wind and solar capacity factors, and then run this code.

"""

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)

def convert_longitude(ds):
    """
    Convert longtiude from 0-360 to -180-180

    Args:
    ds: xarray dataset

    Returns:
    None    

    """

    lon_name = "longitude"
    data = xr.open_dataset(ds)
    print(f"Processing {ds}")

    if lon_name is data.coords:
        raise ValueError(f"Longitude coordinate name is not {lon_name}")
    
    data[lon_name] = (data[lon_name] + 180) % 360 - 180
    data = data.sortby(lon_name)

    compression_setting = {var_name : {"zlib" : True, 'complevel' : 4} for var_name in data.data_vars}
    filename  = ds.split("/")[-1].split(".")[0] + "_correctlon.nc"
    final_path = ds.rsplit("/", 1)[0] + "/" + filename

    data.to_netcdf(final_path, encoding=compression_setting)


def main(filename):
    convert_longitude(ds = filename)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        "Correct Longitude from 0 to 360 to -180 to 180 for Solar and Wind Capacity factor files as they are calculated from ERA5 wind speed and shortwave downwelling"
    )
    parser.add_argument(
        "--filename",
        default="/scratch/shah0012/ERA5_Land_processed/wind_CF_factor/global_monthly_capacity_factor_wind_2000_2022.nc",
        help="Directory with file name (netcdf format) containing either wind or solar capacity factor file"
    )
    args = parser.parse_args()

    main(
        filename = args.filename
    )

 