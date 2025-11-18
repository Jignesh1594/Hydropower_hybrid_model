import xarray as xr
import numpy as np
import os
import glob
import pandas as pd
import argparse
import logging



"""
ERA5 surface wind is measured at 10m height

Power law function to caculate wind speed at 100m height

V2 = V1 * (H2/H1)^a

V2 = wind speed at 100m
V1 = wind speed at 10m
H2 = 100m
H1 = 10m
a = 0.143


Wind speed is converted to wind power using the formula given in 'Meteorology and climatology of historical weekly wind and solar power resource droughts over western North America in ERA5' paper by Patrick T. Brown, Ken Caldeira 2021 

To calculate capacity factor of wind power, wind power calculated at monthly wind speed is divided by 2 MW (2000 KW) which is the capacity of a typical wind turbine.

We used ERA5 surface wind speed data to calculate wind power and capacity factor of wind power.

"""

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)

def calc_100m_wind(era5_wind_10m_ucomponent_path, era5_wind_10m_vcomponent_path, output_100m_windspeed):
    
    """
    Function to convert 10m wind speed to 100m wind speed

    """

    logger.info("Calculate wind speed from its u and v component")

    u_component = xr.open_dataset(
        era5_wind_10m_ucomponent_path,
        chunks={'valid_time':100}.sel(valid_time=slice('2000-01-01', '2022-12-31'))
    )

    v_component = xr.open_dataset(
        era5_wind_10m_vcomponent_path,
        chunks={'time':100}.sel(valid_time=slice('2000-01-01', '2022-12-31'))
    )

    wind_speed_10m = np.sqrt(u_component.u10**2 + v_component.v10**2)

    logger.info("Convert wind speed from 10m to 100m hub height")

    # Define power-law exponent (adjust if needed based on stability conditions)
    alpha = 0.14

    # Compute wind speed at 100m
    wind_speed_100m = wind_speed_10m * (100/10) ** alpha

    # Create a new dataset
    ds_wind_100m = xr.Dataset({"wind_speed_100m": wind_speed_100m})

    compress_setting = {"wind_speed_100m" : {"zlib" : True, "complevel" : 9}}

    logger.info("Store 100m wind speed data")

    # Save the NetCDF if needed
    ds_wind_100m.to_netcdf(output_100m_windspeed, encoding=compress_setting)


def calc_wind_power(wind_speed_100m_arr):

    logger.info("Convert wind speed to wind power")

    mask = np.where((wind_speed_100m_arr < 3) | (wind_speed_100m_arr > 25), 0, wind_speed_100m_arr)
    mask = np.where((mask > 12.5), 12.5, mask)
    mask = (
        634.228
        - 1248.5 * mask
        + 999.57 * mask**2
        - 426.224 * mask**3
        + 105.617 * mask**4
        - 15.4587 * mask**5
        + 1.3223 * mask**6
        - 0.0609186 * mask**7
        + 0.00116265 * mask**8
    )
    return mask

def main(era5_wind_10m_ucomponent_path, era5_wind_10m_vcomponent_path, output_100m_windspeed, output_wind_capacity_factor):
    calc_100m_wind(era5_wind_10m_ucomponent_path, era5_wind_10m_vcomponent_path, output_100m_windspeed)

    wind_speed_100_data = xr.open_dataset(output_100m_windspeed)

    wind_power = xr.apply_ufunc(calc_wind_power, wind_speed_100_data["wind_speed_100m"])

    logger.info("Convert wind power to its capacity factor")

    capacity_factor = xr.apply_ufunc(lambda x: (x / (1000)) / 2, wind_power)
    capacity_factor.name = "wind_capacity_factor"

    compression_settings = {'wind_capacity_factor': {'zlib': True, 'complevel': 9}}

    logger.info("Store wind capacity factor data")

    capacity_factor.to_netcdf(output_wind_capacity_factor, encoding = compression_settings)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        "Calculate wind speed to wind capacity factor"
    )

    parser.add_argument(
        "--era5_wind_10m_ucomponent_dir",
        default='/home/shah0012/ERA5_Land/surface_wind/10m_u_component/10m_u_component_of_wind_1981_2024.nc',
        help="Directory with netcdf file containing u component of wind speed at 10m"
    )
    
    parser.add_argument(
        "--era5_wind_10m_vcomponent_dir",
        default='/home/shah0012/ERA5_Land/surface_wind/10m_v_component/10m_v_component_of_wind_1981_2024.nc',
        help="Directory with netcdf file containing v component of wind speed at 10m"
    )

    parser.add_argument(
        "--output_100m_windspeed_dir",
        default="/home/shah0012/ERA5_Land/surface_wind/wind_speed_100m.nc",
        help="Directory with netcdf file name to store 100m wind speed data"
    )

    parser.add_argument(
        "--output_wind_capacity_factor_dir",
        default="/scratch/shah0012/ERA5_Land_processed/wind_CF_factor/global_monthly_capacity_factor_wind_2000_2022.nc",
        help="Directory withe netcdf file name to store wind capacity factor"
    )

    args = parser.parse_args()

    main(
        era5_wind_10m_ucomponent_path = args.era5_wind_10m_ucomponent_dir,
        era5_wind_10m_vcomponent_path = args.era5_wind_10m_vcomponent_dir,
        output_100m_windspeed = args.output_100m_windspeed_dir,
        output_wind_capacity_factor = args.output_wind_capacity_factor_dir
    )






