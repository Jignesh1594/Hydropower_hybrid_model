import xarray as xr
import numpy as np
import os
import glob
import pandas as pd
import argparse
import logging


"""
For Solar power generation 
Solar power is calculated using the formula given in 'Meteorology and climatology of historical weekly wind and solar power resource droughts over western North America in ERA5' paper by Patrick T. Brown, Ken Caldeira 2021



Capacity factor of solar power = (G/Gstc)*(0.9)*(0.95)
G = total incident downwelling
Gstc = 1000 W/m^2, standard test condition
0.9 is perfomance
For solar power, we used ISIMIP surface shortwave downwelling radiation used. ISIMIP uses the ERA5 reanalysis data. ERA5 gives direct and diffuse solar radiation (https://cds.climate.copernicus.eu/cdsapp#!/dataset/derived-near-surface-meteorological-variables?tab=overview).
https://confluence.ecmwf.int/pages/viewpage.action?pageId=197702790
convert J/m2 to W/m2 by dividing by 86400

"""

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)

def calc_solar_power(era5_shortwave_downwelling):
    
    shortwave_downwelling_data = xr.open_dataset(era5_shortwave_downwelling)

    shortwave_downwelling_data = shortwave_downwelling_data.sel(valid_time = slice("2000-01-01", "2022-12-31"))


def main(era5_shortwave_downwelling, output_solar_capacity_factor):
    
    shortwave_downwelling_data = calc_solar_power(era5_shortwave_downwelling)

    solar_power = xr.apply_ufunc(
    lambda x: (x / (1000 * 86400)) * (0.9) * (0.95), shortwave_downwelling_data["ssrd"]
    )

    solar_power.name = "solar_capacity_factor"

    compression_settings = {'solar_capacity_factor': {'zlib': True, 'complevel': 4}}

    solar_power.to_netcdf(output_solar_capacity_factor, encoding = compression_settings)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        "Calculate solar capacity factor"
    )

    parser.add_argument(
        "--era5_shortwave_downwelling_dir",
        default="/scratch/shah0012/ERA5_Land_processed/shortwave_downwelling/shortwave_downwelling_data_1981_2022.nc",
        help="Directory with file name (netcdf format) which contains shortwave downwelling data"
    )

    parser.add_argument(
        "--output_solar_capacity_factor",
        default="/scratch/shah0012/ERA5_Land_processed/solar_CF_factor/global_monthly_capacity_factor_solar_2000_2022.nc",
        help="Directory with file name (netcdf format) to store the final solar capacity factor"
    )
    
    args = parser.parse_args()

    main(
        era5_shortwave_downwelling = args.era5_shortwave_downwelling_dir,
        output_solar_capacity_factor = args.output_solar_capacity_factor
    )



