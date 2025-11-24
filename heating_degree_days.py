import xarray as xr
import numpy as np
import glob
import logging
import argparse
from multiprocessing import Pool

"""
This code calculate Heating Degree Days (HDD) using threshold as 18.3C or 293.45K. To calculate HDD, we are using the 2m height daily temperature data provided by ERA5. 
ERA5 provides temperature data is Kelvin. Here all daily temperature values were subtracted from 293.45 K and after that all negative values are converted from daily to monthly
by summing them up. 

"""

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)

def HDD_mask(arr):

    """
    Subtract daily temperature from 293.45K and mask out all values less than 0

    """

    logger.info("Calculate Heating Degree Days")

    subtracted = np.subtract(arr, 293.45)
    mask = np.where(subtracted < 0, subtracted, 0)
    mask = np.abs(mask)
    return mask

def read_write_HDD(files, output_dir):

    logger.info("Open ERA5 Land temperature data and calculate HDD")

    filename = files.split("/")[-1]
    filename_parts = filename.split("_")
    year_month = filename_parts[-3] + "_" + filename_parts[-2]

    ds = xr.open_dataset(files)
    HDD_daily_data = xr.apply_ufunc(HDD_mask, ds["t2m"])

    # Convert daily data to monthly
    HDD_monthly_data = HDD_daily_data.resample(valid_time='1M').sum('valid_time').compute()
    HDD_monthly_data.name = "HDD"
    compression_settings = {'HDD': {'zlib': True, 'complevel': 5}}

    logging.info("Store HDD file")

    HDD_monthly_data.to_netcdf(output_dir + "ERA5_Land_HDD_" + year_month + "_correctlon.nc",
                          encoding=compression_settings)
 
def main(era5_land_temperature_files, output_dir):

    all_files = glob.glob(era5_land_temperature_files)

    if not all_files:
        logger.warning(f"No NetCDf exist found in {era5_land_temperature_files}")
        return


    with Pool(processes=10) as p:
        p.map(read_write_HDD, all_files, output_dir)

    return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        "Read ERA5 daily temperature data and calculate HDD"

    )

    parser.add_argument(
        "--era5_temperature_data",
        default="/scratch/shah0012/ERA5_Land_processed/temperature/correct_lon_files/*.nc",
        help="Directory with nc extension to which contains daily temperature data"

    )

    parser.add_argument(
        "--output_dir",
        default="/scratch/shah0012/ERA5_Land_processed/CDD_HDD_files/",
        help="Directory which going to store HDD output data"


    )

    args = parser.parse_args()
    
    main(
        era5_land_temperature_files = args.era5_temperature_data,
        output_dir = args.output_dir

    )