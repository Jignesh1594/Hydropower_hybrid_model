import xarray as xr
import numpy as np
import glob
import argparse
from multiprocessing import Pool
import logging

"""
This code calculate cooling degree days (CDD) using threshold as 18.3C or 293.45 K. To calculate CDD, we are using the 2m height daily temperature data provided by ERA5. 
ERA5 provides temperature data is Kelvin. Here all daily temperature values were subtracted from 293.45 K and after that all positive values are converted from daily to monthly
by summing them up. 

"""

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)

def CDD_mask(arr):

    """
    Subtract daily temperature from 293.45K and mask out all values greater than 0

    """
    logger.info("Subtracting daily temperature values from 293.45K")
    subtracted = np.subtract(arr, 293.45)
    mask = np.where(subtracted > 0, subtracted, 0)
    return mask

def read_write_CDD(file, output_dir):

    logger.info("Processing ERA5 temperature netcdf files")
    
    filename = file.split("/")[-1]
    filename_parts = filename.split("_")
    year_month = filename_parts[-3] + "_" + filename_parts[-2]

    ds = xr.open_dataset(file)
    CDD_daily_data = xr.apply_ufunc(CDD_mask, ds["t2m"])

    # Convert daily data to monthly
    CDD_monthly_data = CDD_daily_data.resample(valid_time='1M').sum('valid_time').compute()
    CDD_monthly_data.name = "CDD"
    compression_settings = {'CDD': {'zlib': True, 'complevel': 5}}
    CDD_monthly_data.to_netcdf(output_dir + "ERA5_Land_CDD_" + year_month + "_correctlon.nc",
                          encoding=compression_settings)



def main(file, output_dir):

    all_files = glob.glob(file)

    if not all_files:
        logger.warning(f"No NetCDf exist found in {file}")
        return

    with Pool(processes=10) as p:
        p.map(read_write_CDD, all_files, output_dir)

    
if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        "Read ERA5 daily temperature data and calculate CDD"
    )

    parser.add_argument(
        "--era5_temperature_data",
        default="/scratch/shah0012/ERA5_Land_processed/temperature/correct_lon_files/*.nc",
        help="Directory with nc extension to which contains daily temperature data"
    )

    parser.add_argument(
        "--output_dir",
        default="/scratch/shah0012/ERA5_Land_processed/CDD_HDD_files/",
        help="Directory which going to store CDD output data"
    )

    args = parser.parse_args()

    main(
        file = args.era5_temperature_data,
        output_dir = args.output_dir
    )



