import xarray as xr
import os
from multiprocessing import Pool
import logging
import argparse

"""
Code here will change the longitude values from 0 to 360 which ERA5 provides to -180 to 180

"""

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)


def convert_longitude(ds, output_path):
    """
    Convert longtiude from 0-360 to -180-180

    Args:
    ds: xarray dataset

    Returns:
    None    

    """

    logger.info("Correcting the longitude values")

    lon_name = "longitude"
    data = xr.open_dataset(ds)
    print(f"Processing {ds}")

    if lon_name is data.coords:
        raise ValueError(f"Longitude coordinate name is not {lon_name}")
    
    data[lon_name] = (data[lon_name] + 180) % 360 - 180
    data = data.sortby(lon_name)

    compression_setting = {var_name : {"zlib" : True, 'complevel' : 4} for var_name in data.data_vars}
    filename  = ds.split("/")[-1].split(".")[0] + "_correctlon.nc"
    final_path = output_path + "/" + filename

    logger.info("Saving corrected file")

    data.to_netcdf(final_path, encoding=compression_setting)



def list_all_files(directory):
    all_files = []

    logger.info(f"Reading netcdf files from {directory}")

    if not os.path.exists(directory):
        logger.warning(f"Path {directory} does not exist")
        return
    
    # Walk through directory, subdirectories, and files
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(".nc"):
                # Get full path of each file
                file_path = os.path.join(root, file)
                all_files.append(file_path)
    
    return all_files


def main(era5_temperature_path, output_path):
    all_files = list_all_files(era5_temperature_path)

    with Pool(processes=10) as pool:
        pool.map(convert_longitude, all_files, output_path)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        "Correct Longitude from 0 to 360 to -180 to 180 for ERA5 temperature data"
    )

    parser.add_argument(
        "--era5_temperature_dir",
        default="/scratch/shah0012/ERA5_Land_processed/temperature/",
        help="Directory cotaining temperature netcdf files from ERA5"
    )

    parser.add_argument(
        "--output_path",
        default="/scratch/shah0012/ERA5_Land_processed/temperature/correct_lon_files",
        help="Directory in which to save the files",

    )

    args = parser.parse_args()

    main(
        era5_temperature_path = args.era5_temperature_dir,
        output_path =  args.output_path
    )