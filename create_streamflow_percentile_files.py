"""
Drought is indetified here using variable threshold method. 
If streamflow falls below 20% then it is considered as drought

"""

import xarray as xr
import pandas as pd
import numpy as np
from scipy.stats import percentileofscore
import os
import shutil
import glob
import logging
import argparse

from multiprocessing import Pool
from functools import partial

# Set  up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def process_plant(plant_id, glohydrores_df, streamflow_data, output_dir, time_series, month_mask):

    """
    Process a single hydropower plant to identify drought conditions

    """

    try:
        # Get lon/lat indices of plant
        plant_info = glohydrores_df[glohydrores_df.loc[:, "ID"] == plant_id]
        logger.info(f"Processing plant {plant_id}")
        if plant_info.empty:
            logger.warning(f"Plant {plant_id} not found in GloHydroRes data")
            return
        
        lon, lat = plant_info[["pcr_lon_index", "pcr_lat_index"]].iloc[0]
        lon, lat  = int(lon), int(lat)

        logger.info("Extact Streamflow data")
        plant_streamflow_data = streamflow_data.isel(lat=lat, lon=lon).discharge.values

        month_percentile_array = np.zeros((time_series.shape[0]))

        logger.info("Calculate percentile for each month streamflow by comparing values of all same month values")

        for month in range(1, 13):
            month_selector = month_mask == month
            month_data = plant_streamflow_data[month_selector]

            if len(month_data) > 0:
                month_percentile_array[month_selector] = [
                    percentileofscore(month_data, value, kind="rank")
                    for value in month_data
                ]
        
        logger.info(f"Save result for plant {plant_id}")
        result_df = pd.DataFrame(
            {
                "streamflow_data" : plant_streamflow_data,
                "percentile" : month_percentile_array,
                "date" : time_series
            }
        )
        result_df.to_csv(f"{output_dir}/{plant_id}.csv", index = False)

        return plant_id
    except Exception as e:
        logger.error(f"Error processing plant {plant_id}")


def main(streamflow_path, pcr_lon_lat_file, glohydrores_file, output_dir, start_year, end_year):
  
    logger.info("Starting drought identification process")

    # Create or clear output directory
    if os.path.exists(output_dir):
        logger.info(f"Removing existing output directory {output_dir}")
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)
    logger.info(f"Created output directory {output_dir}")

    # Load streamflow data
    logger.info("Loading Streamflow data")
    files = glob.glob(streamflow_path + "*.nc")
    streamflow = xr.open_mfdataset(files, concat_dim="time", combine="nested", data_vars="minimal")
    streamflow = streamflow.rename({"latitude": "lat", "longitude": "lon"})
    start_year_str = str(start_year)
    end_year_str = str(end_year)
    streamflow = streamflow.sel(time=slice(start_year_str, end_year_str))

    # Load plant location data
    logger.info("Loading file with PCR-GLOBWB lat/lon indices for each power plant")
    pcr_lon_lat_df = pd.read_csv(pcr_lon_lat_file)

    # Load GloHydroRes data
    logger.info("Loading GloHydroRes data")
    glohydrores_df = pd.read_excel(glohydrores_file, sheet_name="Data")
    glohydrores_df["pcr_lat_index"] = glohydrores_df.ID.map(dict(zip(pcr_lon_lat_df["glohydrores_plant_id"], pcr_lon_lat_df["pcr_lat_index"])))
    glohydrores_df["pcr_lon_index"] = glohydrores_df.ID.map(dict(zip(pcr_lon_lat_df["glohydrores_plant_id"], pcr_lon_lat_df["pcr_lon_index"])))

    # Filter out plants without location data
    glohydrores_df = glohydrores_df[~glohydrores_df.pcr_lat_index.isna()]
    unique_ids = glohydrores_df.ID.unique().tolist()
    logger.info(f"Identified {len(unique_ids)} unique plant IDs")

    # Create time series and month mask
    time_series = pd.date_range(f"{start_year}-01-01", f"{end_year}-12-31", freq="M")
    month_mask = time_series.month.values

    # Create partial function for with fixed arguments
    process_plant_partial = partial(
        process_plant,
        glohydrores_df=glohydrores_df,
        streamflow_data=streamflow,
        output_dir=output_dir,
        time_series=time_series,
        month_mask=month_mask)
    
    # Process plants in parallel
    logger.info("Starting parallel processing of plants")
    with Pool(processes=10) as p:
        results = p.map(process_plant_partial, unique_ids)



if __name__ == "__main__":

    parser = argparse.ArgumentParse(
        "Convert the streamflow data to percentiles for all hydropower plants and save the results in separate CSV files for each plant."
    )
    
    parser.add_argument(
        "--streamflow_dir",
        default="/home/shah0012/PCRGLOWB_data/discharge/gswp3_w5e5/runs_till_2022/",
        help="Directory containing streamflow data"
    )

    parser.add_argument(
        "--lon_lat_dir",
        default="/scratch/shah0012/hybrid_hydropower_model/data/hybrid_model_data/power_plant_pcrglob_lat_lon.csv",
        help="A directory containing CSV files that list each hydropower plant along with the longitude and latitude of its reservoir."
    )

    parser.add_argument(
        "--glohydrores_dir",
        default="/home/shah0012/GloHydroRes/Output_data/GloHydroRes_vs2.xlsx",
        help="Directory containing GlHydroRes file."
    )

    parser.add_argument(
        "--output_dir",
        default = "/scratch/shah0012/hybrid_hydropower_model/data/percentile_files",
        help="Directory where final csv files are going to stored"
    )

    parser.add_argument(
        "--start_year",
        default=1981,
        help="The first year of the streamflow data.",
        type = int
    )

    parser.add_argument(
        "--end_year",
        default=2022,
        help="The end year of the streamflow data.",
        type = int
    )


    args = parser.parse_args()

    main(
        streamflow_path = args.streamflow_dir,
        pcr_lon_lat_file  = args.lon_lat_dir,
        glohydrores_file = args.glohydrores_dir,
        output_dir = args.output_dir,
        start_year = args.start_year,
        end_year = args.end_year

    )



