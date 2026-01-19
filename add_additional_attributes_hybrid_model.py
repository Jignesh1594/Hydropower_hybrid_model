import logging
import pandas as pd
import numpy as np
import geopandas as gpd
from tqdm import tqdm
import xarray as xr
import argparse


logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

# Global variable
base_path = "/scratch/shah0012/hybrid_hydropower_model/data/hybrid_model_data/"


def calc_var_stat(hybrid_input_file : str,
                 var_file_path: str, 
                 var_name: str, 
                 latitude_name="lat", 
                 longitude_name="lon", 
                 lag=6):

    """
    Get lagged values of variable

    """

    plant_id_list = []
    date_list = []

    logger.info(f"Read hybrid model input data from {hybrid_input_file}")

    hybrid_input_data = pd.read_csv(base_path + hybrid_input_file, parse_dates=["date"])
    max_year = str(max(hybrid_input_data["date"].dt.year))

    logger.info(f"Read variable xarray file {var_file_path}")

    var_xarray = xr.open_mfdataset(
        var_file_path + "*.nc",
        concat_dim="time",
        combine="nested",
        data_vars="minimal")
    var_xarray = var_xarray.sel(time=slice("1999", max_year))

    times = var_xarray.time.values
    
    if np.any(pd.DatetimeIndex(times).day != 1):
        logger.info("Converting variable dates to 1st of month")
        new_times = pd.DatetimeIndex([pd.Timestamp(year=pd.Timestamp(t).year, 
                                                    month=pd.Timestamp(t).month, 
                                                    day=1) for t in times])
    var_xarray = var_xarray.assign_coords(time=new_times)

    var_xarray = var_xarray.load()

    # Create a dictionary to store lagged snow cover values
    var_hydro_list = {f"snowcover_{i}n": [] for i in range(lag + 1)}

    for row in  tqdm(hybrid_input_data.itertuples(index=False), total=len(hybrid_input_data), desc=f"Calculating each hydropower station {var_name}"):
        dam_lat = int(row.pcr_dam_lat_index)
        dam_lon = int(row.pcr_dam_lon_index)
        plant_id = row.glohydrores_plant_id
        current_date = row.date
        plant_id_list.append(plant_id)
        date_list.append(current_date)

        # Calculate date lag months before current date
        previous_month = pd.Timestamp(current_date) - pd.DateOffset(months=lag)

        try:
             # Get time index for current date
            current_date_index = np.where(times == np.datetime64(current_date))[0][0]
            # Get time index for previous month
            previous_month_index = np.where(times == np.datetime64(previous_month))[0][0]

             # Get all months from previous_month to current_date (inclusive)
            # This should give us lag+1 values (including current month)
            get_all_lagged_months = var_xarray.isel({
                latitude_name: dam_lat, 
                longitude_name: dam_lon, 
                "time": slice(previous_month_index, current_date_index + 1)
            })[var_name].values

            # Check if we have the right number of values
            if len(get_all_lagged_months) == lag + 1:
                # Properly assign each month's snow cover value to the corresponding list
                for i in range(lag + 1):
                    # snowcover_0n is current month, snowcover_1n is 1 month ago, etc.
                    var_hydro_list[f"snowcover_{lag-i}n"].append(get_all_lagged_months[i])
            else:
                # Handle case where we didn't get exactly lag+1 values
                for i in range(lag + 1):
                    var_hydro_list[f"snowcover_{i}n"].append(np.nan)
                logger.warning(f"Expected {lag+1} values for plant {plant_id} at date {current_date}, but got {len(get_all_lagged_months)}")

        
        except (KeyError, IndexError) as e:
            # Handle case where date is not found or other errors
            logger.warning(f"Error processing {var_name} data for plant {plant_id} at date {current_date}: {e}")
            for i in range(lag + 1):
                var_hydro_list[f"snowcover_{i}n"].append(np.nan)
    

    # Create the result DataFrame
    result_df = pd.DataFrame({
        "glohydrores_plant_id": plant_id_list,
        "date": date_list
    })
    
    logger.info("Create final dataframe which contains glohydrores unique ID, data and variable values")
    # Add the snowcover columns
    for col_name, values in var_hydro_list.items():
        result_df[col_name] = values


    return result_df
        

            

def main(hybrid_input_file,var_file_path, var_name, latitude_name, longitude_name, lag, glohydrores_file, final_output_file):

    logger.info("Function to get lagged values for a variable")

    result_df =  calc_var_stat(hybrid_input_file, var_file_path, var_name, latitude_name, longitude_name, lag)

   # Read the data
    hybrid_model_training_data = pd.read_csv(
        base_path + hybrid_input_file,
        parse_dates=["date"]
        )

    hybrid_model_training_data = hybrid_model_training_data.merge(result_df, on = ["glohydrores_plant_id", "date"], how = "left")

    # Import glohydrores data
    logger.info("Loading and integrating GloHydroRes data...")
    GloHydroRes = pd.read_excel(glohydrores_file, sheet_name="Data")

    logger.info("Integrate plant latitude")
    # Add plant latitude to the data
    hybrid_model_training_data["plant_lat"] = hybrid_model_training_data["glohydrores_plant_id"].map(GloHydroRes.set_index("ID")["plant_lat"])

    hybrid_model_training_data.to_csv(
    base_path + final_output_file,
    index=False)    

#hybrid_input_file,var_file_path, var_name, latitude_name, longitude_name, lag, glohydrores_file, final_output_file

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        "Add extra variables to hybrid input file"
    )

    parser.add_argument(
        "--hybrid_input_file",
        default="hybrid_model_train_generation_data_1981_2022_selected_variables.csv",
        help = "File containing hybrid input data which will get modified using additional attributes"
    )

    parser.add_argument(
        "--var_file_path",
        default="/home/shah0012/PCRGLOWB_data/SnowCover/runs_till_2022/",
        help = "Path containing netcdf files of variable to add to hybrid input data"
    )

    parser.add_argument(
        "--var_name",
        default="snow_water_equivalent",
        help = "Variable name in netcdf files"
    )

    parser.add_argument(
        "--latitude_name",
        default="latitude",
        help = "latitude variable name in netcdf files"
    )

    parser.add_argument(
        "--longitude_name",
        default="longitude",
        help = "longitude variable name in netcdf files"
    )

    parser.add_argument(
        "--lag",
        default=6,
        help = "longitude variable name in netcdf files",
        type = int
    )

    parser.add_argument(
        "--glohydrores_file",
        default="/home/shah0012/GloHydroRes/Output_data/GloHydroRes_vs2.xlsx",
        help = "Path with excel file containing GloHydroRes data"
    )

    parser.add_argument(
        "--final_output_file",
        default="hybrid_model_train_generation_data_1981_2022_selected_variables_snow_cover_historical_plant_lat.csv",
        help = "Final file name to save the data"
    )

    args = parser.parse_args()

    main(
        hybrid_input_file = args.hybrid_input_file,
        var_file_path =  args.var_file_path,
        var_name  = args.var_name,
        latitude_name = args.latitude_name,
        longitude_name  = args.longitude_name,
        lag  = args.lag,
        glohydrores_file = args.glohydrores_file,
        final_output_file = args.final_output_file)
    
    


