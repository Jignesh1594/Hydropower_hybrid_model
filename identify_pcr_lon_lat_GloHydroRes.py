import xarray as xr
import pandas as pd
import geopandas as gpd
import glob
import numpy as np
from multiprocessing import Pool
import logging
import argparse


"""

The code is used to identify the most suitable longitude and latitude for PCR-GLOWB2.0, for each hydropower plant based on its reservoir location from GloHydroRes. Using the 
reservoir location, I compare the nine surrounding grid cells to determine the hydropower potential and  the cell whose potential is closest to the plant's installed capacity. 
Ideally, the hydropower potential of the selecte cell should be greater than the installed capacity but the method also aims to avoid cases where the hydropower potential is 
lower than installed capacity.

"""

logging.basicConfig(
    level = logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)

def open_file(glohydrores_file, streamflow_path, grand_path, geodar_path, hydrolakes_path):

    logger.info("Open GloHydroRes file")

    # Load the GloHydroRes dataset
    GloHydroRes = pd.read_excel(glohydrores_file, sheet_name = "Data")

    # if head is not a
    GloHydroRes["final_head_m"] = GloHydroRes.apply(
        lambda row: row["dam_height_m"] if pd.isna(row["head_m"])
        else row["head_m"],
        axis = 1
    )

    GloHydroRes = GloHydroRes[~GloHydroRes.final_head_m.isna()]
    GloHydroRes = GloHydroRes[~GloHydroRes.res_dam_source.isna()]

    # Model Discharge Data

    logger.info(f"Open streamflow data {streamflow_path}")
    files = glob.glob(streamflow_path, "*.nc")
    streamflow_data = xr.open_mfdataset(files, concat_dim = "time", combine="nested", data_vars = "minimal")
    pcr_lat = streamflow_data.latitude.values
    pcr_lon = streamflow_data.longitude.values

    streamflow_data = streamflow_data.rename({"longitude" : "lon", "latitude" : "lat"})

    logger.info(f"GranD reservoir data {grand_path}")
    grand_dam_data = gpd.read_file(grand_path  + "GRanD_dams_v1_3.shp")

    logger.info(f"GeoDAR reservoir data {geodar_path}")
    geodar_dam_data = gpd.read_file(geodar_path + "GeoDAR_v11_dams.shp")

    logger.info(f"HydroLakes data {hydrolakes_path}")
    hydrolakes_data = gpd.read_file(hydrolakes_path + "HydroLAKES_points_v10.shp")

    return GloHydroRes, pcr_lat, pcr_lon, streamflow_data,  grand_dam_data, geodar_dam_data, hydrolakes_data



def dam_lon_lat_func(power_plant_data : pd.DataFrame, 
grand_dam_data : gpd.GeoDataFrame, 
geodar_dam_data : gpd.GeoDataFrame,
hydrolakes_points_data : gpd.GeoDataFrame):

    """
    This function will return the longitude and latitude of the dam and reservoir for each hydropower plant based on the source they mapped

    Args:
    generation_data : pd.DataFrame : generation data of hydropower plants

    """
    logger.info("Get the longitude and latitude of each hydropower plant based on the source data")

    plant_id_list = []
    lon_list = []
    lat_list = []
    unique_ids = pd.unique(power_plant_data.ID).tolist()

    for i in unique_ids:
        logger.info(f"Processing the plant {i}")
        record = power_plant_data[
            power_plant_data.ID == i
        ].drop_duplicates(subset = "ID")

        try:
            if record.res_dam_source.iloc[0] == "GranD":
                selected_dam_data = grand_dam_data[
                    grand_dam_data["GRAND_ID"] == record.res_dam_source_id.iloc[0]
                ]
                lon_bound = selected_dam_data.geometry.x.iloc[0]
                lat_bound = selected_dam_data.geometry.y.iloc[0]
            
            elif record.res_dam_source.iloc[0] == "GeoDAR":
                selected_dam_data = geodar_dam_data[
                    geodar_dam_data["id_v11"] == record.res_dam_source_id.iloc[0]
                ]
                lon_bound = selected_dam_data.geometry.x.iloc[0]
                lat_bound = selected_dam_data.geometry.y.iloc[0]
            
            elif record.res_dam_source.iloc[0] == "HydroLakes":
                selected_dam_data = hydrolakes_points_data[
                    hydrolakes_points_data["Hylak_id"] == record.res_dam_source_id.iloc[0]
                ]
                lon_bound = selected_dam_data.geometry.x.iloc[0]
                lat_bound = selected_dam_data.geometry.y.iloc[0]

            else:
                lon_bound = record.man_dam_lon.iloc[0]
                lat_bound = record.man_dam_lat.iloc[0]

            lon_list.append(lon_bound)
            lat_list.append(lat_bound)
            plant_id_list.append(i)
        
        except IndexError:
            continue
    
    return plant_id_list, lon_list, lat_list


def process_single_plant(args):
    
    """

    Process a single power plant - helper function for multiprocessing.

    Args:
        args (tuple): Tuple containing power plant data, latitude array, longitude array, streamflow data, and search radius.

    Returns:
        tuple: (plant_id, best_lat_index, best_lon_index)
    """
    
    plant, pcr_lat_ar, pcr_lon_ar, streamflow_data, search_radius = args
    lat, lon = plant['dam_lat'], plant['dam_lon']
    installed_capacity, head_m = plant['capacity_mw'], plant['final_head_m']

    # Find first closest grid cell
    lat_idx = np.argmin(np.abs(pcr_lat_ar - lat))
    lon_idx = np.argmin(np.abs(pcr_lon_ar - lon))

    # Search neighbourhodd for better hydropower simulation'
    lat_range = range(max(lat_idx - search_radius, 0), min(lat_idx + search_radius + 1, len(pcr_lat_ar)))
    lon_range = range(max(lon_idx - search_radius, 0), min(lon_idx + search_radius + 1, len(pcr_lon_ar)))

    logger.info(f"Processing plant {plant['ID']} with lat {lat_range} and lon {lon_range}")

    try:
        streamflow_data_sliced = streamflow_data.sel(
            time= slice('2000', '2022'),
            lat=pcr_lat_ar[lat_range],
            lon=pcr_lon_ar[lon_range]
        )
        q_90_values = streamflow_data_sliced.discharge.quantile(0.90, dim = "time").compute()
        simulated_power = (q_90_values*head_m*997*9.81)/1000000
    
    except Exception as e:
        logger.info(f"Error processing plant {plant['ID']}: {e}")
        return plant['ID'], lat_idx, lon_idx
    
    # Initialize variables
    best_potential = np.inf  # Start with infinity
    best_lat_idx, best_lon_idx = None, None
    first_iteration = True  # Flag to handle first comparison

    for i in (0,1,2):
        for j in (0,1,2):
            try:
                potential_comparison =  (simulated_power[i,j] - installed_capacity)/installed_capacity
                if first_iteration:
                    best_potential = potential_comparison
                    best_lat_idx = i - 1 
                    best_lon_idx = j - 1
                    first_iteration = False
                else:
                    if potential_comparison > 0 :
                      if best_potential <= 0 or potential_comparison < best_potential:
                        best_potential = potential_comparison
                        best_lat_idx = i - 1
                        best_lon_idx = j - 1
                    else:
                          if best_potential <= 0 and potential_comparison > best_potential:
                            best_potential = potential_comparison
                            best_lat_idx = i - 1
                            best_lon_idx = j - 1
            except Exception as e:
                print(f"Error processing lat {i}, lon {j}: {e}")
    best_lat_idx = lat_idx + best_lat_idx if best_lat_idx is not None else lat_idx
    best_lon_idx = lon_idx + best_lon_idx if best_lon_idx is not None else lon_idx

    return plant['ID'], best_lat_idx, best_lon_idx


def main(glohydrores_file, streamflow_path, grand_path,  geodar_path, hydrolakes_path, search_radius, n_processes, output_path) -> tuple:
    
    logger.info("Open all the files")
    
    """
    Find the lon and lat index of the PCRGLOWB which best matches with installed capacity of the hydropower plant using multiprocessing.
    Args:
        power_plant_data (pd.DataFrame): Hydropower plant data.
        pcr_lat_ar (np.array): Latitude array of PCRGLOBWB model.
        pcr_lon_ar (np.array): Longitude array of PCRGLOBWB model.
        streamflow_data (xr.Dataset): Xarray dataset of streamflow data.
        search_radius (int): Radius (in grid cells) to search for better streamflow.
        n_processes (int): Number of processes to use for multiprocessing.

    Returns:
        tuple: Lists of ID, latitude indices, and longitude indices.

    """

    GloHydroRes, pcr_lat, pcr_lon, streamflow_data, grand_dam_data, geodar_dam_data, hydrolakes_points_data = open_file(glohydrores_file, streamflow_path, grand_path, geodar_path, hydrolakes_path)
    
    # Rechunk the dataset along time dimension
    streamflow_data = streamflow_data.chunk({'time': -1})  # Single chunk for time dimension
    GloHydroRes = GloHydroRes.drop_duplicates(subset = ["ID"])

    logger.info("Get dam longitude and latitude of each hydropower plant")

    glohydrores_id_list, lon_list, lat_list = dam_lon_lat_func(power_plant_data= GloHydroRes, grand_dam_data = grand_dam_data, geodar_dam_data = geodar_dam_data, hydrolakes_points_data = hydrolakes_points_data)

    glohydrores_id_dam_lon_dict = {glohydrores_id_list[i]: lon_list[i] for i in range(len(glohydrores_id_list))}
    glohydrores_id_dam_lat_dict = {glohydrores_id_list[i]: lat_list[i] for i in range(len(glohydrores_id_list))}
    
    GloHydroRes["dam_lon"] = GloHydroRes.ID.map(glohydrores_id_dam_lon_dict)
    GloHydroRes["dam_lat"] = GloHydroRes.ID.map(glohydrores_id_dam_lat_dict)

    logger.info("Prepare argument for multiprocessing tool")
    arg_list = [(plant._asdict(), pcr_lat, pcr_lon, streamflow_data, search_radius) for plant in GloHydroRes.itertuples(index=False)]

    logger.info("Perform Multiprocessing")

    with Pool(processes=n_processes) as pool:
        results = pool.map(process_single_plant, arg_list)

    # Unzip results
    glohydrores_plant_id_list, pcr_lat_index_list, pcr_lon_index_list = list(zip(*results))

    glohydrores_plant_id_list, pcr_lat_index_list, pcr_lon_index_list = list(glohydrores_plant_id_list), list(pcr_lat_index_list), list(pcr_lon_index_list)

    logger.info("Storing the results")

    pd.DataFrame({"glohydrores_plant_id": glohydrores_plant_id_list, "pcr_lat_index": pcr_lat_index_list, "pcr_lon_index": pcr_lon_index_list}).to_csv(output_path, index = False)

    # Unzip results
    return results


if __name__  == "__main__":
    parser = argparse.ArgumentParse(
        "Get best possible hydrological model longitude and latitude combination for which simulated hydropower potential is close to hydropower installed capacity"
    )

    parser.add_argument(
        "--glohydrores_dir",
        default = "/home/shah0012/GloHydroRes/Output_data/GloHydroRes_vs2.xlsx",
        help = "Directory with file name (xlsx format) containing GloHydroRes file"
    )

    parser.add_argument(
        "--streamflow_dir",
        default = "/home/shah0012/PCRGLOWB_data/discharge/gswp3_w5e5/runs_till_2022/",
        help = "Directory containing gridded streamflow data"
    )

    parser.add_argument(
        "--grand_dir",
        default = "/home/shah0012/GloHydroRes/Input_data/GRanD_Version_1_3/",
        help = "Directory containing GranD dam file"
    )

    parser.add_argument(
        "--geodar_dir",
        default = "/home/shah0012/GloHydroRes/Input_data/GeoDAR_v10_v11/",
        help = "Directory containing GeoDAR dam file"
    )

    parser.add_argument(
        "--hydrolakes_dir",
        default = "/home/shah0012/GloHydroRes/Input_data/HydroLakes/HydroLAKES_points_v10_shp/",
        help = "Directory containing HydroLAKES dam file"
    )

    parser.add_argument(
        "--search_radius",
        default = 1,
        help = "Neighbouring cells to look",
        type = int
    )

    parser.add_argument(
        "--n_processes",
        default = 10,
        help = "Number of parallel processes to run",
        type = int
    )

    parser.add_argument(
        "--output_dir",
        default = "/home/shah0012/Hydropower_hybrid_model/data/Hybrid_model_data/power_plant_pcrglob_lat_lon.csv",
        help = "Directory that going to store final csv output"
    )



    args = parser.parser_args()

    main(
        glohydrores_file = args.glohydrores_dir,
        streamflow_path = args.streamflow_dir,
        grand_path = args.grand_dir,
        geodar_path = args.geodar_dir,
        hydrolakes_path = args.hydrolakes_dir,
        search_radius = args.search_radius,
        n_processes=  args.n_processes,
        output_path = args.output_dir
    )




















