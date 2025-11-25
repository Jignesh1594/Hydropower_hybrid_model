import logging
import pandas as pd
import xarray as xr
import numpy as np
from pandas.tseries.offsets import DateOffset
import geopandas as gpd
from tqdm import tqdm

"""
This code creates a single CSV file that contains all input variables and the observed hydropower generation data at the plant level. 
The file includes training, validation, and testing subsets and is used to train and evaluate the Hydropower Hybrid Model.

"""


logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

# Global variables
base_path = "/scratch/shah0012/hybrid_hydropower_model/data/"

def process_global_hydropower_data(output_filepath=None):

    """
    Processes and integrated global hydropower data from multiple sources

    This function:
    1. Loads generation data from multiple countries/regions
    2. Standardizes the date formats to first day of month
    3. Convert generation values to common units
    4. Filters plants based on quality of criteria
    6. Adds geospatial information

    Parameters:
    ----------
    output_filepath : str, optional
        Path to save the processed data

    Returns:
    -------
    pandas.DataFrame
        Processed global hydropower generation data

    """

    logger.info("Loading generation data from multiple sources...")

    # Load generation data from different sources
    data_sources = {
        "Europe": pd.read_excel(f"{base_path}Transparency_ENTSOE/ENTSOE_GloHydroRes_final_timeseries_generation_data_2015_2022.xlsx"),
        "USA": pd.read_csv(f"{base_path}SWD_Turner_USA_monthly_hydropower_generation/RectifHyd_v1.3_with_GloHydroRes_plant_ID_run_till_2022.csv"),
        "Russia": pd.read_excel(f"{base_path}Russia_generation_data/Russia_actual_generation_data_2011_2021.xlsx"),
        "India": pd.read_excel(f"{base_path}India_generation_data/India_final_generation_data.xlsx"),
        "France": pd.read_excel(f"{base_path}France_generation_data/France_plant_generation_data_2012_2022.xlsx")
    }

    # Concatenate all the data to create a global generation data
    global_generation_data = pd.concat(data_sources.values(), axis=0)

    # Add Brazil data
    Brazil = pd.read_excel(f"{base_path}Brazil_ONS_generation_data/ons_monthly_generation_data_GloHydroRes_2000_2021.xlsx")

    # Clean Brazil data
    Brazil_sorted = Brazil.sort_values(
    by = ['glohydrores_plant_id', 'date', 'generation_value'])
    Brazil_sorted_removed_duplicated = Brazil_sorted.drop_duplicates(subset=["glohydrores_plant_id", "date"], keep="first")

    global_generation_data = pd.concat([global_generation_data, Brazil_sorted_removed_duplicated], axis=0)
    

    logger.info("Standardizing dates and units...")

    global_generation_data["date"] = pd.to_datetime(global_generation_data["date"], format="%Y-%m-%d")
    global_generation_data["date"] = global_generation_data["date"].dt.to_period("M").dt.to_timestamp()
    
    global_generation_data.drop_duplicates(subset=['glohydrores_plant_id',  'date'], inplace=True)

    global_generation_data["Year"] = global_generation_data["date"].dt.year
    global_generation_data["Month"] = global_generation_data["date"].dt.month

    logger.info("Convert generation values to a common unit (MW)")

    # Convert generation values to a common unit (MW)
    global_generation_data["observed_generation_mw"] = global_generation_data.apply(
        lambda row: 
            row["generation_value"]/(row["date"].days_in_month*24) if row["generation_unit"] == "MWH" else
            row["generation_value"]*1000/(row["date"].days_in_month*24) if row["generation_unit"] == "GWH" else
            row["generation_value"]*1000/(row["date"].days_in_month*24) if row["generation_unit"] == "MU" else
            row["generation_value"],
        axis=1
    )

    logger.info("Filtering out plants with quality issues...")

    # Remove plants with negative generation (likely pumped storage)
    logger.info("Remove the hydropower plants with negative monthly hydropower generation values")
    negative_gen_plant_ids = global_generation_data[global_generation_data.observed_generation_mw < 0].glohydrores_plant_id.unique()
    global_generation_data = global_generation_data[~global_generation_data.glohydrores_plant_id.isin(negative_gen_plant_ids)]

    logger.info("Remove hydropower plants that have more than 20% of their values equal to zero.")

    # Calculate percentage of zero generation values per plant
    zero_gen_counts = global_generation_data.groupby("glohydrores_plant_id")["observed_generation_mw"].apply(lambda x: (x == 0).sum())
    zero_gen_percentages = global_generation_data.groupby("glohydrores_plant_id")["observed_generation_mw"].apply(lambda x: (x == 0).sum()/len(x))

    global_generation_data["count_zero_generation"] = global_generation_data.glohydrores_plant_id.map(zero_gen_counts.to_dict())
    global_generation_data["percentage_zero_generation"] = global_generation_data.glohydrores_plant_id.map(zero_gen_percentages.to_dict())
    
    # Remove plants with more than 20% zero generation records
    global_generation_data = global_generation_data[global_generation_data.percentage_zero_generation < 0.2]
    global_generation_data.drop(columns=["count_zero_generation", "percentage_zero_generation"], inplace=True)

    # Removing some countries data where data quality is not good
    logger.info("Remove data from countries where data quality was found to be poor based on a manual review.")
    global_generation_data = global_generation_data[~global_generation_data.country.isin(["Albania", "North Macedonia", "Romania", "Serbia"])]
    global_generation_data  = global_generation_data[~((global_generation_data.country == "United Kingdom") & (global_generation_data.date.dt.year > 2018))]

    # Remove data after 2022
    global_generation_data = global_generation_data[global_generation_data.date.dt.year < 2023]

    logger.info("Loading and integrating GloHydroRes data...")
    GloHydroRes = pd.read_excel("/home/shah0012/GloHydroRes/Output_data/GloHydroRes_vs2.xlsx", sheet_name="Data")

    # Create mapping dictionaries from GloHydroRes
    attribute_mappings = {
        "installed_capacity_mw": dict(zip(GloHydroRes.ID, GloHydroRes.capacity_mw)),
        "plant_type": dict(zip(GloHydroRes.ID, GloHydroRes.plant_type)),
        "res_vol_km3": dict(zip(GloHydroRes.ID, GloHydroRes.res_vol_km3)),
        "res_area_km2": dict(zip(GloHydroRes.ID, GloHydroRes.res_area_km2)),
        "res_depth_m": dict(zip(GloHydroRes.ID, GloHydroRes.res_avg_depth_m)),
        "dam_height_m": dict(zip(GloHydroRes.ID, GloHydroRes.dam_height_m)),
        "head_m": dict(zip(GloHydroRes.ID, GloHydroRes.head_m)),
        "res_dam_source": dict(zip(GloHydroRes.ID, GloHydroRes.res_dam_source)),
        "res_dam_source_id": dict(zip(GloHydroRes.ID, GloHydroRes.res_dam_source_id)),
        "man_dam_lat": dict(zip(GloHydroRes.ID, GloHydroRes.man_dam_lat)),
        "man_dam_lon": dict(zip(GloHydroRes.ID, GloHydroRes.man_dam_lon)),
        "plant_lat": dict(zip(GloHydroRes.ID, GloHydroRes.plant_lat)),
        "plant_lon": dict(zip(GloHydroRes.ID, GloHydroRes.plant_lon)),
        "country": dict(zip(GloHydroRes.ID, GloHydroRes.country))
    }

    # Fill in missing capacity values from GloHydroRes
    global_generation_data.installed_capacity_mw.fillna(
        global_generation_data.glohydrores_plant_id.map(attribute_mappings["installed_capacity_mw"]), 
        inplace=True
    )

    logger.info("Saving the excel with updated installed capacity for europe plants based on ENTSOE data")

    europe_data = data_sources["Europe"]
    europe_data.drop_duplicates(subset = ["glohydrores_plant_id"], keep="first", inplace = True)
    europe_data[["glohydrores_plant_id", "installed_capacity_mw"]].to_excel(f"{base_path}GloHydroRes_updates/ENTSOE_installed_capacity_glohydrores_plant_id.xlsx", index=False)

    # Sort data for better organization
    global_generation_data.sort_values(["country", "plant_name", "date"], inplace=True)


    for attr, mapping in attribute_mappings.items():
        if attr != "installed_capacity_mw":
            global_generation_data[attr] = global_generation_data.glohydrores_plant_id.map(mapping)

        
    logger.info("Filtering and cleaning plant types. Removing hydropower plants which either function as Pumped Storage and Canal as water source")

    # Remove pumped storage or canal plants
    global_generation_data = global_generation_data[~global_generation_data["plant_type"].isin(["PS", "Canal"])]

    logger.info("Set reservoir volume to 0 for ROR plants with missing values")
    global_generation_data.loc[(pd.isna(global_generation_data.res_vol_km3)) & 
                               (global_generation_data.plant_type == "ROR"), "res_vol_km3"] = 0

    logger.info("Using dam height as head for those hydropower plants for which head value is missing")

    # Use dam height as head when head is missing
    global_generation_data["final_head_m"] = global_generation_data.apply(
        lambda row: row["dam_height_m"] if pd.isna(row["head_m"]) else row["head_m"],
        axis=1
    )

    # Remove plants with missing head or generation data
    global_generation_data = global_generation_data[~global_generation_data.final_head_m.isna()]
    global_generation_data = global_generation_data.dropna(subset=["observed_generation_mw"])
    global_generation_data = global_generation_data[~global_generation_data.res_dam_source.isna()]


    logger.info("Loading geospatial data and PCRGLOBWB data...")

     # Add PCR-GLOBWB grid indices
    glohydrores_pcr_lat_lon = pd.read_csv(f"{base_path}hybrid_model_data/power_plant_pcrglob_lat_lon.csv")
    plant_ID_pcr_lon_dict = glohydrores_pcr_lat_lon.set_index("glohydrores_plant_id")["pcr_lon_index"].to_dict()
    plant_ID_pcr_lat_dict = glohydrores_pcr_lat_lon.set_index("glohydrores_plant_id")["pcr_lat_index"].to_dict()

    global_generation_data["pcr_dam_lat_index"] = global_generation_data.glohydrores_plant_id.map(plant_ID_pcr_lat_dict)
    global_generation_data["pcr_dam_lon_index"] = global_generation_data.glohydrores_plant_id.map(plant_ID_pcr_lon_dict)
    global_generation_data = global_generation_data[~global_generation_data.pcr_dam_lat_index.isna()]

    logger.info("Update the installed capacity if observed generation is higher than installed capacity")

    summary_df = global_generation_data.groupby("glohydrores_plant_id").agg({
        "observed_generation_mw": "max",
        "installed_capacity_mw" : "max"})

    summary_df.reset_index(inplace = True)

    summary_df["observed_generation_mw"] = summary_df["observed_generation_mw"].round()

    # Update installed capacity when observed generation is greater
    summary_df["installed_capacity_mw"] = summary_df.apply(
    lambda row: max(row["installed_capacity_mw"], row["observed_generation_mw"]), 
    axis=1)

    summary_df.to_excel(f"{base_path}GloHydroRes_updates/updated_installed_capacity_based_on_observed_generation.xlsx", index=False)

    updated_installed_cap_dict = summary_df[["glohydrores_plant_id", "installed_capacity_mw"]].set_index("glohydrores_plant_id")["installed_capacity_mw"].to_dict()
    global_generation_data["installed_capacity_mw"] = global_generation_data["glohydrores_plant_id"].map(
    updated_installed_cap_dict)

    logger.info(f"Processing complete. Dataset contains {len(global_generation_data)} records for {global_generation_data['glohydrores_plant_id'].nunique()} plants.")

    # reset index
    global_generation_data.reset_index(drop=True, inplace=True)

    # Save the processed data if a filepath is provided
    if output_filepath:
        global_generation_data.to_csv(output_filepath, index=False)
        logger.info(f"Processed data saved to {output_filepath}")
    
    return global_generation_data


def physical_based_hydropower_func(discharge_path, df, start_year, lag_month):
    streamflow_data = xr.open_mfdataset(
        discharge_path + "*.nc",
        concat_dim = "time",
        combine = "nested",
        data_vars = "minimal"
    )

    times = streamflow_data.time.values

    if np.any(pd.DatetimeIndex(times).day != 1):
        logger.warning("Time dimension does not start on the first of the month")
        new_times = pd.DatetimeIndex([pd.Timestamp(year=pd.Timestamp(t).year, 
                                                  month=pd.Timestamp(t).month, 
                                                  day=1) for t in times])
        streamflow_data = streamflow_data.assing_coords(time = new_times)

    max_year = str(df.date.max().year)
    streamflow_data = streamflow_data.sel(time=slice(start_year, max_year))
    streamflow_data = streamflow_data.load()
    pcr_lat_indices = df["pcr_dam_lat_index"].to_numpy(dtype= np.int32)
    pcr_lon_indices = df["pcr_dam_lon_index"].to_numpy(dtype= np.int32)
    plant_ids = df["glohydrores_plant_id"].to_numpy()
    final_heads = df["final_head_m"].to_numpy()
    dates = df["date"].to_numpy()

    simulated_hydro_list = {f"physical_hydropower_{i}n": [] for i in range(lag_month + 1)}
    max_potential_dict = {}
    plant_id_list, date_list = [], []

    for idx in tqdm(range(len(df)), total=len(df), desc="Simulating physical hydropower generation"):
        pcr_lat_index, pcr_lon_index = pcr_lat_indices[idx], pcr_lon_indices[idx]
        plant_streamflow_data = streamflow_data.isel(latitude = pcr_lat_index, longitude = pcr_lon_index)
        plant_id = plant_ids[idx]
        date = dates[idx]
        date = pd.Timestamp(date)
        final_head_m = final_heads[idx]
        plant_streamflow_values = plant_streamflow_data.discharge.values
        plant_id_list.append(plant_id)
        date_list.append(date)

        if plant_id not in max_potential_dict:
            max_potential = np.max(plant_streamflow_values * final_head_m * 9.81 / 1000)
            max_potential_dict[plant_id] =max_potential
        
        # Calculate generation for different lag months
        previous_timestep_date = date - DateOffset(months=lag_month)
        lagged_streamflow_data = plant_streamflow_data.sel(time=slice(previous_timestep_date, date))
        lagged_streamflow_values = lagged_streamflow_data.discharge.values

        for i in range(lag_month + 1):
            generation_value = lagged_streamflow_values[i] * final_head_m * 9.81 / 1000
            simulated_hydro_list[f"physical_hydropower_{lag_month - i}n"].append(generation_value)

    result_df = pd.DataFrame({
        "glohydrores_plant_id": plant_id_list,
        "date": date_list
    })

    for col_name, values in simulated_hydro_list.items():
        result_df[col_name] = values
    result_df["max_potential"] = result_df.glohydrores_plant_id.map(max_potential_dict)

    return result_df


def convert_irena_year_to_monthly(irena_df, var_name):
    """
    Converts yearly IRENA data to monthly data

    Args:
    irena_df: DataFrame containing yearly IRENA data
    var_name: Name of the variable to be converted
    
    Returns:
    DataFram: Monthly IRENA data

    """

    logger.info(f"Converting {var_name} to monthly data")

    month_data = []
    for _, row in irena_df.iterrows():
        Year = row['Year']
        Country = row['Country']
        var = row[var_name]

        for month in range(1, 13):
            month_data.append(
                {
                    'Year': Year,
                    'Month': month,
                    'Country': Country,
                    'Date' : pd.Timestamp(year = Year, month = month, day = 1),
                    var_name: var
                }
            )
    IRENA_monthly = pd.DataFrame(month_data)
    return IRENA_monthly


def process_other_renewable_dict():

    logger.info("Loading installed capacity data from IRENA...")

    
    IRENA_wind_offshore_path = f"{base_path}IRENA_data/IRENA_WindOffShore_installed_capacity.xlsx"
    IRENA_wind_onshore_path = f"{base_path}IRENA_data/IRENA_WindOnShore_installed_capacity.xlsx"
    IRENA_solar_path = f"{base_path}IRENA_data/IRENA_SolarPV_insatlled_capacity.xlsx"
    
    logger.info("Loading IRENA wind offshore data")
    
    IRENA_wind_offshore = pd.read_excel(IRENA_wind_offshore_path, header=2)
    IRENA_wind_offshore.rename(columns={"Unnamed: 0": "Country"}, inplace=True)

    IRENA_wind_offshore.dropna(axis=0, how="any", inplace=True)
    IRENA_wind_offshore = IRENA_wind_offshore.melt(id_vars="Country", var_name="Year", value_name="WindOffShore")

    IRENA_wind_offshore.replace("-", 0, inplace=True)
    IRENA_wind_offshore["Year"] = IRENA_wind_offshore["Year"].astype(int)

    logger.info("Loading IRENA wind onshore data")
    IRENA_wind_onshore = pd.read_excel(IRENA_wind_onshore_path, header=2)
    IRENA_wind_onshore.rename(columns={"Unnamed: 0": "Country"}, inplace=True)

    IRENA_wind_onshore.dropna(axis=0, how="any", inplace=True)
    IRENA_wind_onshore = IRENA_wind_onshore.melt(id_vars="Country", var_name="Year", value_name="WindOnShore")

    IRENA_wind_onshore.replace("-", 0, inplace=True)
    IRENA_wind_onshore["Year"] = IRENA_wind_onshore["Year"].astype(int)

    IRENA_wind = IRENA_wind_offshore.merge(IRENA_wind_onshore, on=["Country", "Year"], how="outer")
    IRENA_wind["TotalWind"] = IRENA_wind["WindOffShore"].fillna(0) + IRENA_wind["WindOnShore"].fillna(0)
    IRENA_wind["Year"] = IRENA_wind["Year"].astype(int)

    logger.info("Loading IRENA solar data")
    IRENA_solar = pd.read_excel(IRENA_solar_path, header=2)
    IRENA_solar.rename(columns={"Unnamed: 0": "Country"}, inplace=True)
    IRENA_solar.dropna(axis=0, how="any", inplace=True)
    IRENA_solar = IRENA_solar.melt(id_vars="Country", var_name="Year", value_name="SolarPV")
    IRENA_solar.replace("-", 0, inplace=True)
    IRENA_solar["Year"] = IRENA_solar["Year"].astype(int)

    logger.info("Converting IRENA yearly data to monthly data")
    IRENA_wind_monthly = convert_irena_year_to_monthly(IRENA_wind, "TotalWind")
    IRENA_solar_monthly = convert_irena_year_to_monthly(IRENA_solar, "SolarPV")
    IRENA_wind_monthly = IRENA_wind_monthly[IRENA_wind_monthly.Year < 2023]
    IRENA_solar_monthly = IRENA_solar_monthly[IRENA_solar_monthly.Year < 2023]

    logger.info("Loading country shape data")
    country_shape_path =f"{base_path}country_shape_file/country_shape.shp"
    country_shape = gpd.read_file(country_shape_path)

    logger.info("Loading grid cell area data")
    grid_cell_area_path = f"{base_path}country_shape_file/Global_grid_cell_area_point1_degree_resampled.nc"
    grid_cell_area = xr.open_dataset(grid_cell_area_path)
    grid_cell_area = grid_cell_area.reindex(lat=grid_cell_area.lat[::-1]) # This is done because latitude in grid cell area starting from -90 rather than 90

    # Calculate area of each country
    grid_cell_area_values = grid_cell_area["area"].values
    country_code = grid_cell_area["country_code"].values

    country_area_list = []
    country_id_list = []

    for row in country_shape.itertuples(index=False):
        country_code_value = row.ID
        country_area = grid_cell_area_values[country_code == country_code_value].sum()
        country_area_list.append(country_area)
        country_id_list.append(country_code_value)

    country_shape["country_area"] = country_area_list

    logger.info("Matching IRENA and GloHydroRes country names")

    from fuzzywuzzy import fuzz
    
    # Filter out missing glohydrore values
    country_shape = country_shape[~country_shape.glohydrore.isna()]
    glohydrores_country_list = country_shape["glohydrore"].unique().tolist()
    IRENA_country_list = IRENA_wind["Country"].unique().tolist()

    tuple_list = [
        max([(fuzz.token_set_ratio(i, j), j) for j in IRENA_country_list])
        for i in glohydrores_country_list
    ]

    similarity_score, fuzzy_match = map(list, zip(*tuple_list))

    IRENA_GloHydroRes_dict = {
        glohydrores_country_list[i]: fuzzy_match[i]
        for i in range(len(glohydrores_country_list))
    }

    # Manual overrides for country name matching
    IRENA_GloHydroRes_dict["China"] = "China"
    IRENA_GloHydroRes_dict["Congo"] = "Congo (the)"
    IRENA_GloHydroRes_dict["Czech Republic"] = "Czechia"
    IRENA_GloHydroRes_dict["Equatorial Guinea"] = "Equatorial Guinea"   
    IRENA_GloHydroRes_dict["Georgia"] = "Georgia"
    IRENA_GloHydroRes_dict["Guinea"] = "Guinea"
    IRENA_GloHydroRes_dict["Ireland"] = "Ireland"
    IRENA_GloHydroRes_dict["Laos"] = "Lao People's Democratic Republic (the)"
    IRENA_GloHydroRes_dict["North Korea"] = "Democratic People's Republic of Korea (the)"
    IRENA_GloHydroRes_dict["Russia"] = "Russian Federation (the)"
    IRENA_GloHydroRes_dict["South Korea"] = "Republic of Korea (the)"
    IRENA_GloHydroRes_dict["Taiwan"] = "Chinese Taipei"
    IRENA_GloHydroRes_dict["Vietnam"] = "Viet Nam"

    IRENA_GloHydroRes_dict_reversed = {v: k for k, v in IRENA_GloHydroRes_dict.items()}
    IRENA_wind_monthly["glohydrores_country"] = IRENA_wind_monthly["Country"].map(IRENA_GloHydroRes_dict_reversed)   
    IRENA_solar_monthly["glohydrores_country"] = IRENA_solar_monthly["Country"].map(IRENA_GloHydroRes_dict_reversed)
    IRENA_wind_monthly = IRENA_wind_monthly[~IRENA_wind_monthly.glohydrores_country.isna()]
    IRENA_solar_monthly = IRENA_solar_monthly[~IRENA_solar_monthly.glohydrores_country.isna()]

    IRENA_wind_monthly["shape_country"] = IRENA_wind_monthly["glohydrores_country"].map(
        dict(zip(country_shape["glohydrore"], country_shape["Name"]))
    )
    IRENA_solar_monthly["shape_country"] = IRENA_solar_monthly["glohydrores_country"].map(
        dict(zip(country_shape["glohydrore"], country_shape["Name"]))
    )
    
    IRENA_wind_monthly["shape_country_code"] = IRENA_wind_monthly["shape_country"].map(
        dict(zip(country_shape["Name"], country_shape["ID"]))
    )
    IRENA_solar_monthly["shape_country_code"] = IRENA_solar_monthly["shape_country"].map(
        dict(zip(country_shape["Name"], country_shape["ID"]))
    )
    
    IRENA_wind_monthly["country_area"] = IRENA_wind_monthly["shape_country_code"].map(
        dict(zip(country_shape["ID"], country_shape["country_area"]))
    )
    IRENA_solar_monthly["country_area"] = IRENA_solar_monthly["shape_country_code"].map(
        dict(zip(country_shape["ID"], country_shape["country_area"]))
    )

    logger.info("Combined wind and solar data for each country")

    return IRENA_wind_monthly, IRENA_solar_monthly


def get_country_mean_variable(
        df : pd.DataFrame,
        country_code : np.ndarray,
        var_xarray : xr.Dataset,
        var_name : str,
        country_shape : gpd.GeoDataFrame
        ):
    
    """
    Calculate the monthly mean of a variable for each country

    Parameters:
    ----------
    df : pd.DataFrame : dataframe containing the generation data
    country_code : np.ndarray : Array of country codes corresponding to grid cells
    var_xarray : xr.Dataset : xarray dataset containing the variable data
    var_name : str : name of variable
    country_shape : gpd.GeoDataFrame : GeoDataFrame with countries shape
    

    Returns:
    pd.DataFrame : dataframe containing the variable data for each country with columns:
    "glohydrores_plant_id", "date", "glohydrores_country" and the "variable" 
    
    """
    plant_id_list, variable_list, date_list, glohydrores_country_list = [], [], [], []

    # Ensure CRS is the same in both shapefile and xarray

    var_crs = var_xarray.rio.crs
    if var_crs is None and country_shape.crs is not None:
        var_xarray.rio.write_crs(country_shape.crs, inplace=True)
    elif var_crs is not None and var_crs != country_shape.crs:
        country_shape = country_shape.to_crs(var_crs)

    if "time" in var_xarray.dims:
        logger.info("Converting coordinate name time to valid_time")
        var_xarray = var_xarray.rename({"time": "valid_time"})

    var_xarray = var_xarray.load()    

    

    # Use cache to avoid redundant calculations
    mean_cache = {}
            
    for row in tqdm(df.itertuples(index=False), total=len(df), desc=f"Calculating country mean for {var_name}"):
        glohydrores_country, plant_id, date, country_code_value = row.country, row.glohydrores_plant_id, row.date, row.country_code
        
        # Create a unique key
        cache_key = (glohydrores_country, date)

        if cache_key not in mean_cache:
            try:
                # Try to get data for specific date
                sel_data = var_xarray.sel(valid_time = date)

                sel_data_values = sel_data[var_name].values
                country_mean = np.nanmean(sel_data_values[country_code == country_code_value])

                # If country code method fails, try using shapefile
                if np.isnan(country_mean):
                    logger.warning(f"Country code {country_code_value} is not found in country code data")
                    logger.info("Trying to find mean value using shapefile")

                    # Get the country shape for this country code
                    country_shape_code = country_shape[country_shape.ID == country_code_value]

                    clipped_data = sel_data.rio.clip(country_shape_code.geometry, country_shape_code.crs, drop=True)
                    country_mean = clipped_data[var_name].mean().item()

                # Store the results in cache    
                mean_cache[cache_key] = country_mean

            except KeyError as e:
                logger.warning(f"Date {date} not found in {var_name} data: {e}")
                mean_cache[cache_key] = np.nan
            
        else:
            country_mean = mean_cache[cache_key]
        
        country_mean = mean_cache[cache_key]
        plant_id_list.append(plant_id)
        variable_list.append(country_mean)
        date_list.append(date)
        glohydrores_country_list.append(glohydrores_country)   
    return pd.DataFrame({"glohydrores_plant_id": plant_id_list, "date": date_list, "glohydrores_country": glohydrores_country_list, var_name: variable_list})




def calc_var_stat(generation_data : pd.DataFrame,
                var_xarray : xr.Dataset, 
                var_name : str, 
                latitude_name = "lat", 
                longitude_name = "lon"):
    
    """
    Calculate the statitics of the variable for each hydropower plant

    """
    
    current_month_var_list = []
    plant_id_list = []
    date_list = []

    times = var_xarray.time.values
    var_xarray = var_xarray.load()

    #plant_data = generation_data.drop_duplicates(subset = ["glohydrores_plant_id"])    
    for row in tqdm(generation_data.itertuples(index=False), total=len(generation_data), desc=f"Calculating each hydropower station {var_name}" ):
        dam_lat = int(row.pcr_dam_lat_index)
        dam_lon = int(row.pcr_dam_lon_index)
        plant_id = row.glohydrores_plant_id
        current_date = row.date
        
        # Get time index for current date
        current_date_index = np.where(times == np.datetime64(current_date))[0][0]

        try:
            current_month_var = var_xarray.isel({latitude_name: dam_lat, longitude_name: dam_lon, "time": current_date_index})[var_name].values

        except KeyError as e:
            logger.warning(f"Date {current_date} not found in {var_name} data:{e}")
            current_month_var = np.nan
        
        current_month_var_list.append(current_month_var)
        date_list.append(current_date)
        plant_id_list.append(plant_id)
    
    return pd.DataFrame({"glohydrores_plant_id": plant_id_list, "date" : date_list, var_name: current_month_var_list})


def main():
    global_generation_data = process_global_hydropower_data()
    global_generation_data["observed_CF"] = (global_generation_data["observed_generation_mw"])/(global_generation_data["installed_capacity_mw"])
    global_generation_data["observed_CF"].clip(0, 1, inplace=True)
  
    lag_month = 6
    discharge_path = "/home/shah0012/PCRGLOWB_data/discharge/gswp3_w5e5/runs_till_2022/"
    
    simulated_hydro = physical_based_hydropower_func(discharge_path, global_generation_data, "1981", lag_month)


    global_generation_data = global_generation_data.merge(simulated_hydro, on=["glohydrores_plant_id", "date"], how="left")
    min_year = str(global_generation_data["Year"].min())
    max_year = str(global_generation_data["Year"].max())


    logger.info("Loading wind and solar data")

    IRENA_wind_monthly, IRENA_solar_monthly = process_other_renewable_dict()

    global_generation_data["country_code"] = global_generation_data["country"].map(dict(zip(IRENA_wind_monthly["glohydrores_country"], IRENA_wind_monthly["shape_country_code"])))
    global_generation_data["country_area"] = global_generation_data["country"].map(dict(zip(IRENA_wind_monthly["glohydrores_country"], IRENA_wind_monthly["country_area"])))

    logger.info("Merging wind and solar installed capacity data to global generation data")

    global_generation_data = global_generation_data.merge(IRENA_wind_monthly[["glohydrores_country", "Date", "TotalWind"]], left_on=["country", "date"], right_on=["glohydrores_country", "Date"], how="left")
    global_generation_data = global_generation_data.merge(IRENA_solar_monthly[["glohydrores_country", "Date", "SolarPV"]], left_on=["country", "date"], right_on=["glohydrores_country", "Date"], how="left")
    global_generation_data.rename(columns={"TotalWind": "IRENA_wind_installed_capacity", "SolarPV": "IRENA_solar_installed_capacity"}, inplace=True)

    logger.info("Loading solar capacity factor data")
    solar_cf_path = "/scratch/shah0012/ERA5_Land_processed/solar_CF_factor/global_monthly_capacity_factor_solar_2000_2022_correctlon.nc"
    solar_cf_data = xr.open_dataset(solar_cf_path)
    solar_cf_data = solar_cf_data.sel(valid_time=slice(min_year, max_year))

    # Convert solar CF time to 1st of month if needed
    times = solar_cf_data.valid_time.values
    if np.any(pd.DatetimeIndex(times).day != 1):
        logger.info("Converting solar CF dates to 1st of month")
        new_times = pd.DatetimeIndex([pd.Timestamp(year=pd.Timestamp(t).year, 
                                                 month=pd.Timestamp(t).month, 
                                                  day=1) for t in times])
        solar_cf_data = solar_cf_data.assign_coords(valid_time=new_times)

    
    logger.info("Loading wind capacity factor data")
    wind_cf_path =  "/scratch/shah0012/ERA5_Land_processed/wind_CF_factor/global_monthly_capacity_factor_wind_2000_2022_correctlon.nc"
    wind_cf_data = xr.open_dataset(wind_cf_path)
    wind_cf_data = wind_cf_data.sel(valid_time=slice(min_year, max_year))

    # Convert wind CF time to 1st of month if needed
    times = wind_cf_data.valid_time.values
    if np.any(pd.DatetimeIndex(times).day != 1):
        logger.info("Converting wind CF dates to 1st of month")
        new_times = pd.DatetimeIndex([pd.Timestamp(year=pd.Timestamp(t).year, 
                                                    month=pd.Timestamp(t).month, 
                                                    day=1) for t in times])
        wind_cf_data = wind_cf_data.assign_coords(valid_time=new_times)

    logger.info("Loading grid cell area data")
    grid_cell_area_path = f"{base_path}country_shape_file/Global_grid_cell_area_point1_degree_resampled.nc"
    grid_cell_area = xr.open_dataset(grid_cell_area_path)
    grid_cell_area = grid_cell_area.reindex(lat=grid_cell_area.lat[::-1]) # This is done because latitude in grid cell area starting from -90 rather than 90


    # Get country code data
    country_code = grid_cell_area["country_code"].values

    logger.info("Shape file of countries")
    country_shape_path = f"{base_path}country_shape_file/country_shape.shp"
    country_shape = gpd.read_file(country_shape_path)

    
    logger.info("Calculating wind and solar capacity factor for each country")
    wind_cf_country_means =  get_country_mean_variable(global_generation_data, country_code, wind_cf_data, "wind_capacity_factor", country_shape)
    global_generation_data = global_generation_data.merge(wind_cf_country_means[["glohydrores_plant_id", "date", "wind_capacity_factor"]], on = ["glohydrores_plant_id", "date"], how = "left")

    solar_cf_country_means = get_country_mean_variable(global_generation_data, country_code, solar_cf_data, "solar_capacity_factor", country_shape)
    global_generation_data = global_generation_data.merge(solar_cf_country_means[["glohydrores_plant_id", "date", "solar_capacity_factor"]], on = ["glohydrores_plant_id", "date"], how = "left")

    global_generation_data["Solar_monthly_generation"] = global_generation_data["solar_capacity_factor"] * global_generation_data["IRENA_solar_installed_capacity"]*global_generation_data["date"].dt.days_in_month * 24
    global_generation_data["Wind_monthly_generation"] = global_generation_data["wind_capacity_factor"] * global_generation_data["IRENA_wind_installed_capacity"]*global_generation_data["date"].dt.days_in_month * 24

    global_generation_data["Solar_monthly_generation_persqkm"] = global_generation_data["Solar_monthly_generation"]/global_generation_data["country_area"]
    global_generation_data["Wind_monthly_generation_persqkm"] = global_generation_data["Wind_monthly_generation"]/global_generation_data["country_area"]

    logger.info("Calculating Cooling Degree Days and Heating Degree Days for each country")

    logger.info("Loading Cooling Degree Days data")

    CDD_data = xr.open_mfdataset(
            "/scratch/shah0012/ERA5_Land_processed/CDD_HDD_files/" + "ERA5_Land_CDD*.nc",
            concat_dim="valid_time",
            combine="nested",
            data_vars="minimal"

        )
    
    CDD_data = CDD_data.sel(valid_time=slice(min_year, max_year))
    #CDD_data = CDD_data.load()
    
    # Convert CDD time to 1st of month if needed
    times = CDD_data.valid_time.values
    if np.any(pd.DatetimeIndex(times).day != 1):
        logger.info("Converting CDD dates to 1st of month")
        new_times = pd.DatetimeIndex([pd.Timestamp(year=pd.Timestamp(t).year, 
                                                    month=pd.Timestamp(t).month, 
                                                    day=1) for t in times])
        CDD_data = CDD_data.assign_coords(valid_time=new_times)

    logger.info("Loading Heating Degree Days data")

    HDD_data = xr.open_mfdataset(
            "/scratch/shah0012/ERA5_Land_processed/CDD_HDD_files/" + "ERA5_Land_HDD*.nc",
            concat_dim="valid_time",
            combine="nested",
            data_vars="minimal"

        )
    
    HDD_data = HDD_data.sel(valid_time=slice(min_year, max_year))
    
    # Convert HDD time to 1st of month if needed
    times = HDD_data.valid_time.values
    if np.any(pd.DatetimeIndex(times).day != 1):
        logger.info("Converting HDD dates to 1st of month")
        new_times = pd.DatetimeIndex([pd.Timestamp(year=pd.Timestamp(t).year, 
                                                    month=pd.Timestamp(t).month, 
                                                    day=1) for t in times])
        HDD_data = HDD_data.assign_coords(valid_time=new_times)


    cdd_cf_country_means =  get_country_mean_variable(global_generation_data, country_code, CDD_data, "CDD", country_shape)
    global_generation_data = global_generation_data.merge(cdd_cf_country_means[["glohydrores_plant_id", "date", "CDD"]], on = ["glohydrores_plant_id", "date"], how = "left")

    hdd_cf_country_means = get_country_mean_variable(global_generation_data, country_code, HDD_data, "HDD", country_shape)
    global_generation_data = global_generation_data.merge(hdd_cf_country_means[["glohydrores_plant_id", "date", "HDD"]], on = ["glohydrores_plant_id", "date"], how = "left")


    logger.info("Calculating temperature for each hydropower plant")
    temperature_path = "/home/shah0012/PCRGLOWB_data/Temperature/gswp3-w5e5/runs_till_2022/combined_temperature_monthAvg_output_1981_to_2022.nc"

    temperature_data = xr.open_dataset(
        temperature_path)
    
    temperature_data = temperature_data.sel(time=slice(min_year, max_year))
    #temperature_data = temperature_data.load()
    
    
    times = temperature_data.time.values
    if np.any(pd.DatetimeIndex(times).day != 1):
        logger.info("Converting temperature dates to 1st of month")
        new_times = pd.DatetimeIndex([pd.Timestamp(year=pd.Timestamp(t).year, 
                                                    month=pd.Timestamp(t).month, 
                                                    day=1) for t in times])
        temperature_data = temperature_data.assign_coords(time=new_times)

    temperature_country_means = calc_var_stat(global_generation_data, temperature_data, "temperature", "latitude", "longitude")

    global_generation_data = global_generation_data.merge(temperature_country_means[["glohydrores_plant_id", "date", "temperature"]], on = ["glohydrores_plant_id", "date"], how = "left")

    col_names = ['plant_name', 'country',
             'glohydrores_plant_id', 'date',
             "installed_capacity_mw",
             'observed_generation_mw', 'plant_type', 
             'res_vol_km3', 'res_area_km2', 
             'res_depth_m', 'final_head_m',
            'pcr_dam_lat_index', 'pcr_dam_lon_index',
            'observed_CF',
       'physical_hydropower_0n', 'physical_hydropower_1n',
       'physical_hydropower_2n', 'physical_hydropower_3n',
       'physical_hydropower_4n', 'physical_hydropower_5n',
       'physical_hydropower_6n', 'Solar_monthly_generation_persqkm',
       'Wind_monthly_generation_persqkm', 'CDD', 'HDD', 'temperature']

    global_generation_data[col_names].to_csv("/scratch/shah0012/hybrid_hydropower_model/data/hybrid_model_data/hybrid_model_train_generation_data_1981_2022_selected_variables.csv", index=False)
    global_generation_data.to_csv("/scratch/shah0012/hybrid_hydropower_model/data/hybrid_model_data/hybrid_model_train_generation_data_1981_2022_all_variables.csv", index=False)

    # Update 
    return global_generation_data


if __name__ == "__main__":
    global_generation_data = main()













    








    











