# Impact of each drought on hydropower

# 1. Time series of hydropower generation of each power plant from 1981 to 2022
#    1.1 Fix a hybrid model 
#    1.2 For each plant run the model and simulate the generation
# 2. Time series of drought index of each drought from 1981 to 2022


import pandas as pd
import logging
import glob
import os
import numpy as np



logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


input_precentile_dir = "/scratch/shah0012/hybrid_hydropower_model/data/drought_identification/"
input_global_dir = "/scratch/shah0012/hybrid_hydropower_model/hybrid_model_results/"

plant_files = glob.glob(f"{input_precentile_dir}/*.csv")
logger.info(f"Found {len(plant_files)} plant files")

dfs = []
for file in plant_files:
    print(file)
    plant_id = file.split("/")[-1].split(".")[0]
    df = pd.read_csv(file)
    df["glohydrores_plant_id"] = plant_id
    dfs.append(df)

plant_percentile_data = pd.concat(dfs, ignore_index=True)
plant_percentile_data["date"] = pd.to_datetime(plant_percentile_data["date"])
plant_percentile_data["glohydrores_plant_id"].unique()
plant_percentile_data["date"] = plant_percentile_data["date"].dt.to_period('M').dt.to_timestamp()

input_global_data = pd.read_csv(input_global_dir + "global_data_prediction_designed_dataset_1982_2022_target_CF.csv", parse_dates=["date"])
input_global_data["glohydrores_plant_id"].unique()

input_global_data = input_global_data.merge(plant_percentile_data, on=["date", "glohydrores_plant_id"], how="left")
input_global_data.to_csv(input_global_dir + "global_data_prediction_designed_dataset_drought_attributes_included_1982_2022_target_CF.csv", index = False)
