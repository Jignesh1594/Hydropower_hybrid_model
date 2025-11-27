import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import numpy as np
import logging
import argparse

"""

Multi Linear Regression Model

"""

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)



# Define float range
float16_max = np.finfo(np.float16).max
float16_min = np.finfo(np.float16).min
float32_max = np.finfo(np.float32).max
float32_min = np.finfo(np.float32).min

def convert_to_float1632(column):
    logger.info("Converting the data type of columns for better memory handling")

    if column.dtype == 'float64':
        if column.max() < float16_max and column.min() > float16_min:
            column = column.astype('float16')
        elif column.max() < float32_max and column.min() > float32_min:
            column = column.astype('float32')
    return column


def main(train_file, valid_file, test_file, output_file,  min_observations = 24):
    logger.info(
    f"Loading dataset files — "
    f"train: {train_file}, validation: {valid_file}, test: {test_file}")

    train_data = pd.read_csv(train_file, parse_dates=['date'])
    valid_data = pd.read_csv(valid_file, parse_dates=['date'])
    test_data = pd.read_csv(test_file, parse_dates=['date'])

    features = ["date", "glohydrores_plant_id", "physical_hydropower_6n", "physical_hydropower_5n", "physical_hydropower_4n",
    "physical_hydropower_3n", "physical_hydropower_2n", "physical_hydropower_1n", "physical_hydropower_0n", "temperature", 
    "CDD", "HDD", "res_vol_km3", "installed_capacity_mw", "Solar_monthly_generation_persqkm", "Wind_monthly_generation_persqkm",
    "observed_generation_mw", "plant_lat", "snowcover_6n", "snowcover_5n", "snowcover_4n", "snowcover_3n", "snowcover_2n", "snowcover_1n",
    "snowcover_0n", "country"]

    train_data = train_data.loc[:, features]
    valid_data = valid_data.loc[:, features]
    test_data = test_data.loc[:, features]

    logger.info("Removing some hydropower plants data due to data quality issues identified during manual inspection")

    train_plants_to_drop = ["GHR00387", "GHR00388",  "GHR00394", "GHR00405", "GHR00412" , 
                        "GHR00420", "GHR00432", "GHR00435", "GHR00439", "GHR00552",  "GHR00594",
                        "GHR00599",  "GHR00617", "GHR00628", "GHR00643", "GHR00644", "GHR00649", 
                        "GHR00667", "GHR00740", "GHR00852", "GHR00926", "GHR00931", "GHR03412",
                        "GHR03446",  "GHR03459", "GHR03461", "GHR03468", "GHR03488", "GHR03511",
                        "GHR03533", "GHR03595", "GHR06434",  "GHR06512", "GHR06518", "GHR06524",
                        "GHR06540", "GHR06542",  "GHR06554", "GHR06563", "GHR06596", "GHR06615",
                        "GHR06651", "GHR06664", "GHR06687", "GHR06689", "GHR06702", "GHR06709",
                        "GHR06739", "GHR06772",  "GHR06773", "GHR06825", "GHR06852", "GHR06855",
                        "GHR06871", "GHR06929", "GHR06942", "GHR06950", "GHR06960", "GHR06980",
                        "GHR07010", "GHR07069", "GHR07133", "GHR07140", "GHR07147", "GHR07158",
                        "GHR07206", "GHR07208", "GHR07266", "GHR07273", "GHR07302", "GHR07330",
                        "GHR07344", "GHR07366", "GHR07372", "GHR07386", "GHR07396", "GHR07442",
                        "GHR07457", "GHR07474", "GHR07511", "GHR07530", "GHR07536"]

    train_data = train_data[~train_data.glohydrores_plant_id.isin(train_plants_to_drop)]
    valid_data = valid_data[~valid_data.glohydrores_plant_id.isin(train_plants_to_drop)]
    test_data = test_data[~test_data.glohydrores_plant_id.isin(train_plants_to_drop)]

    logger.info("Removing hydropower plants with installed capacity less than 10 MW")
    train_data  = train_data[train_data.installed_capacity_mw > 10]
    valid_data  = valid_data[valid_data.installed_capacity_mw > 10]
    test_data  = test_data[test_data.installed_capacity_mw > 10]

    logger.info(f"Removing plants with less than {min_observations} months data")

    data = pd.concat([train_data, valid_data, test_data])
    plant_observation_data_counts = data.groupby("glohydrores_plant_id").size().reset_index()
    plant_observation_data_counts.columns = ["glohydrores_plant_id", "count"]
    insufficient_plants = plant_observation_data_counts[plant_observation_data_counts["count"] < min_observations].glohydrores_plant_id.tolist()

    if len(insufficient_plants) > 0:
        train_data = train_data[~train_data.glohydrores_plant_id.isin(insufficient_plants)]
        valid_data = valid_data[~valid_data.glohydrores_plant_id.isin(insufficient_plants)]
        test_data = test_data[~test_data.glohydrores_plant_id.isin(insufficient_plants)]

    train_data = train_data.dropna(axis = 0)
    valid_data = valid_data.dropna(axis = 0)
    test_data = test_data.dropna(axis = 0)

    train_data = train_data.apply(convert_to_float1632)
    valid_data = valid_data.apply(convert_to_float1632)
    test_data = test_data.apply(convert_to_float1632)

    logger.info("Calculate capacity factor for observed generation")

    target_col = "observed_generation_mw"
    target_cf_col = "observed_plant_CF"
    
    train_data[target_cf_col] = train_data[target_col]/train_data["installed_capacity_mw"]
    valid_data[target_cf_col] = valid_data[target_col]/valid_data["installed_capacity_mw"]
    test_data[target_cf_col] = test_data[target_col]/test_data["installed_capacity_mw"]

    feature_cols = [
    "physical_hydropower_0n", "physical_hydropower_1n", "physical_hydropower_2n",
    "physical_hydropower_3n", "physical_hydropower_4n", "physical_hydropower_5n", 
    "physical_hydropower_6n", "temperature", "CDD", "HDD",
    "Solar_monthly_generation_persqkm", "Wind_monthly_generation_persqkm", 
    "snowcover_6n", "snowcover_5n", "snowcover_4n", "snowcover_3n", 
    "snowcover_2n", "snowcover_1n", "snowcover_0n"]

    output_cols = [
    'physical_0n_plant_std', 'physical_1n_plant_std', 'physical_2n_plant_std',
    'physical_3n_plant_std', 'physical_4n_plant_std', 'physical_5n_plant_std',
    'physical_6n_plant_std', 'temperature_std', 'CDD_std', 'HDD_std',
    'Solar_monthly_generation_persqkm_std', 'Wind_monthly_generation_persqkm_std',
    'snowcover_6n_std', 'snowcover_5n_std', 'snowcover_4n_std', 'snowcover_3n_std',
    'snowcover_2n_std', 'snowcover_1n_std', 'snowcover_0n_std']

    logger.info("Standardized the input features before training the model")

    
    # 1. If each plant train data is less than 24 months then combine valid and test data for that plant in train 
    logger.info("Ensure hydropower plants used for training have 24 months of observed data")
    train_counts = train_data.groupby("glohydrores_plant_id").size().reset_index()
    train_counts.columns = ["glohydrores_plant_id", "count"]
    insufficient_plants = train_counts[train_counts["count"] < min_observations].glohydrores_plant_id.tolist()

    if len(insufficient_plants) > 0:
        logger.info(f"Found {len(insufficient_plants)} plants with <{min_observations} training observations")
        valid_to_train = valid_data[valid_data.glohydrores_plant_id.isin(insufficient_plants)]
        test_to_train = test_data[test_data.glohydrores_plant_id.isin(insufficient_plants)]

        logger.info("Move these hydropower plants data from validation and test datasets to training")
        train_data = pd.concat([train_data, valid_to_train, test_to_train], ignore_index=True)
        valid_data = valid_data[~valid_data.glohydrores_plant_id.isin(insufficient_plants)]
        test_data = test_data[~test_data.glohydrores_plant_id.isin(insufficient_plants)]

    logger.info("Some hydropower plants are not available in training dataset but only in validation dataset")
    train_plants = train_data.glohydrores_plant_id.unique()
    no_train_plants_valid_data = valid_data[~valid_data.glohydrores_plant_id.isin(train_plants)]
    notrain_valid_counts = no_train_plants_valid_data.groupby("glohydrores_plant_id").size().reset_index()
    notrain_valid_counts.columns = ["glohydrores_plant_id", "count"]
    insufficient_notrain_valid_plants = notrain_valid_counts[notrain_valid_counts["count"] < min_observations].glohydrores_plant_id.tolist()

    if len(insufficient_notrain_valid_plants) > 0:
        logger.info(f"Found {len(insufficient_notrain_valid_plants)} plants with <{min_observations} for plants generation data starting from valid dataset")
        logger.info("Moving such plants data from valid and test to train")
        valid_to_train = valid_data[valid_data.glohydrores_plant_id.isin(insufficient_notrain_valid_plants)]
        test_to_train = test_data[test_data.glohydrores_plant_id.isin(insufficient_notrain_valid_plants)]
        train_data = pd.concat([train_data, valid_to_train, test_to_train], ignore_index=True)
        valid_data = valid_data[~valid_data.glohydrores_plant_id.isin(insufficient_notrain_valid_plants)]
        test_data = test_data[~test_data.glohydrores_plant_id.isin(insufficient_notrain_valid_plants)]
    

    # Strategy 6: Plant-wise StandardScaler
    logger.info(f"Shape of train data is {train_data.shape}")
    logger.info(f"Shape of test data is {test_data.shape}")
    logger.info(f"Shape of valid data is {valid_data.shape}")

    unique_plants = pd.concat([train_data, valid_data, test_data]).glohydrores_plant_id.unique().tolist()

    logger.info("Start scaling the features")
    scalers = {}
    for plant in unique_plants:
        logger.info(f"Scaling for the plant {plant}")
        plant_mask = train_data["glohydrores_plant_id"] == plant
        plant_data = train_data.loc[plant_mask, feature_cols]

        if len(plant_data) > 1:
            scaler = StandardScaler()
            plant_scaled = scaler.fit_transform(plant_data.values)
            scalers[plant] = scaler
            train_data.loc[plant_mask, output_cols] = plant_scaled
        
        valid_mask =  valid_data["glohydrores_plant_id"] == plant
        plant_data = valid_data.loc[valid_mask, feature_cols]

        if (len(plant_data) > 1) & (plant in scalers):
            logger.info(f"For plant {plant} already scaling done in training")
            scaler = scalers[plant]
            plant_scaled = scaler.transform(plant_data.values)
            valid_data.loc[valid_mask, output_cols] = plant_scaled

        elif (len(plant_data) > 1) & (plant not in scalers):
            logger.info(f"For plant {plant} scaling was not done in training therefore going to do in validation")
            scaler = StandardScaler()
            plant_scaled = scaler.fit_transform(plant_data.values)
            valid_data.loc[valid_mask, output_cols] = plant_scaled
            scalers[plant] = scaler

        else:
            logger.info(f"Plant {plant} data is not avaialble for validation dataset")


        test_mask =  test_data["glohydrores_plant_id"] == plant
        plant_data = test_data.loc[test_mask, feature_cols]

        if (len(plant_data) > 1) & (plant in scalers):
            logger.info(f"For plant {plant} already scaling done in training")
            plant_scaled = scaler.transform(plant_data.values)
            test_data.loc[test_mask, output_cols] = plant_scaled
        
        elif (len(plant_data) > 1) & (plant not in scalers):
            logger.info(f"For plant {plant} scaling was not done in training therefore going to do in test dataset")
            scaler = StandardScaler()
            plant_scaled = scaler.fit_transform(plant_data.values)
            test_data.loc[test_mask, output_cols] = plant_scaled
            scalers[plant] = scaler

        else:
            logger.info(f"Plant {plant} data is not avaialble for test dataset")

    standardized_features = ["features", "date", "glohydrores_plant_id", "physical_6n_plant_std", "physical_4n_plant_std",
    "physical_3n_plant_std", "physical_2n_plant_std",  "physical_1n_plant_std",  "physical_0n_plant_std",  "temperature_std",
    "CDD_std", "HDD_std", "res_vol_km3", "installed_capacity_mw", "Solar_monthly_generation_persqkm_std", "Wind_monthly_generation_persqkm_std",
    "observed_plant_CF", "plant_lat", "snowcover_6n_std", "snowcover_5n_std", "snowcover_4n_std", "snowcover_3n_std", "snowcover_2n_std","snowcover_1n_std",
    "snowcover_0n_std", "country"]
    
    train_data_selected = train_data.loc[:, standardized_features]
    valid_data_selected = valid_data.loc[:, standardized_features]
    test_data_selected = test_data.loc[:, standardized_features]

    features_to_limit = [
    'physical_0n_plant_std', 'physical_1n_plant_std', 'physical_2n_plant_std',
    'physical_3n_plant_std', 'physical_4n_plant_std', 'physical_5n_plant_std',
    'physical_6n_plant_std', 'temperature_std', 'CDD_std', 'HDD_std',
    'snowcover_6n_std', 'snowcover_5n_std', 'snowcover_4n_std', 'snowcover_3n_std',
    'snowcover_2n_std', 'snowcover_1n_std', 'snowcover_0n_std']

    logger.info("Removing those data points where standardized values are lower than -3 standard deviation or higher that 3 standard deviation")

    train_data_selected = train_data_selected[(train_data_selected.loc[:, features_to_limit] >= -3).all(axis=1) & (train_data_selected.loc[:, features_to_limit] <= 3).all(axis=1)]
    valid_data_selected = valid_data_selected[(valid_data_selected.loc[:, features_to_limit] >= -3).all(axis=1) & (valid_data_selected.loc[:, features_to_limit] <= 3).all(axis=1)]
    test_data_selected = test_data_selected[(test_data_selected.loc[:, features_to_limit] >= -3).all(axis=1) & (test_data_selected.loc[:, features_to_limit] <= 3).all(axis=1)]

    logger.info("Train the model")

    model = LinearRegression()
    model.fit(train_data_selected.drop(columns = ["date", "glohydrores_plant_id", "country", target_cf_col]).values, train_data_selected.loc[:,target_cf_col].values)
    test_data_selected["predicted_CF"] = model.predict(test_data_selected.drop(columns = ["date", "glohydrores_plant_id", "country", target_cf_col]).values)

    logger.info("Coverting predicted capacity factor to power (MW)")

    test_data_selected["originial_prediction"]  = test_data_selected["predicted_CF"] * test_data_selected["installed_capacity_mw"]

    logger.info("Store the final output")
    test_data_selected.to_csv(output_file, index=False)




if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        "Linear model training"

    )

    parser.add_argument(
        "--train_file_dir",
        default="/scratch/shah0012/hybrid_hydropower_model/data/hybrid_model_data/designed_train_data.csv",
        help="Directory with file name (csv format) contain training data"
    )

    parser.add_argument(
        "--valid_file_dir",
        default="/scratch/shah0012/hybrid_hydropower_model/data/hybrid_model_data/designed_valid_data.csv",
        help="Directory with file name (csv format) contain validation data"
    )

    parser.add_argument(
        "--test_file_dir",
        default="/scratch/shah0012/hybrid_hydropower_model/data/hybrid_model_data/designed_test_data.csv",
        help="Directory with file name (csv format) contain test data"
    )

    parser.add_argument(
        "--output_dir",
        default="/scratch/shah0012/hybrid_hydropower_model/hybrid_model_results/lr_test_result.csv",
        help="Directory with file name (csv format) to store the final results"
    )

    parser.add_argument(
        "--min_observations",
        default=24,
        help="Minimum number of months for which observed data needs to be available for each plant",
        type = int
    )

    args = parser.parse_args()

    main(
        train_file = args.train_file_dir,
        valid_file =  args.valid_file_dir,
        test_file  = args.test_file_dir,
        output_file  = args.output_dir,
        min_observations  = args.min_observations)


  


        
















