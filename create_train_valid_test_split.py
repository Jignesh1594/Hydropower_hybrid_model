import pandas as pd
import numpy as np
from typing import Dict
import logging

"""
Create manual split for training, validation and testing datasets. For this paper, threshold years which divide the data into training, validation and testing datasets varies 
for each country based on the number of hydropower plants and time period for which observed data is available.

For Norway split is done seprately because the country has distinct energy pattern which can provide variability to input data. 70% of data goes for training
and rest 30% for testing.

"""

logging.basicConfig(
    level = logging.INFO,
    format = "%(asctime)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)

def create_country_splits(hybrid_model_data: pd.DataFrame) -> Dict[str, pd.DataFrame]:

    """
    Create train/validation/test splits for each country based on manually identifed thresholds

    """

    split_config = {
        "Austria": {"train_end": 2019, "valid_year": 2020, "test_start": 2021},
        "Bosnia and Herzegovina": {"train_end": 2020, "valid_year": None, "test_start": 2021},
        "Brazil": {"train_end": 2017, "valid_year": 2018, "test_start": 2019},
        "France": {"train_end": 2019, "valid_year": 2020, "test_start": 2021},
        "Germany": {"train_end": 2019, "valid_year": 2020, "test_start": 2021},
        "Greece": {"train_end": 2019, "valid_year": 2020, "test_start": 2021},
        "India": {"train_end": 2020, "valid_year": None, "test_start": 2021},
        "Ireland": {"train_end": 2019, "valid_year": 2020, "test_start": 2021},
        "Italy": {"train_end": 2019, "valid_year": 2020, "test_start": 2021},
        "Latvia": {"train_end": 2019, "valid_year": 2020, "test_start": 2021},
        "Montenegro": {"train_end": 2019, "valid_year": 2020, "test_start": 2021},
        "Portugal": {"train_end": 2019, "valid_year": 2020, "test_start": 2021},
        "Russia": {"train_end": 2018, "valid_year": 2019, "test_start": 2020},
        "Sweden": {"train_end": 2019, "valid_year": 2020, "test_start": 2021},
        "Switzerland": {"train_end": 2019, "valid_year": 2020, "test_start": 2021},
        "United Kingdom": {"train_end": 2016, "valid_year": 2017, "test_start": 2018},
        "United States of America": {"train_end": 2012, "valid_year": 2013, "test_start": 2014}
    }

    logger.info("Splitting the dataset based on manually selected threholds")
    splits = {}

    for country, config in split_config.items():
        country_data = hybrid_model_data[hybrid_model_data.country == country].copy()

        if len(country_data) == 0:
            logger.warning(f"Warning: No data found for country {country}")
            continue
        
        # Create splits

        train_data = country_data[country_data.date.dt.year <= config["train_end"]]

        if config["valid_year"] is not None:
             valid_data = country_data[country_data.date.dt.year == config["valid_year"]]

        else:
            valid_data = pd.DataFrame()

        test_data = country_data[country_data.date.dt.year >= config["test_start"]]

        splits[country] = {
            "train": train_data,
            "valid" : valid_data,
            "test": test_data

        }

        logger.info(f"{country}: Train={len(train_data)}, Valid = {len(valid_data)}, Test = {len(test_data)}")

    return splits



def create_norway_split(hybrid_model_data: pd.DataFrame, train_ratio: float = 0.7, valid_date: int = "2022-05-01", test_start: int = "2022-06-01") -> Dict[str, pd.DataFrame]:

    """
    Create plant-based splits for Norway 

    """

    logger.info("Split the data for Norway")

    norway_data = hybrid_model_data[hybrid_model_data.country == "Norway"].copy()
    plants = norway_data.glohydrores_plant_id.unique()

    # Split plants
    n = len(plants)
    train_plants = plants[: int(n * train_ratio)]
    test_plants = plants[len(train_plants):]

     # Create splits
    norway_train = norway_data[norway_data["glohydrores_plant_id"].isin(train_plants)]
    norway_valid = norway_data[
        (norway_data['glohydrores_plant_id'].isin(test_plants)) & 
        (norway_data.date <= valid_date)
    ]

    norway_test = norway_data[
        (norway_data['glohydrores_plant_id'].isin(test_plants)) & 
        (norway_data.date >= test_start)
    ]

    print(f"Norway plant split: {len(train_plants)} train plants, {len(test_plants)} test plants")
    print(f"Norway data: Train={len(norway_train)}, Valid={len(norway_valid)}, Test={len(norway_test)}")


    return {
        "train": norway_train,
        "valid": norway_valid,
        "test": norway_test,
        "train_plants": train_plants,
        "test_plants": test_plants
    }

def main(hybrid_model_data: pd.DataFrame, designed_train_file, designed_valid_file, designed_test_file):

    logging.info("Split the dataset into training, validation and test dataset except for Norway")

    country_splits = create_country_splits(hybrid_model_data)

    logging.info("Split the dataset into training, validation and test dataset for Norway")

    norway_splits = create_norway_split(hybrid_model_data)

    # Combine all training data
    all_train_data = []
    all_valid_data = []
    all_test_data = []

    # Add country data
    for country, splits in country_splits.items():
        all_train_data.append(splits["train"])
        if len(splits["valid"]) > 0:
            all_valid_data.append(splits["valid"])
        all_test_data.append(splits["test"])
    
    # Add Norway data
    all_train_data.append(norway_splits["train"])
    all_valid_data.append(norway_splits["valid"])
    all_test_data.append(norway_splits["test"])

    # Concatenate all splits
    designed_train_data = pd.concat(all_train_data, ignore_index=True)
    designed_valid_data = pd.concat(all_valid_data, ignore_index=True) if all_valid_data else pd.DataFrame()
    designed_test_data = pd.concat(all_test_data, ignore_index = True)

    logger.info(f"Storing the test dataset as {designed_train_file}, valid dataset as {designed_valid_file} and test dataset as {designed_test_file}")

    # Optional: Save the splits
    designed_train_data.to_csv(designed_train_file, index=False)
    designed_valid_data.to_csv(designed_valid_file, index=False)
    designed_test_data.to_csv(designed_test_file, index=False)

if __name__ == "__main__":
    main(hybrid_model_data = "/scratch/shah0012/hybrid_hydropower_model/data/hybrid_model_data/hybrid_model_train_generation_data_1981_2022_selected_variables_snow_cover_historical_plant_lat.csv",
    designed_train_file = "/scratch/shah0012/hybrid_hydropower_model/data/hybrid_model_data/designed_train_data.csv",
    designed_valid_file = "/scratch/shah0012/hybrid_hydropower_model/data/hybrid_model_data/designed_valid_data.csv",
    designed_test_file = "/scratch/shah0012/hybrid_hydropower_model/data/hybrid_model_data/designed_test_data.csv")




