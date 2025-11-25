import pandas as pd
import numpy as np
import glob
import logging
import shutil
import os
import argparse

## Multiprocessing function
from multiprocessing import Pool
from functools import partial

"""
This section identifies drought periods for each hydropower plant using defined streamflow thresholds and drought duration criteria.

# drought thresholds
# percentile length
#  20         >= 1
#  20         >= 3
#  20         >= 6
#  20         >= 12
#  25         >= 1
#  25         >= 3
#  25         >= 6
#  25         >= 12
#  10          >= 1    
#  10          >= 3
#  10          >= 6
#  10          >= 12

"""

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def drought_pooling(series):

    """
    If the drought is less than 2 months apar, pool them together
    
    Args:
    series: pandas series

    Returns:
    np.array : Modified series with drought pooling applied
    
    """

    series = np.array(series)
    result = series.copy()

    for i in range(1, len(series) - 1):
        if series[i] == 0 and series[i-1] == 1 and series[i+1] == 1:
            result[i] = 1
    
    return result

def filter_short_group(series, min_length=3):
    mask = (series != series.shift()).cumsum()
    group_sizes = series.groupby(mask).transform('size')
    return mask.where((series == 1) & (group_sizes >= min_length), 0)


def process_plant(plant_file, output_dir):
    """Process a single hydropower plant to identify drought conditions"""

    logger.info(f"Processing plant {plant_file}")
    plant_data = pd.read_csv(plant_file)
    plant_id = plant_file.split("/")[-1].split(".")[0]
    drought_10 = plant_data["percentile"].apply(lambda x: 1 if x < 10 else 0)
    drought_20 = plant_data["percentile"].apply(lambda x: 1 if x < 20 else 0)
    drought_25 = plant_data["percentile"].apply(lambda x: 1 if x < 25 else 0)
    result_10 = drought_pooling(drought_10)
    result_20 = drought_pooling(drought_20)
    result_25 = drought_pooling(drought_25)
    result_10 = pd.Series(result_10)
    result_20 = pd.Series(result_20)
    result_25 = pd.Series(result_25)
    drought_10_pooled_1length = filter_short_group(result_10, 1)
    drought_20_pooled_1length = filter_short_group(result_20, 1)
    drought_25_pooled_1length = filter_short_group(result_25, 1)
    drought_10_pooled_3length = filter_short_group(result_10, 3)
    drought_10_pooled_6length = filter_short_group(result_10, 6)
    drought_10_pooled_12length = filter_short_group(result_10, 12)
    drought_20_pooled_3length = filter_short_group(result_20, 3)
    drought_20_pooled_6length = filter_short_group(result_20, 6)
    drought_20_pooled_12length = filter_short_group(result_20, 12)
    drought_25_pooled_3length = filter_short_group(result_25, 3)
    drought_25_pooled_6length = filter_short_group(result_25, 6)
    drought_25_pooled_12length = filter_short_group(result_25, 12)

    plant_data["drought_10_pooled_1length"] = drought_10_pooled_1length
    plant_data["drought_20_pooled_1length"] = drought_20_pooled_1length
    plant_data["drought_25_pooled_1length"] = drought_25_pooled_1length
    plant_data["drought_10_pooled_3length"] = drought_10_pooled_3length
    plant_data["drought_10_pooled_6length"] = drought_10_pooled_6length
    plant_data["drought_10_pooled_12length"] = drought_10_pooled_12length
    plant_data["drought_20_pooled_3length"] = drought_20_pooled_3length
    plant_data["drought_20_pooled_6length"] = drought_20_pooled_6length
    plant_data["drought_20_pooled_12length"] = drought_20_pooled_12length
    plant_data["drought_25_pooled_3length"] = drought_25_pooled_3length
    plant_data["drought_25_pooled_6length"] = drought_25_pooled_6length
    plant_data["drought_25_pooled_12length"] = drought_25_pooled_12length
    output_path = os.path.join(output_dir, f"{plant_id}.csv")

    plant_data.to_csv(output_path, index=False)

    return plant_data


def main(input_dir, output_dir):

    logger.info("Starting drought identification process")

    # Create or clear output directory
    if os.path.exists(output_dir):
        logger.info(f"Output directory {output_dir} already exists. Clearing directory")
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)
    logger.info(f"Created output directory {output_dir}")

    # Load power plant percentile time series files
    plant_files = glob.glob(f"{input_dir}/*.csv")
    logger.info(f"Found {len(plant_files)} plant files")

    process_plant_partial = partial(process_plant, output_dir=output_dir)

    # Process each plant in parallel
    with Pool(processes=8) as p:
        results = p.map(process_plant_partial, plant_files)



if __name__ == "__main__":
    parser = argparse.ArgumentParse(
        "Code to indentify the drought based different percentile threshold and minimum duration"

    )

    parser.add_argument(
        "--input_dir",
        default="/scratch/shah0012/hybrid_hydropower_model/data/percentile_files/",
        help="Directory containing percentile csv files"
    )

    parser.add_argument(
        "--output_dir",
        default="/scratch/shah0012/hybrid_hydropower_model/data/drought_identification/",
        help="Directory to store csv files"
    )

    args = parser.parse_args()

    main(
        input_dir = args.input_dir,
        output_dir  = args.output_dir)



