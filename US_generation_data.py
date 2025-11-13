from unittest import defaultTestLoader
import pandas as pd
import geopandas as gpd
import numpy as np
import logging
import argparse
import os

"""

This script will clean and harmonized USA generation data. Only those plants are included which are also available in GloHydroRes dataset

"""
# Configures global logging system
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# Now creates logger object for the current module. __name__ is special python variable that holds name of current module
logger = logging.getLogger(__name__)



def usa_generation_data_function(usa_generation_data_file, hilarri_file, glohydrores_file, usa_final_output):

    """

    """

    if not os.path.exists(usa_generation_data_file):
        raise FileNotFoundError("File doest not exist. Download from Sean W. D. Turner (2022)")
        
    
    logger.info(f"Reading USA hydropower generation data {usa_generation_data_file}")
    usa_generation_data = pd.read_csv(usa_generation_data_file)
    eia_id_list = usa_generation_data.EIA_ID.unique().tolist()

    if not os.path.exists(hilarri_file):
        raise FileNotFoundError("HILARRI does not exist. Download HILARRI data conatining hydropower plants infromation for United States")

    # HILARRI data have eia_id and eha_id therefore can be used to link to the GloHydroRes using plant_source_id
    logger.info(f"Reading HILARRI hydropower dataset {hilarri_file}")

    HILARRI_data = gpd.read_file(hilarri_file)

    HILARRI_data["eia_ptid"] = HILARRI_data.eia_ptid.where(
        HILARRI_data.eia_ptit != "NA", None
    )

    HILARRI_data["eia_ptid"] = HILARRI_data["eia_ptid"].astype(float)

    HILARRI_data = HILARRI_data[HILARRI_data["eia_ptid"].isin(eia_id_list)]

    HILARRI_data = HILARRI_data.loc[
        HILARRI_data.eia_ptid.drop_duplicates(keep=False).index
    ]

    # EHA ID is the indentifier used in GloHydroRes as plant_source_id with plant_source as EHA

    usa_generation_data["eha_ptid"] = usa_generation_data.EIA_ID.map(
        dict(zip(HILARRI_data.eia_ptid, HILARRI_data.eha_ptid))
    )

    usa_generation_data["plant_source"] = "EHA"

    # After adding eha_ptid and plant_source, we will add GLoHydroRes main GloHydroRes unique ID
    # This is done by matching plant_source and plant_source_id in GloHydroRes data

    if not os.path.exists(glohydrores_file):
        raise FileNotFoundError("GloHydroRes file does not exist")

    logger.info(f"Reading GloHydrores hydropower dataset {glohydrores_file}")

    GloHydroRes = pd.read_excel(glohydrores_file, sheet_name="Data")

    glohydrores_plant_source_ID_plant_source_dict = {
        (row["plant_source"], row["plant_source_id"]) : row["ID"]
        for _, row in GloHydroRes.iterrows()
    }

    usa_generation_data["glohydrores_plant_id"] = usa_generation_data.apply(
        lambda row: glohydrores_plant_source_ID_plant_source_dict.get(
            (row["plant_source"], row["eha_ptid"])
        ),
        axis = 1
    )

    # Finally remove those plants which are not available in GloHydroRes
    usa_generation_data = usa_generation_data[
        ~usa_generation_data["glohydrores_plant_id"].isna()
    ]

    usa_generation_data["generation_unit"] = "MWH"
    usa_generation_data["country"] = "USA"

    month_dict = {
    "Jan": 1,
    "Feb": 2,
    "Mar": 3,
    "Apr": 4,
    "May": 5,
    "Jun": 6,
    "Jul": 7,
    "Aug": 8,
    "Sep": 9,
    "Oct": 10,
    "Nov": 11,
    "Dec": 12,
    }

    usa_generation_data["month"] = usa_generation_data.month.map(month_dict)
    usa_generation_data["date"] = pd.to_datetime(
        usa_generation_data[["year", "month"]].assign(DAY=1)
    )

    # Selecting recommended datasource for hydropwer generation data
    usa_generation_data["final_generation_MWh"] = np.where(
        usa_generation_data["recommended_data"] == "RectifHyd",
        usa_generation_data["RectifHyd_MWh"],
        usa_generation_data["EIA_MWh"]
        

    )

    usa_generation_data = usa_generation_data[
        [
            "plant",
            "country",
            "glohydrores_plant_id",
            "date",
            "final_generation_MWh",
            "generation_unit"
        ]
    ]

    usa_generation_data["generation_source"] = (
        "https://www.nature.com/articles/s41597-022-01748-x"
    )

    usa_generation_data.rename(
        columns={"plant": "plant_name", "final_generation_MWh": "generation_value"},
        inplace=True
    )

    usa_generation_data.to_csv(
        usa_final_output, index = False
    )

    logger.info(f"Saving final dataset {usa_final_output}")


def main(usa_generation_data_file, hilarri_file, glohydrores_file, usa_final_output):
    usa_generation_data_function(usa_generation_data_file, hilarri_file, glohydrores_file, usa_final_output)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        "Read USA hydropower generation data provided by Sean W. D. Turner (2022) Revised monthly energy generation estimates for 1,500 hydroelectric power plants in the United States"
    )

    parser.add_argument(
        "--usa_generation_data_dir",
        default="/scratch/shah0012/hybrid_hydropower_model/data/SWD_Turner_USA_monthly_hydropower_generation/RectifHyd_v1.3.csv",
        help = "Directory containing USA hydropower generation data"
    )

    parser.add_argument(
        "--hilarri_dir",
        default = "/scratch/shah0012/hybrid_hydropower_model/data/HILARRI_v1_1_0/HILARRI_v1_1/HILARRI_v1_1_Shapefile/HILARRI_v1_1_Public.shp",
        help = "Directory containing HILARRI data. Must be in shape file format"
    )

    parser.add_argument(
        "--glohydrores_file",
        default = "/home/shah0012/GloHydroRes/Output_data/GloHydroRes_vs2.xlsx",
        help = "Directory containing GlHydroRes file."

    )

    parser.add_argument(
        "--usa_final_output",
        default = "/scratch/shah0012/hybrid_hydropower_model/data/SWD_Turner_USA_monthly_hydropower_generation//RectifHyd_v1.3_with_GloHydroRes_plant_ID_run_till_2022.csv",
        help = "Final output file"
    )

    args = parser.parse_args()

    main(
        usa_generation_data_file = args.usa_generation_data_dir,
        hilarri_file  = args.hilarri_dir,
        glohydrores_file = args.glohydrores_file,
        usa_final_output = args.usa_final_output

    )



