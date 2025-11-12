from ntpath import exists
import pandas as pd
import glob as glob
import logging
import argparse
import os

"""
ONS Brazil Enegry Data Processing 

This module processes yearly CSV files from ONS.
The data comes with Portuguese column names and hourly granularity, which are converted to English and monthly aggregates respectively.

Note: To convert hourly to monthly using mean as ONS data is in the unit of power (MW)

"""

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)


logger = logging.getLogger(__name__)

def brazil_ons_data(brazil_csv_path, brazil_output_file):
    
    """
    Read yearly csv files for Brazil which contains hourly hydropower generation data
    Convert hourly data to monthly resolution by taking mean. 

    Args:
        brazil_csv_path: Directory containing csv files
        brazil_output_file: Location where final csv files should be saved

    """

    logger.info(f"Reading CSV files from: {brazil_csv_path}")
    # read all csv files in the current directory
    brazil_ons_files = glob.glob(brazil_csv_path + "*.csv")

    if not brazil_ons_files:
        logger.warning(f"No CSV files found in {brazil_csv_path}")
        return

    #"C:/Users/shah0012/OneDrive - Universiteit Utrecht/Project/Data/Brazil_ONS_generation_data/*.csv"
    
    brazil_ons_data = pd.concat([pd.read_csv(f, on_bad_lines="skip", sep = ";", encoding="latin1") for f in brazil_ons_files], ignore_index=True)
    
    # change column names from Portuguese to English
    brazil_ons_data.rename(columns = {"din_instante" : "datetime",
                                  "id_subsistema" : "subsystem_id",
                                  "nom_subsistema" : "subsystem_name",
                                  "id_estado" : "state_id",
                                  "nom_estado" : "state_name",
                                  "cod_modalidadeoperacao" : "operation_mode_code",
                                  "nom_tipousina" : "plant_type_name",
                                  "nom_tipocombustivel" : "fuel_type_name",
                                  "nom_usina" : "plant_name",
                                  "val_geracao" : "generation_value"}, inplace = True)

    # only hydropower plants
    brazil_ons_data = brazil_ons_data[brazil_ons_data.fuel_type_name  == "HidrÃ¡ulica"]

    # Select relevant variables
    brazil_ons_data = brazil_ons_data[["datetime", "plant_name", "generation_value"]]

    # Resample the data from hourly to monthly
    brazil_ons_data["datetime"] = pd.to_datetime(brazil_ons_data["datetime"], format = "%Y-%m-%d")

    brazil_monthly_ons_data = brazil_ons_data.groupby("plant_name").resample("M", on = "datetime").mean().reset_index()

    # Export the results
    brazil_monthly_ons_data.to_csv(brazil_output_file, index = False)

    logger.info(f"Saved aggregated data to: {brazil_output_file}")



def main(brazil_csv_path,
        brazil_output_file, 
        brazil_ons_glohydrores_indicator_file,
        brazil_final_output):
        
        brazil_ons_data(brazil_csv_path, brazil_output_file)

        if not os.path.exists(brazil_ons_glohydrores_indicator_file):
            raise FileNotFoundError("This file must be created manually by mapping Brazil ONS plant name with plant available in GloHydroRes")

        else:
            logger.info("ONS GloHydroRes Mapping File")

            ONS_glohydrores_mapping_data = pd.read_excel(brazil_ons_glohydrores_indicator_file)
            ONS_glohydrores_mapping_data = ONS_glohydrores_mapping_data[~ONS_glohydrores_mapping_data.hydropower_plant_id.isna()]
            
            logger.info("Reading ONS monthly generation file")
            ONS_monthly_generation_data = pd.read_csv(brazil_output_file)

            ONS_monthly_generation_data["glohydrores_plant_id"] =  ONS_monthly_generation_data.plant_name.map(dict(zip(ONS_glohydrores_mapping_data.ons_plant_name, ONS_glohydrores_mapping_data.hydropower_plant_id)))
            ONS_monthly_generation_data = ONS_monthly_generation_data[~ONS_monthly_generation_data["glohydrores_plant_id"].isna()]
            ONS_monthly_generation_data["generation_unit"] = "MW"
            ONS_monthly_generation_data["country"] = "Brazil"
            ONS_monthly_generation_data["generation_source"] = "https://dados.ons.org.br/dataset/geracao-usina-2/resource/b56e215c-074e-418f-83c7-3d7a6d1292ff"
            ONS_monthly_generation_data.rename(columns={"datetime": "date"}, inplace=True)
            ONS_monthly_generation_data.to_excel(brazil_final_output, index=False)
            logging.info("Storing final output_file")



             







if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        "Read Brazil hydropower generation data provided by ONS and convert from hourly to monthly"
    )

    parser.add_argument(
        "--brazil_csv_dir",
        default="/scratch/shah0012/hybrid_hydropower_model/data/Brazil_ONS_generation_data/Brazil_hourly_input_data/",
        help = "Directory containing Brazil ONS CSV files"

    )

    parser.add_argument(
        "--brazil_monthly_output",
        default = "/scratch/shah0012/hybrid_hydropower_model/data/Brazil_ONS_generation_data/brazil_monthly_ons_data_2000_2021.csv",
        help = "Output file to store monthly generation data for Brazil (csv format)"
    )

    parser.add_argument(
        "--brazil_ons_glohydrores_indictor_dir",
        default = "/scratch/shah0012/hybrid_hydropower_model/data/Brazil_ONS_generation_data/ons_generation_data_plant_name.xlsx",
        help = "File which have Brazil plant name with glohydrores plant id (Excel format)"
    )

    parser.add_argument(
        "--brazil_final_output",
        default = "/scratch/shah0012/hybrid_hydropower_model/data/Brazil_ONS_generation_data/ons_monthly_generation_data_GloHydroRes_2000_2021.xlsx",
        help = "Final output file containing clean version of Brazil hydropower plants (Excel format)"
    )



    args = parser.parse_args()

    main(
        brazil_csv_path = args.brazil_csv_dir,
        brazil_output_file = args.brazil_monthly_output,
        brazil_ons_glohydrores_indicator_file = args.brazil_ons_glohydrores_indictor_dir,
        brazil_final_output = args.brazil_final_output
    )



