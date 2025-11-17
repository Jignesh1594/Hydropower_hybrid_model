import pandas as pd
import glob
from fuzzywuzzy import fuzz
import copy
import os
import logging
import argparse

"""

This code will clean ENTSOE tranparency data.

ENTSOE provides hourly data for hydropower plants. However, it provides data for every generator in the plant. Therefore we can not
use this data directly for our analysis. Data needs to be converted from generator level to plant level and then aggregated to monthly level.

ENTSOE only provides the generation id, generation data, installed capacity for each generator. Therefore, we used 
JRC Open Power Plant Database (JRC-PPDB-OPEN) to match the generation id with the plant id and plant name.

The JRC-PPDB-OPEN is primarily based on a collection of all the information published by ENTSO-E on the European power plants at unit level.

JRC-PPDB-OPEN provides the following information:
eic_g: Generation unit EIC code
eic_p: Plant EIC code
name_p: Plant name
type_g: Generation unit type
capacity_p: Installed capacity of the plant unit
country: Country where the plant is located

GenerationUnitEIC in ENTSOE and eic_g in JRC-PPDB-OPEN are the same. Therefore, we used this column to match the 
generation id with the plant id and plant name.

After these steps, we will match the plant names with the plant names in GloHydroRes dataset. This will help select those plants 
which are common in both ENTSOE and GloHydroRes dataset. These plants will be used for our analysis

"""

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)


def europe_generation_data_preparation(europe_entsoe_path, entsoe_monthly_output_data, JRC_path,  glohydrores_path, compare_entsoe_JRC_hydropower_file, entsoe_final_output_file):
    
    """

    """
    
    # Read the ENTSOE csv files

    logger.info(f"Reading CSV files from : {europe_entsoe_path}")
    entsoe_csv_files = glob.glob(europe_entsoe_path + "*.csv") 
    entsoe_csv_files = sorted(entsoe_csv_files)

    if not entsoe_csv_files:
        raise FileNotFoundError("CSV files does not exist")

    entsoe_data = pd.DataFrame()

    plant_type = ['Hydro Pumped Storage', 'Hydro Run-of-river and poundage', 'Hydro Water Reservoir'] # Selecting only hydropower plants

    for file in entsoe_csv_files:
        data = pd.read_csv(file, sep='\t', header=0)
        data = data[data['ProductionType'].isin(plant_type)]
        data['DateTime'] = pd.to_datetime(data['DateTime'])
        data.set_index('DateTime', inplace=True)
        data = data.groupby(["PowerSystemResourceName", 'ProductionType', 'MapCode', "AreaName", "GenerationUnitEIC"])[["ActualGenerationOutput", "InstalledGenCapacity"]].resample('M').mean() # Resampling the data to monthly level
        data.reset_index(inplace = True)
        entsoe_data = pd.concat([entsoe_data, data])

    # Export monthly file -- This file does not contain any information regarding GLoHydroRes
    entsoe_data.to_excel(entsoe_monthly_output_data, index = False)

    ## JRC Open Power Plant Database (JRC-PPDB-OPEN) [https://data.jrc.ec.europa.eu/dataset/9810feeb-f062-49cd-8e76-8d8cfd488a05]
    plant_type = ['Hydro Pumped Storage', 'Hydro Run-of-river and poundage', 'Hydro Water Reservoir']

    logger.info(f"Reading JRC PPDB file from {JRC_path}")
    JRC_PPDB_OPEN = pd.read_csv(JRC_path)
    JRC_PPDB_OPEN = JRC_PPDB_OPEN[JRC_PPDB_OPEN.type_g.isin(plant_type)] ## Reservoir, Run-of-river and Pumped storage

    # Selecting only the hydropower plant data which are common in both ENTSOE and JRC-PPDB-OPEN
    entsoe_JRC_shared_data = entsoe_data[~entsoe_data.GenerationUnitEIC.isin(JRC_PPDB_OPEN.eic_g.tolist())]


    # Insert plant id (eic_p) and plant name (name_p) to entsoe_JRC_shared_data. This is was done to aggregate the data from generator level to plant level
    eic_generation_id_plant_id_dict = dict(zip(JRC_PPDB_OPEN.eic_g, JRC_PPDB_OPEN.eic_p))
    generation_id_plant_name_dict = dict(zip(JRC_PPDB_OPEN.eic_g, JRC_PPDB_OPEN.name_p))

    entsoe_JRC_shared_data["eic_p"] = entsoe_JRC_shared_data.GenerationUnitEIC.map(eic_generation_id_plant_id_dict) #eic_p: Plant EIC code. This is the plant id
    entsoe_JRC_shared_data["eic_plant_name"] = entsoe_JRC_shared_data.GenerationUnitEIC.map(generation_id_plant_name_dict) #eic_plant_name: Plant name

    # Aggregating the data to plant level
    entsoe_JRC_shared_data_agg_plant_level = entsoe_JRC_shared_data.groupby(["eic_p", "eic_plant_name", "DateTime"]).sum().reset_index()

    # Inserting the country name in the aggregated data. This was done for later manual matching.
    eic_plant_id_country_dict = dict(zip(JRC_PPDB_OPEN.eic_p, JRC_PPDB_OPEN.country))
    entsoe_JRC_shared_data_agg_plant_level["eic_country_name"] = entsoe_JRC_shared_data_agg_plant_level.eic_p.map(eic_plant_id_country_dict)

    # Matching the plant names with the plant names in GloHydroRes dataset

    logger.info(f"Reading GloHydrores file from {glohydrores_path}") 
    GloHydroRes = pd.read_excel(glohydrores_path, sheet_name = "Data")
    GloHydroRes_plant_name = GloHydroRes["name"].tolist() #list of plant names in GloHydroRes dataset

    # Unique plant names in JRC transparency data. THis will be match with all glohydrores plant names to get initial mapping
    entsoe_JRC_plant_names =  entsoe_JRC_shared_data_agg_plant_level.eic_plant_name.unique().tolist()

    # To match the plant names in JRC transparency data with the plant names in GloHydroRes dataset, we used Levenshtein distance
    GloHydroRes_entsoe_JRC_plant_name_match = [
        max([(fuzz.token_set_ratio(i, j), j) for j in GloHydroRes_plant_name])
        for i in entsoe_JRC_plant_names
        ]
    similarity_score, fuzzy_match = map(list, zip(*GloHydroRes_entsoe_JRC_plant_name_match))

    

    ENTSOE_JRC_GloHydroRes_plant_names_dict = {
    GloHydroRes_entsoe_JRC_plant_name_match[i]: fuzzy_match[i]
    for i in range(len(GloHydroRes_entsoe_JRC_plant_name_match))
    }

    # Here we are not going to match only plant name but also installed capacities and countries name also.
    entsoe_JRC_plantname_countryname_installedcap = entsoe_JRC_shared_data_agg_plant_level.drop_duplicates(subset = ["eic_plant_name"], keep = "first")[["eic_plant_name", "eic_country_name", "InstalledGenCapacity"]]
    ENTSOE_JRC_plant_name_country_name_dict = dict(zip(entsoe_JRC_plantname_countryname_installedcap.eic_plant_name, entsoe_JRC_plantname_countryname_installedcap.eic_country_name))
    ENTSOE_JRC_plant_name_installed_cap_dict = dict(zip(entsoe_JRC_plantname_countryname_installedcap.eic_plant_name, entsoe_JRC_plantname_countryname_installedcap.InstalledGenCapacity))


    # We are going to create a separate excel file using all the variables so that we can compare plant name, installed capacity, country name with GloHydroRes
    ENTSOE_JRC_GloHydroRes = pd.DataFrame(list(ENTSOE_JRC_GloHydroRes_plant_names_dict.items()), columns=['JRC_transparency_plant_name', 'GloHydroRes_plant_name'])
    ENTSOE_JRC_GloHydroRes["GloHydroRes_country_name"] = ENTSOE_JRC_GloHydroRes.GloHydroRes_plant_name.map(dict(zip(GloHydroRes.name, GloHydroRes.country)))
    ENTSOE_JRC_GloHydroRes["JRC_transparency_country_name"] = ENTSOE_JRC_GloHydroRes.JRC_transparency_plant_name.map(ENTSOE_JRC_plant_name_country_name_dict)
    ENTSOE_JRC_GloHydroRes["GloHydroRes_installed_cap"] = ENTSOE_JRC_GloHydroRes.GloHydroRes_plant_name.map(dict(zip(GloHydroRes.name, GloHydroRes.capacity_mw)))
    ENTSOE_JRC_GloHydroRes["JRC_transparency_installed_cap"] = ENTSOE_JRC_GloHydroRes.JRC_transparency_plant_name.map(ENTSOE_JRC_plant_name_installed_cap_dict)

    if not os.path.exists(compare_entsoe_JRC_hydropower_file):
        logger.info("Create a excel file to manually compare ENTSOE & JRC with GloHydroRes hydropower plants")
        ENTSOE_JRC_GloHydroRes.to_excel(compare_entsoe_JRC_hydropower_file)

        # These plants do not have a match in GloHydroRes dataset. Therefore, we will remove these plants from the final list. These information got from the manual comparison
        plants_donot_match = ["Vorarlberger Illwerke AG", "Schluchseewerk AG", "Schluchsee", "KLL - KW Linth-Limmern AG", "Blenio (OFIBLE)", "KW Oberhasli AG (KWO)", "Kraftwerke Mauvoisin AG", 
            "Emosson (ESA)", "AET Leventina",  "Electra-Massa (EM)", "Hongrin-Léman (FMHL)", "Kraftwerk Göschenen", "Kraftwerk CB-VE", "Gde-Dixence (GD)", "KSL - KW Sarganserland AG",
            "Maggia (OFIMA)",  "Kaprun-Hauptstufe", "SAINT-PIERRE", "DUERO G", "JUCAR", "SIL G", "TAJO G", "GNF MIÑO", "S.FIORANO_GENERAZIONE", "CEDEGOLO",  "EDESSAIOS", "HE_DUBROVNIK",
            "Solberg2"]

        # Read the final file after manual comparison and update the final list
        ENTSOE_JRC_GloHydroRes = pd.read_excel(compare_entsoe_JRC_hydropower_file)

        ENTSOE_JRC_GloHydroRes = ENTSOE_JRC_GloHydroRes[~ENTSOE_JRC_GloHydroRes.JRC_transparency_plant_name.isin(plants_donot_match)]

        # Some plant matched incorrectly. Therefore, we will manually correct these plants. In the we are adding GloHydroRes plant ID to the final dataframe which can connect this dataset with GloHydroRes dataset.
        glohydrores_name_country_ID_dict = {(row['name'], row['country']): row['ID'] for _, row in GloHydroRes.iterrows() }

        # Map the dictionary to the dataframe
        ENTSOE_JRC_GloHydroRes['glohydrores_plant_id'] = ENTSOE_JRC_GloHydroRes.apply(
                lambda row: glohydrores_name_country_ID_dict.get((row['plant_name'], row['GloHydroRes_country_name'])), axis=1)
            
        # Save the final file after all manual corrections
        ENTSOE_JRC_GloHydroRes.to_excel(compare_entsoe_JRC_hydropower_file)

            
    else:
        ENTSOE_JRC_GloHydroRes = pd.read_excel(compare_entsoe_JRC_hydropower_file)

    # From tranparency generation data, we will select only those plants which are common in both JRC transparency data and GloHydroRes dataset
    entsoe_JRC_shared_data_agg_plant_level = entsoe_JRC_shared_data_agg_plant_level[~entsoe_JRC_shared_data_agg_plant_level.eic_plant_name.isin(plants_donot_match)]
    entsoe_JRC_shared_data_agg_plant_level["plant_name"] = entsoe_JRC_shared_data_agg_plant_level.eic_plant_name.map(dict(zip(entsoe_JRC_shared_data_agg_plant_level.JRC_transparency_plant_name, entsoe_JRC_shared_data_agg_plant_level.GloHydroRes_plant_name)))
    entsoe_JRC_shared_data_agg_plant_level["country"] = entsoe_JRC_shared_data_agg_plant_level.eic_plant_name.map(dict(zip(entsoe_JRC_shared_data_agg_plant_level.JRC_transparency_plant_name, entsoe_JRC_shared_data_agg_plant_level.GloHydroRes_country_name)))
    entsoe_JRC_shared_data_agg_plant_level["glohydrores_plant_id"] = entsoe_JRC_shared_data_agg_plant_level.eic_plant_name.map(dict(zip(entsoe_JRC_shared_data_agg_plant_level.JRC_transparency_plant_name, entsoe_JRC_shared_data_agg_plant_level.GloHydroRes_plant_ID)))
    entsoe_JRC_shared_data_agg_plant_level.drop(columns=["Unnamed: 0"], inplace = True)

    # This was done because some generation units belongs to same plant but have different eic_plant_id therefore, we will group by GloHydroRes_plant_name, GloHydroRes_country_name and DateTime
    entsoe_JRC_shared_data_agg_plant_level = entsoe_JRC_shared_data_agg_plant_level.groupby(["glohydrores_plant_id", "DateTime"]).sum().reset_index()
    entsoe_JRC_shared_data_agg_plant_level["plant_name"] = entsoe_JRC_shared_data_agg_plant_level.glohydrores_plant_id.map(dict(zip(GloHydroRes.ID, GloHydroRes.name)))
    entsoe_JRC_shared_data_agg_plant_level["country"] = entsoe_JRC_shared_data_agg_plant_level.glohydrores_plant_id.map(dict(zip(GloHydroRes.ID, GloHydroRes.country)))

    # Final data preparation 
    entsoe_JRC_shared_data_agg_plant_level = entsoe_JRC_shared_data_agg_plant_level[["plant_name", "country", "glohydrores_plant_id", "DateTime", "ActualGenerationOutput", "InstalledGenCapacity"]]
    entsoe_JRC_shared_data_agg_plant_level.rename(columns = {"ActualGenerationOutput": "generation_value", "InstalledGenCapacity": "installed_capacity_mw", "DateTime" : "date"}, inplace = True)
    entsoe_JRC_shared_data_agg_plant_level["generation_unit"] = "MW"
    entsoe_JRC_shared_data_agg_plant_level["generation_source"] = "ENTSOE"
    entsoe_JRC_shared_data_agg_plant_level.drop(columns=["Unnamed: 0"], inplace=True)

    logger.info(f"Saving the final dataset to {entsoe_final_output_file}")


    # Save the haromnized ENTSOE transparency data
    entsoe_JRC_shared_data_agg_plant_level.to_excel(entsoe_final_output_file, index = False)




def main(europe_entsoe_path, entsoe_monthly_output_data, JRC_path, glohydrores_path, compare_entsoe_JRC_hydropower_file, entsoe_final_output_file):
    europe_generation_data_preparation(europe_entsoe_path, entsoe_monthly_output_data, JRC_path, glohydrores_path, compare_entsoe_JRC_hydropower_file, entsoe_final_output_file)


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(
        "Read ENTSOE hydropower generation data combined JRC data to merge with GloHydroRes"
    )

    parser.add_argument(
        "--europe_entsoe_dir",
        default="/home/shah0012/Hydropower_hybrid_model/data/Transparency_ENTSOE/",
        help="Directory containing CSV files from ENTSOE"
    )

    parser.add_argument(
        "--entsoe_monthly_output_dir",
        default="/home/shah0012/Hydropower_hybrid_model/data/Transparency_ENTSOE/ENTSOE_monthly_hydropower_generation_data_2015_2022.xlsx",
        help="Directory with path (excel format) to save monthly data of ENTSOE. This version is without merging with GloHydroRes"
    )

    parser.add_argument(
        "--JRC_path",
        default="/home/shah0012/Hydropower_hybrid_model/data/JRC_data/JRC-PPDB-OPEN.ver1.0/JRC_OPEN_UNITS.csv",
        help="Directory with file (csv format), which contains JRC data which has plant id and generator id as ENTSOE data does provide generator ID"
    )

    parser.add_argument(
        "--glohydrores_dir",
        default="/home/shah0012/GloHydroRes/Output_data/GloHydroRes_vs2.xlsx",
        help="Directory with file (excel format) which contain GloHydroRes file"

    )

    parser.add_argument(
        "--compare_entsoe_JRC_hydropower_file",
        default="/home/shah0012/Hydropower_hybrid_model/data/Transparency_ENTSOE/Compare_JRC_transparency_GloHydroRes_plant_names.xlsx",
        help="This file will created to manually compare if ENTSOE & JRC plants matches with GloHydroRes. Make changes to this file only if plants are incorrectly mapped as this file will finally again to going to read with correct mapping"
    )

    parser.add_argument(
        "--entsoe_final_output_dir",
        default="/home/shah0012/Hydropower_hybrid_model/data/Transparency_ENTSOE/ENTSOE_GloHydroRes_final_timeseries_generation_data_2015_2022.xlsx",
        help="Final output file (excel format) of Europe generation data"
    )

    args = parser.parse_args()

    main(
       europe_entsoe_path = args.europe_entsoe_dir,
       entsoe_monthly_output_data = args.entsoe_monthly_output_dir,
       JRC_path = args.JRC_path,
       glohydrores_path = args.glohydrores_dir,
       compare_entsoe_JRC_hydropower_file = args.compare_entsoe_JRC_hydropower_file,
       entsoe_final_output_file = args.entsoe_final_output_dir
    )








































