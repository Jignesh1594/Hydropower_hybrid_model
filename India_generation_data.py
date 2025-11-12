import tabula
import pandas as pd
import numpy as np
import glob
import os
import datetime
import argparse
import logging

"""
Code here will going to cleans and harmonizes hydropower electricity generation data for India, which sourced from three entities:
1)  Uttarakhand Jal Vidyut Nigam Limited (UJVNL), [monthly files]
2) Himachal Pradesh Area Load Despatch Centre (HPALDC) [daily files but data is in 15 min blocks]
3) Central Electric Authority (CEA).  



UJVNL provide data is Million Unit (MU) which is equal to GWH
HPALDC provide data in GWH



Note: HPALDC provided some daily data in pdf format and some in excel. All pdf files were converted to single excel file. And excel files I dealt with normal running code
Note: India level generation data from April 2021 to December 2022 is available in excel format that is why created a function. However the data from 2016 to 2020 come in pdf format therefore manually
created file which contains all the data.

"""

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)

## These are plants for which are avilable in GloHydroRes and with theirs GloHydroRes ID. 
avialable_ujvnl_plants_name = ["CHIBRO", "KHODRI", "DHAKRANI", "DHALIPUR", 'TILOTH_(MB-I)', 'DHARASU_(MB-II)', "CHILLA", "VYASI", "RAMGANGA"]
plant_id = ["GHR03410", "GHR03458", "GHR03415", "GHR03416", "GHR03496", "GHR03495", "GHR03411",  "GHR03610", "GHR03543"]

def uttarakhand_read_pdf_data_func(uttarakhand_pdf_file_path, output_file_path):
    
    """
    Extract data from Uttarkhand UJVNL PDF files and convert to CSV.

    Args:
        uttarakhand_pdf_file_path: Directory containing PDF files
        output_file_path: Directory where CSV files should saved

    """
    logger.info(f"Reading PDF files from: {uttarakhand_pdf_file_path}")

    pdf_file_names = sorted(glob.glob(uttarakhand_pdf_file_path + "*.pdf"))

    if not pdf_file_names:
        logger.warning(f"No PDF files found in {uttarakhand_pdf_file_path}")
        return
    
    for name in pdf_file_names:
        df = tabula.read_pdf(name, pandas_options={"header" :  None})
        df = df[1].replace("\r", "_", regex=True)
        df = df.replace("\n", "_", regex=True)
        end_idx = df[df[0] == "TOTAL"].index[0]
        final_df = df[2:end_idx]
        file_name = name.split(".")[0]
        file_name = file_name  + "*.csv"
        output_path = os.path.join(output_file_path, file_name)
        final_df[[1, 8]].to_csv(output_path)
        logger.info(f"Processed: {name} -> {output_path}")


def uttarakhand_aggregated_actual_gen_data(uttarakhand_csv_file_path, output_file):

    """
    Aggregate all monthly data from Uttarakhand into single Excel file.

    Args:
        uttarakhand_csv_file_path : Directory containing CSV files
        output_file : Output Excel file path

    """

    logger.info(f"Aggregating CSV files from : {uttarakhand_csv_file_path}")


    csv_files = sorted(glob.glob(uttarakhand_csv_file_path + "*csv"))
    
    if not csv_files:
        logger.info(f"No CSV files found in {uttarakhand_csv_file_path}")
        return
    
    df_append = pd.DataFrame()
    for file in csv_files:
        df_temp = pd.read_csv(file)
        df_temp = df_temp.drop(columns="Unnamed: 0")
        n_rows = df_temp.shape[0]
        month = int(file.split("/")[-1].split(".")[0].split("_")[1])
        year = int(file.split("/")[-1].split(".")[0].split("_")[2])
        dt_df = pd.DataFrame({'year' : year,
              'month' : month}, index = [0])
        date = pd.to_datetime(dt_df[['year', 'month']].assign(DAY=1), format="%d/%m/%Y").repeat(n_rows)
        df_temp["date"] = date.values
        df_append = df_append.append(df_temp, ignore_index=True)

    df_append = df_append.rename(columns={"1": "plant_name", "8": "actual_generation", "date": "date"})
    df_append["country"] = pd.Series("India").repeat(df_append.shape[0]).values
    #df_append["plant_source"] = pd.Series("GloHydroRes").repeat(df_append.shape[0]).values
    df_append["generation_source"] = pd.Series("https://www.ujvnl.com/").repeat(df_append.shape[0]).values
    df_append["country"] = pd.Series("India").repeat(df_append.shape[0]).values
    df_append["generation_unit"] = pd.Series("GWH").repeat(df_append.shape[0]).values
    plant_name_id_dict =  {avialable_ujvnl_plants_name[i]: plant_id[i] for i in range(len(avialable_ujvnl_plants_name))}
    df_append["glohydrores_plant_id"] =  df_append.plant_name.map(plant_name_id_dict)
    #df_append["lat"] = df_append.plant_id.map(plant_id_lat_dict)
    #df_append["lon"] = df_append.plant_id.map(plant_id_lon_dict)
    dff = df_append[~pd.isna(df_append.glohydrores_plant_id)]
    dff = dff.sort_values(['plant_name', 'date'])
    dff.rename(columns = {"actual_generation": "generation_value"}, inplace = True)
    dff.to_excel(output_file, index = False)
    logger.info(f"Saved aggregated data to : {output_file}")


## Import excel file of Himachal Pradesh which was created after converting pdf to excel manually.

def himachal_handle_pdf_converted_excel_file(file_path):

    """

    Clean Excel file converted from PDF for Himachal Pradesh data.
    
    Args:
        file_path : Path to the PDF-converted Excel file

    Returns:
        tuple : (generation_dict, date_list, not_available_date_list)
    """

    logger.info(f"Processing Himachal PDF-converted Excel: {file_path}")
    pdf_converted_excel_data = pd.read_excel(file_path)
    
    for series_name, series in pdf_converted_excel_data.items():
        series[pd.isna(series)] = '0' 
        series = series.str.lower()
        series = series.str.replace("-", "")
        pdf_converted_excel_data[series_name] = series
        
    # as all files contain somewhat different power plant names making it all the same.
    plant_name_list = ['baspa', 'larji', 'bhaba', 'bassi', 'giri', 'kashang', 'malana']

    for series_name, series in pdf_converted_excel_data.items():
        series[series.str.contains('total')] = 'total'
        for name in plant_name_list:
            series[series.str.contains(name)] = name
        pdf_converted_excel_data[series_name] = series

    plant_name_list = ['baspa', 'larji', 'bhaba', 'bassi', 'giri', 'kashang', 'malana']
    generation_dict = {"baspa":[],"larji":[],"bhaba":[], 'bassi': [], 'giri': [], 'kashang': [], 'malana': []}

    date_list = []
    not_available_date_list = []

    unique_source = pd.unique(pdf_converted_excel_data["Source.Name"]).tolist() # unique ids for 

    for s in unique_source:
        
        date = pd.to_datetime(s.split(".")[0], format="%d%m%Y")

        # below will get minimum and maximum index of each day
        min_idx = pdf_converted_excel_data["Source.Name"][pdf_converted_excel_data["Source.Name"] == s].index.min()
        max_idx = pdf_converted_excel_data["Source.Name"][pdf_converted_excel_data["Source.Name"] == s].index.max()
        day_df = pdf_converted_excel_data.loc[min_idx:max_idx, :]
        ending_row_indices = day_df["Column2"][day_df["Column2"] == "12:00"].index

        if ending_row_indices.empty:
            not_available_date_list.append(date)
            next
        else:
            df = day_df.loc[:ending_row_indices[0], :]
            date_list.append(date)
            # looping through each plant name for each day
            for name in plant_name_list:
                index = (day_df == name).any()
                print(name)
                if index.any():
                    one_plant_df = df.loc[:, index]
                    first_half_day = one_plant_df.iloc[:, 0]
                    first_half_day[pd.isna(first_half_day)] = '0'
                    first_half_day_generation = first_half_day[first_half_day.str.isdigit()].astype(float).sum()
                    second_half_day = one_plant_df.iloc[:, 1]
                    second_half_day[pd.isna(second_half_day)] = '0'
                    second_half_day_generation = second_half_day[second_half_day.str.isdigit()].astype(float).sum()
                
                # below will remove unwanted rows from dataframe 
                    generation_each_day = (first_half_day_generation + second_half_day_generation)/400
                    generation_dict[name].append(generation_each_day)
                    
                else:
                    generation_dict[name].append(0)
    
    return generation_dict, date_list, not_available_date_list



def himachal_clean_excel_file(himachal_file_path, file_extension):

    """

    Clean Excel files for Himachal Pradesh data.

    Args:
        file_extension: File extension to process ('xlsx' and 'xls')
        himachal_file_path: Directory containing Excel files


    Returns:
        tuple: (generation_dict, date_list, not_available_date_list)    
    """
    logger.info(f"Processing Himachal {file_extension} files from: {himachal_file_path}")


    files = sorted(entry.path for entry in os.scandir(himachal_file_path) if entry.is_file() and entry.path.endswith('.' + file_extension))
    
    generation_dict = {"baspa":[],"larji":[],"bhaba":[], 'bassi': [], 'giri': [], 'kashang': [], 'malana': []}
    date_list = []
    not_available_date_list = []

    for f in files:
        date = pd.to_datetime(f.split("/")[-1].split('.')[0], format="%d%m%Y")
        day_df = pd.read_excel(f)

        # taking object only because column with generation data also contains plant name therefore dtype should be object
        index = day_df.dtypes == "object"
        day_df = day_df.loc[:, index]

        # Change each column values to string. Some column contains both float and string values so hard to excute next steps of string lower and replace
        day_df = day_df.astype(str)
        day_df.replace('nan', '0', inplace = True)

        day_df = day_df.apply(lambda x: x.str.lower().str.replace("-", ""))

        plant_name_list = ['baspa', 'larji', 'bhaba', 'bassi', 'giri', 'kashang', 'malana']

        for series_name, series in day_df.items():
            series[series.str.contains("total")] = "total"
            for name in plant_name_list:
                series[series.str.contains(name)] = name
                day_df[series_name] = series

        if (day_df.columns == "Unnamed: 1").any():
            ending_row_indices = day_df["Unnnamed: 1"][day_df["Unnamed: 1"] == "12:00:00"].index

            if ending_row_indices.empty:
                not_available_date_list.append(date)
                continue

            else:
                df = day_df.loc[:ending_row_indices[0], :]
                date_list.append(date)

                for name in plant_name_list:
                    if (day_df == name).any().any():
                        one_plant_df = df.loc[:, (day_df == name).any()]
                        first_half_day = one_plant_df.iloc[:, 0]
                        first_half_day_generation = first_half_day[first_half_day.str.isdigit()].astype(float).sum()
                        second_half_day = one_plant_df.iloc[:, 1]
                        second_half_day_generation = second_half_day[second_half_day.str.isdigit()].astype(float).sum()
                
                        # below will remove unwanted rows from dataframe 
                        generation_each_day = (first_half_day_generation + second_half_day_generation)/400
                        generation_dict[name].append(generation_each_day)
                    
                    else:
                        generation_dict[name].append(0)
        else:
            not_available_date_list.append(date)

    return generation_dict, date_list, not_available_date_list




def extract_india_generation_data(india_2021_april_file_path, india_2021_2022_file_path):
    
    """
    Clean India generation data from April 2021 to December 2022

    Args:
        india_2021_april_file_path: Path to April 2021 data file
        india_2021_2022_file_path: Directory containing 2021-2022 files
    
    Returns:
        DataFrame: Assembled data


    """

    logger.info("Processing India 2021-2022 generation data")
    
    # file containing the data for April 2021 month generation data for whole India. Forgot the reason why I created this file but this file contains glohydrores unique ID
    assembled_data = pd.read_excel(india_2021_april_file_path)

    # Convert month name to month number
    month_dict = {
        'JAN': 1, 'FEB': 2, 'MAR': 3, 'APR': 4, 'MAY': 5, 'JUN': 6,
        'JUL': 7, 'AUG': 8, 'SEP': 9, 'OCT': 10, 'NOV': 11, 'DEC': 12
    }

     # Loop through years and months

    for year in range(2021, 2023):
        for month in ["JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"]:
            # Create column name for the generation values
            month_num = month_dict.get(month, None)
            date_obj = datetime.datetime(year, month_num, 1)
            formatted_date = date_obj.strftime('%d-%m-%Y')
            print(f"Processing {year}-{month} -> {formatted_date}")

            # create an empty DataFrame to store the data
            monthly_data = pd.DataFrame()

            # Process each unique code
            for unique_code in [8, 10, 16]:
                pattern = f'*{unique_code}_{year}-{month}*'
                matching_files = glob.glob(os.path.join(india_2021_2022_file_path, pattern))

                if not matching_files:
                    print(f"No files found for {year}-{month} with code {unique_code}")
                    continue # Skip to the next code if no files found
            
                # Read the appropriate file based on the unique code
                if unique_code in [8,10]:
                    print(f"File found: {matching_files}")
                    
                    try:
                        hydro_data = pd.read_excel(matching_files[0], header=12).iloc[:, [0, 1, 4]]
                    
                    except Exception as e:
                        print(f"Error reading file {matching_files[0]}: {e}")
                        continue
                else:
                    try:
                        specific_file = f"/scratch/shah0012/hybrid_hydropower_model/data/India_generation_data/April2021-2022/18 col act-16_{year}-{month}.xls"
                        if os.path.exists(specific_file):
                            hydro_data = pd.read_excel(specific_file, header=12).iloc[:, [0, 4, 8]]
                        else:
                            print(f"File not found: {specific_file}")
                            continue
                    except Exception as e:
                        print(f"Error reading file {specific_file}: {e}")
                        continue

                 # Standardize column names
                hydro_data.columns = ['plant_name', "installed_capacity_mw", formatted_date]
                hydro_data["generation_unit"] = "GWH"
                hydro_data["country"] = "India"

                # Append to the monthly data
                monthly_data = pd.concat([monthly_data, hydro_data], ignore_index=True)
           
            # If we have data for this month, update the assembled_data DataFrame
            if not monthly_data.empty:
                # Create the column if it doesn't exist
                if formatted_date not in assembled_data.columns:
                    assembled_data[formatted_date] = np.nan
                
                # Update values in assembled_data using the lookup logic
                assembled_data[formatted_date] = assembled_data.apply(
                    lambda row: monthly_data.loc[
                        (monthly_data.plant_name == row.plant_name) & 
                        (monthly_data.installed_capacity_mw == row.installed_capacity_mw), 
                        formatted_date].values[0] if not monthly_data.loc[
                            (monthly_data.plant_name == row.plant_name) & 
                            (monthly_data.installed_capacity_mw == row.installed_capacity_mw), 
                            formatted_date].empty else 0, axis=1)
    return assembled_data  







## Hydropower plant data available for Uttrakhand and their IDs in newly created hydropower database
def main(uttarakhand_pdf_file_path, 
        uttrankhand_pdf_output_path,
        uttarakhand_csv_file_path, 
        uttrakhand_csv_output_path, 
        himachal_pdf_converted_excel_file,
         himachal_pdf_cleaned_output_file,
         himachal_excel_file_path, 
         himachal_final_output_file, 
         india_2021_2022_input_file, 
         india_2021_2022_file_path,
         india_final_2021_2022_output_file, 
         india_final_2016_2020_file, 
         india_final_output):


    """
    Main function to process all India hydropower generation data

    """
    logger.info("=" * 80)
    logger.info("Starting India Hydropower Data Processing")
    logger.info("=" * 80)


    # Processing Uttarakhand data
    logger.info("\n--- Processing Uttarakhand Data ---")
    uttarakhand_read_pdf_data_func(uttarakhand_pdf_file_path , uttrankhand_pdf_output_path)
    uttarakhand_aggregated_actual_gen_data(uttarakhand_csv_file_path, uttrakhand_csv_output_path)

    Uttrakhand_generation_data = pd.read_excel(uttrakhand_csv_output_path)
    Uttrakhand_generation_data = Uttrakhand_generation_data[~Uttrakhand_generation_data["glohydrores_plant_id"].isna()]
    Uttrakhand_generation_data["date"] = pd.to_datetime(Uttrakhand_generation_data["date"], format='%Y-%m-%d')
    Uttrakhand_generation_data["date"] = Uttrakhand_generation_data["date"].dt.to_period("M").dt.to_timestamp() # Convert date to month end date


    # Process Himachal Pradesh data
    logger.info("\n--- Processing Himachal Pradesh Data ---")
    himachal_pdf_generation_data, himachal_pdf_generation_date, himachal_not_available_date_list = himachal_handle_pdf_converted_excel_file(himachal_pdf_converted_excel_file)
    himachal_pdf_df = pd.DataFrame(himachal_pdf_generation_data)
    himachal_pdf_df["time"] = himachal_pdf_generation_date
    himachal_pdf_df.to_excel(himachal_pdf_cleaned_output_file, index = False)

    himachal_excel_generation_data, himachal_excel_date, not_available_date_list = himachal_clean_excel_file(file_extension="xlsx", himachal_file_path=himachal_excel_file_path)
    himachal_xls_generation_data, himachal_xls_date, not_available_date_list = himachal_clean_excel_file(file_extension="xls", himachal_file_path=himachal_excel_file_path)

    # Combining all the data from pdf, excel and xls files
    himachal_excel_df = pd.DataFrame(himachal_excel_generation_data)
    himachal_excel_df["time"] = himachal_excel_date

    himachal_xls_df = pd.DataFrame(himachal_xls_generation_data)
    himachal_xls_df["time"] = himachal_xls_date

    # Combining excel and xls data with pdf data
    himachal_excel_xls_df = pd.concat([himachal_excel_df, himachal_xls_df])
    himachal_excel_xls_pdf_df = pd.concat([himachal_excel_xls_df, himachal_pdf_df])
    himachal_excel_xls_pdf_df = himachal_excel_xls_pdf_df.sort_values("time")

    himachal_excel_xls_pdf_df.set_index("time", inplace = True)
    himachal_excel_xls_pdf_df = himachal_excel_xls_pdf_df[himachal_excel_xls_pdf_df.index.year > 2004]

    
    # resampling daily data to monthly data by summing all the daily data. This will give total generation of the month.
    himachal_excel_xls_pdf_df = himachal_excel_xls_pdf_df.resample("M").sum()

    # Converting Lakh Units tp GWH
    himachal_excel_xls_pdf_df = himachal_excel_xls_pdf_df/(10)
    himachal_excel_xls_pdf_df = himachal_excel_xls_pdf_df.rest_index()
    himachal_excel_xls_pdf_df = himachal_excel_xls_pdf_df.melt('time', var_name='plant_name', value_name='generation_value')
    plant_name_plant_ID_dict = {'baspa': 'GHR03388', 'bassi': 'GHR03389', 'larji': 'GHR03480',  'kashang': "GHR03479", 'malana': 'GHR03493'}
    himachal_excel_xls_pdf_df["glohydrores_plant_id"] = himachal_excel_xls_pdf_df["plant_name"].map(plant_name_plant_ID_dict)
    himachal_excel_xls_pdf_df["generation_unit"] = 'GWH'
    himachal_excel_xls_pdf_df['generation_source'] = 'https://hpaldc.org/index.asp?pg=dgoph'
    himachal_excel_xls_pdf_df['country'] = 'India'
    himachal_excel_xls_pdf_df.rename(columns= {"time": "date"}, inplace=True)
    himachal_excel_xls_pdf_df.drop(columns=["Unnamed: 0"], inplace=True)

    # Final cleaned data 
    himachal_excel_xls_pdf_df.to_excel(himachal_final_output_file=himachal_final_output_file, index = False)

    himachal_excel_xls_pdf_df = himachal_excel_xls_pdf_df[~himachal_excel_xls_pdf_df["glohydrores_plant_id"].isna()]
    himachal_excel_xls_pdf_df["date"] = pd.to_datetime(himachal_excel_xls_pdf_df["date"], format='%Y-%m-%d')
    himachal_excel_xls_pdf_df["date"] = himachal_excel_xls_pdf_df["date"].dt.to_period("M").dt.to_timestamp() # Convert date to month end date


    # Process India-level data 2021-2022
    logger.info("\n--- Processing India 2021-2022 Data ---")
    April2021_2022_data = extract_india_generation_data(india_2021_2022_input_file, india_2021_2022_file_path)
    April2021_2022_data.to_excel(india_final_2021_2022_output_file, index=False)
    April2021_2022_data["generation_source"] = "https://npp.gov.in/"

    # Remove plants with zero installed capacity
    April2021_2022_data = April2021_2022_data[April2021_2022_data["installed_capacity_mw"] != 0]
    
    # Remove plants with nan glohydrores_plant_id
    April2021_2022_data = April2021_2022_data[~April2021_2022_data["glohydrores_plant_id"].isna()]

    April2021_2022_data["generation_value"] = April2021_2022_data["generation_value"].round(1)
    April2021_2022_data["date"] = pd.to_datetime(April2021_2022_data["date"], dayfirst=True)

    # There are some plants with data entries in both state and central data files so need to remove that. Here if plants have same ID, installed capacity, date and generation value then remove the duplicates
    # Note: This is not the best way to do this, but it works for now. 

    April2021_2022_data.drop_duplicates(subset=['glohydrores_plant_id', "installed_capacity_mw", 'date', 'generation_value'], inplace=True)

    April2021_2022_data = April2021_2022_data.groupby((["glohydrores_plant_id", "date", "generation_unit", "country", "generation_source"]))[["generation_value", "installed_capacity_mw"]].sum().reset_index()
    April2021_2022_data["date"] = April2021_2022_data["date"].dt.to_period("M").dt.to_timestamp() # Convert date to month end date

    #
    # Below file is manually created. Will not fine online
    logger.info("\n--- Processing India 2016-2020 Data ---")    
    Indialevel_generation_data_2016_2020 = pd.read_excel(india_final_2016_2020_file)
    Indialevel_generation_data_2016_2020 = Indialevel_generation_data_2016_2020[~Indialevel_generation_data_2016_2020["glohydrores_plant_id"].isna()]
    Indialevel_generation_data_2016_2020["date"] = pd.to_datetime(Indialevel_generation_data_2016_2020["date"], dayfirst=True)
    Indialevel_generation_data_2016_2020["generation_source"] = "https://cea.nic.in/?lang=en"
    Indialevel_generation_data_2016_2020 = Indialevel_generation_data_2016_2020.groupby((["glohydrores_plant_id", "date", "generation_unit", "country", "generation_source"]))[["generation_value", "installed_capacity_mw"]].sum().reset_index()

    Indialevel_generation_data_2016_2020["date"] = Indialevel_generation_data_2016_2020["date"].dt.to_period("M").dt.to_timestamp() # Convert date to month end date
    

    # Combine all data
    logger.info("\n--- Combining All Data Sources ---")
    India_final_generation_data = pd.concat([Indialevel_generation_data_2016_2020, April2021_2022_data, himachal_excel_xls_pdf_df, Uttrakhand_generation_data], ignore_index=True)

    # No plants should two generation values for the same date
    India_final_generation_data.drop_duplicates(subset=['glohydrores_plant_id',  'date'], inplace=True)

    # "/scratch/shah0012/hybrid_hydropower_model/data/India_generation_data/India_final_generation_data.xlsx"

    India_final_generation_data.to_excel(india_final_output, index=False)

    logger.info(f"\n Final output saved to: {india_final_output}")
    logger.info("=" * 80)
    logger.info("Processing Complete!")
    logger.info("=" * 80)






# Build instance of ArgumentParser
# -- to use before attributes of add_argument provide flexibility to use that attributes in any order when running the script. 
# If -- is not provided than the attributes must be provided in the same order as given below.

# attributes name in add_argument described below can differ from the main function which defined above

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        "Read India hydropower generation data from different sources, clean and harmonize it"
    )

    parser.add_argument(
        "--uttarakhand_pdf_dir",
        default="/home/shah0012/Hydropower_hybrid_model/data/India_generation_data/Uttrakhand_generation_data/Uttrakhand_generation_data/",
        help = "Directory containing Uttarakhand PDF files"
    )

    parser.add_argument(
        "--uttarakhand_pdf_output",
        default = "/home/shah0012/Hydropower_hybrid_model/data/India_generation_data/Uttrakhand_generation_data/Uttrakhand_generation_data/",
        help = "Directory to save Uttarakhand PDF-to-CSV converted files"
    )
    
    parser.add_argument(
        "--uttarakhand_csv_dir",
        default="/home/shah0012/Hydropower_hybrid_model/data/India_generation_data/Uttrakhand_generation_data/Uttrakhand_generation_data/",
        help = "Directory containing Uttarakhand CSV files"
    )

    parser.add_argument(
        "--uttarakhand_output",
        default="/home/shah0012/Hydropower_hybrid_model/data/India_generation_data/Uttrakhand_generation_data/Uttrakhand_actual_generation_data_2019_2023.xlsx",
        help = "Output file for aggregated Uttarakhand data (Excel format)"
    )

    parser.add_argument(
        "--himachal_pdf_excel",
        default="/scratch/shah0012/hybrid_hydropower_model/data/India_generation_data/Himachal_generation_data/aggregated_files/generation_data_pdf_converted.xlsx",
        help = "Himachal PDF-converted Excel file"
    )

    parser.add_argument(
        "--himachal_pdf_cleaned",
        default="/home/shah0012/Hydropower_hybrid_model/data/India_generation_data/Himachal_generation_data/aggregated_files/cleaned_pdf_generation_data.xlsx",
        help = "Output file for cleaned Himachal PDF data"
    )

    parser.add_argument(
        "--himachal_excel_dir",
        default="/home/shah0012/Hydropower_hybrid_model/data/India_generation_data/Himachal_generation_data/",
        help = "Directory containing Himachal Excel files"
    )

    parser.add_argument(
        "--himachal_output",
        default = "/home/shah0012/Hydropower_hybrid_model/data/India_generation_data/Himachal_generation_data/aggregated_files/Himachal_hydropower_generation_data_2010_2024_final_version.xlsx",
        help =  "Output file for final Himachal data (Excel files)"
    )

    parser.add_argument(
        "--india_2021_april_file",
        default = "/scratch/shah0012/hybrid_hydropower_model/data/India_generation_data/All_India_generation_data_2021April_22.xlsx",
        help  = "India generation data file for April 2021"
    )

    parser.add_argument(
        "--india_2021_2022_dir",
        default="/scratch/shah0012/hybrid_hydropower_model/data/India_generation_data/April2021-2022/",
        help="Directory containing India 2021-2022 generation files"
    )

    parser.add_argument(
        "--india_2021_2022_output",
        default = "/scratch/shah0012/hybrid_hydropower_model/data/India_generation_data/April2021-2022/All_India_generation_data_2021April_22_newversion.xlsx",
        help = "Ouput file for India 2021-2022 data (Excel format)"

    )

    parser.add_argument(
        "--india_2016_2020_file",
        default="/scratch/shah0012/hybrid_hydropower_model/data/India_generation_data/All_India_generation_data.xlsx",
        help ="India generation data file for 2016-2020 (manually created)"
    )

    parser.add_argument(
        "--india_final_output",
        default = "/scratch/shah0012/hybrid_hydropower_model/data/India_generation_data/India_final_generation_data.xlsx",
        help = "Final output file combining all India generation data sources (Excel format)"
    )

    args = parser.parse_args()

    # Validate that input directories/files exist

    main(
        uttarakhand_pdf_file_path = args.uttarakhand_pdf_dir,
        uttrankhand_pdf_output_path =  args.uttarakhand_pdf_output,
        uttarakhand_csv_file_path  = args.uttarakhand_csv_dir,
        uttrakhand_csv_output_path  = args.uttarakhand_output,
        himachal_pdf_converted_excel_file  = args.himachal_pdf_excel,
        himachal_pdf_cleaned_output_file = args.himachal_pdf_cleaned,
        himachal_excel_file_path  = args.himachal_excel_dir,
        himachal_final_output_file = args.himachal_output,
        india_2021_2022_input_file = args.india_2021_april_file,
        india_2021_2022_file_path  = args.india_2021_2022_dir,
        india_final_2021_2022_output_file  = args.india_2021_2022_output,
        india_final_2016_2020_file  = args.india_2016_2020_file,
        india_final_output  = args.india_final_output
         
    )
    







    











    
 

    
                        









    
    
  



