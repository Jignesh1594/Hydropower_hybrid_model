import cdsapi
import os
import time
import argparse

def download_era5_land(start_year, end_year, output_dir):
    """
    Download ERA5-Land daily statistics data year by year

    Args:
        start_year (int): start year for download
        end_year (int): end year for download
        output_dir (str): Directory to save the downloaded data

    """

    # Create the output directory
    os.makedirs(output_dir, exist_ok=True)

    # Initialize the CDS API client
    client = cdsapi.Client(url="https://cds.climate.copernicus.eu/api", key="ff5b62cb-3579-4a1b-b49d-035c0301e7e8")

    # Base request template

    base_request = {
        "variable": ["2m_temperature"],
        
        "day": [
        "01", "02", "03",
        "04", "05", "06",
        "07", "08", "09",
        "10", "11", "12",
        "13", "14", "15",
        "16", "17", "18",
        "19", "20", "21",
        "22", "23", "24",
        "25", "26", "27",
        "28", "29", "30",
        "31"],

        "daily_statistic": "daily_mean",
        "time_zone": "utc+00:00",
        "frequency": "1_hourly",
        "data_format": "netcdf"
    }

    months = [f"{i:02d}" for i in range(1, 13)]

    # Download data year by year

    for year in range(start_year, end_year+1):
        print(f"Downloading data for year {year}")

        # Create year directory
        year_dir = os.path.join(output_dir, str(year))
        os.makedirs(year_dir, exist_ok=True)

        for month in months:
            print(f"Downloading data for year {year}, month {month}")

            # Create filename for this month
            output_file = os.path.join(
                year_dir, 
                f"ERA5_Land_daily_mean_temperature_{year}_{month}.nc"
            )
        
            # Skip if file already exists
            if os.path.exists(output_file):
                print(f"File {output_file} already exists, skipping...")
                continue

            # Update the request with the year
            current_request = base_request.copy()
            current_request["year"] = str(year)
            current_request["month"] = month

            # Try to download the retry mechanism
            max_retries = 3
            retry_delay = 60

            for attempt in range(max_retries):
                try:
                    print(f"Downloading data for year {year}--{month} (Attempt {attempt + 1}/{max_retries})")
                    client.retrieve(
                    "derived-era5-land-daily-statistics",
                    current_request,
                    output_file)
                    print(f"Successfully downloaded data for {year}--{month}")
                    break

                except Exception as e:
                    print(f"Error downloading data for {year}--{month}: {str(e)}")
                    if attempt < max_retries - 1:
                        print(f"Retrying in {retry_delay} seconds...")
                        time.sleep(retry_delay)
                    else:
                        print(f"Failed to download data for {year} after {max_retries} attempts")
                        # Optionally, log failed years to a file
                        with open(os.path.join(output_dir, "failed_downloads.txt"), "a") as f:
                            f.write(f"{year}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download ERA5-Land daily statistics data")
    parser.add_argument("start_year", type=int, help="Start year for download")
    parser.add_argument("end_year", type=int, help="End year for download")
    parser.add_argument("output_dir", type=str, help="Directory to save the downloaded data")

    args = parser.parse_args()

    # Download the data
    download_era5_land(args.start_year, args.end_year, args.output_dir)