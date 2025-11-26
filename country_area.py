import warnings
warnings.filterwarnings("ignore")
import netCDF4 as nc
import rasterio
from pyproj import Geod 
from shapely.geometry import LineString, Point, Polygon
import numpy as np
import rioxarray


def reproject_raster(country_nc_file, final_output):
    
    ds = rioxarray.open_rasterio(country_nc_file)
    ds_resampled = ds.rio.reproject(ds.rio.crs, 
                                resolution=(0.1,0.1),
                                resampling=rasterio.enums.Resampling.nearest)
    ds_resampled.rio.to_raster(final_output)


def calc_country_area(country_nc, output_final_name):
    country_shape = nc.Dataset(country_nc)
    lat = country_shape.variables["lat"][:]
    lon = country_shape.variables["lon"][:]
    country_data = country_shape.variables["Band1"][:,:].data

    lon_spr = lon[1]- lon[0]
    lat_spr = lat[1]- lat[0]
    lon_spr = lon_spr/2
    lat_spr = lat_spr/2
    lat_spr = abs(lat_spr)

    nlon = country_data.shape[1]
    nlat = country_data.shape[0]

    geod = Geod(ellps="WGS84")

    Area_cell = np.zeros([nlat, nlon])

    for icol in range(nlon):
        for jrow in range(nlat):
            if country_data[jrow, icol] > 0:
                Area = geod.geometry_area_perimeter(Polygon(LineString([Point(lon[icol] - lon_spr, lat[jrow] - lat_spr), Point(lon[icol] + lon_spr, lat[jrow] - lat_spr), Point(lon[icol] + lon_spr, lat[jrow] + lat_spr), Point(lon[icol] - lon_spr, lat[jrow] + lat_spr)])))[0]/1000000
                Area_cell[jrow, icol] = Area

    #Writing netcdf file
    ds = nc.Dataset(output_final_name, "w", format = "NETCDF4")

    #Create dimensions - lon, lat
    lat_dim = ds.createDimension("lat", nlat)
    lon_dim = ds.createDimension("lon", nlon)

    #define variables
    lats = ds.createVariable(varname = "lat", datatype = "f4", dimensions = ("lat", ))
    lons = ds.createVariable(varname = "lon", datatype = "f4", dimensions = ("lon", ))
    area = ds.createVariable(varname = "area", datatype = "f4", dimensions = ("lat", "lon"))
    country_code = ds.createVariable(varname = "country_code", datatype = "f4", dimensions = ("lat", "lon"))
    area.units = "km2"
    area.long_name = "Area of grid cell in km2"
    country_code.units = "-"

    #Assing longitude and latitude values
    lats[:] = country_shape.variables["lat"][:]
    lons[:] = country_shape.variables["lon"][:]
    area[:,:] = Area_cell
    country_code[:,:] = country_data

    ds.close()

def main(country_nc_file, final_output, country_nc, output_final_name, resample = False):

    if resample:
        reproject_raster(country_nc_file, final_output)
        calc_country_area(country_nc, output_final_name)

    else:
        calc_country_area(country_nc, output_final_name)




if __name__ == "__main__":
    main(country_nc_file = "/home/shah0012/Hydropower_hybrid_model/data/country_shape_file/countries_5arcmin_rem_revLat.nc",
    final_output="/home/shah0012/Hydropower_hybrid_model/data/country_shape_file/countries_5arcmin_rem_revLat_resampled_0.1deg.nc",
    country_nc = "/home/shah0012/Hydropower_hybrid_model/data/country_shape_file/countries_5arcmin_rem_revLat.nc",
    output_final_name = "/home/shah0012/Hydropower_hybrid_model/data/country_shape_file/Global_grid_cell_area.nc",
    resample = False)

    main(country_nc_file = "/home/shah0012/Hydropower_hybrid_model/data/country_shape_file/countries_5arcmin_rem_revLat.nc",
    final_output="/home/shah0012/Hydropower_hybrid_model/data/country_shape_file/countries_5arcmin_rem_revLat_resampled_0.1deg.nc",
    country_nc = "/home/shah0012/Hydropower_hybrid_model/data/country_shape_file/countries_5arcmin_rem_revLat_resampled_0.1deg.nc",
    output_final_name = "/home/shah0012/Hydropower_hybrid_model/data/country_shape_file/Global_grid_cell_area_point1_degree_resampled.nc",
    resample = True)