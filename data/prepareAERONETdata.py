# #################################################################################
# MODIS aerosol optical depth spatial downscaling and post-process correction
# for the DRAGON campaign 2011
# ---
#
# Script to prepare AERONET data
#
# CODE VERSION 24 May 2022.
#
#   * Developed by: Finnish Meteorological Institute and University of Eastern Finland
#   * Contact info: Antti Lipponen (antti.lipponen@fmi.fi)
#
# #################################################################################

import numpy as np
import pandas as pd
from glob import glob
import os
from datetime import datetime
from netCDF4 import Dataset
from astropy.convolution import Gaussian2DKernel
from astropy.convolution import convolve


def dist(lat1, lon1, lat2, lon2):
    lat1R, lat2R, lon1R, lon2R = np.deg2rad(lat1), np.deg2rad(lat2), np.deg2rad(lon1), np.deg2rad(lon2)
    dlon = lon2R - lon1R
    dlat = lat2R - lat1R
    R = 6378.1
    a = (np.sin(dlat / 2.0))**2 + np.cos(lat1R) * np.cos(lat2R) * (np.sin(dlon / 2.0))**2
    c = 2.0 * np.arctan2(np.sqrt(a), np.sqrt(1.0 - a))
    d = R * c
    return d


inputpath = 'preparedData'

# Load AERONET data
AERONETstations = pd.read_csv(os.path.join('AERONET', 'dragon_2011_locations_2011_lev20.txt'), skiprows=1).set_index('Site_Name')
AERONETdata = {}
for station in AERONETstations.index:
    AERONETdata[station] = pd.read_csv(os.path.join('AERONET', 'AOT', 'LEV20', 'ALL_POINTS', '110401_111001_{sitename}.lev20'.format(sitename=station)), skiprows=4, parse_dates={'datetime': ['Date(dd-mm-yy)', 'Time(hh:mm:ss)']}, date_parser=lambda x: datetime.strptime(x, '%d:%m:%Y %H:%M:%S'))
    AERONETdata[station]['lon'] = AERONETstations.loc[station]['Longitude(decimal_degrees)']
    AERONETdata[station]['lat'] = AERONETstations.loc[station]['Latitude(decimal_degrees)']
    AERONETdata[station]['station'] = station
    AERONETdata[station]['timestamp'] = (AERONETdata[station]['datetime'] - datetime(1993, 1, 1)).astype('timedelta64[s]')
AERONETdata = pd.concat([AERONETdata[k] for k in AERONETstations.index], axis=0, ignore_index=True)
AERONETdata = AERONETdata.add_prefix('AERONET_')

# elevation data
nc = Dataset(os.path.join('GMTED2010', 'GMTED2010.nc'))
GMTED2010 = np.array(nc['Band1'][:]).astype(float)[::-1, :]
nc.close()

overpassID = 0

MODISdf = None
AERONETdf = None


files = glob(os.path.join(inputpath, 'MODIS_DRAGON2011_*.nc'))
for file in files:
    print(file)
    nc = Dataset(file)

    variables = ['lat', 'lon', 'Scan_Start_Time', 'MOD04_3K_Latitude', 'MOD04_3K_Longitude', 'MOD04_3K_Solar_Zenith', 'MOD04_3K_Solar_Azimuth', 'MOD04_3K_Sensor_Zenith', 'MOD04_3K_Sensor_Azimuth', 'MOD04_3K_Scattering_Angle', 'MOD04_3K_Glint_Angle', 'MOD04_3K_Land_Ocean_Quality_Flag', 'MOD04_3K_Land_sea_Flag', 'MOD04_3K_Aerosol_Type_Land', 'MOD04_3K_Fitting_Error_Land', 'MOD04_3K_Surface_Reflectance_Land_0', 'MOD04_3K_Surface_Reflectance_Land_1', 'MOD04_3K_Surface_Reflectance_Land_2', 'MOD04_3K_Corrected_Optical_Depth_Land_0', 'MOD04_3K_Corrected_Optical_Depth_Land_1', 'MOD04_3K_Corrected_Optical_Depth_Land_2', 'MOD04_3K_Corrected_Optical_Depth_Land_wav2p1', 'MOD04_3K_Optical_Depth_Ratio_Small_Land', 'MOD04_3K_Number_Pixels_Used_Land_0', 'MOD04_3K_Number_Pixels_Used_Land_1', 'MOD04_3K_Mean_Reflectance_Land_0', 'MOD04_3K_Mean_Reflectance_Land_1', 'MOD04_3K_Mean_Reflectance_Land_2', 'MOD04_3K_Mean_Reflectance_Land_3', 'MOD04_3K_Mean_Reflectance_Land_4', 'MOD04_3K_Mean_Reflectance_Land_5', 'MOD04_3K_Mean_Reflectance_Land_6', 'MOD04_3K_STD_Reflectance_Land_0', 'MOD04_3K_STD_Reflectance_Land_1', 'MOD04_3K_STD_Reflectance_Land_2', 'MOD04_3K_STD_Reflectance_Land_3', 'MOD04_3K_STD_Reflectance_Land_4', 'MOD04_3K_STD_Reflectance_Land_5', 'MOD04_3K_STD_Reflectance_Land_6', 'MOD04_3K_Mass_Concentration_Land', 'MOD04_3K_Aerosol_Cloud_Fraction_Land', 'MOD04_3K_Quality_Assurance_Land_0', 'MOD04_3K_Quality_Assurance_Land_1', 'MOD04_3K_Quality_Assurance_Land_2', 'MOD04_3K_Quality_Assurance_Land_3', 'MOD04_3K_Quality_Assurance_Land_4', 'MOD04_3K_Topographic_Altitude_Land', 'MOD04_3K_BowTie_Flag', 'TOAreflectance_band_01', 'TOAreflectance_band_02', 'TOAreflectance_band_03', 'TOAreflectance_band_04', 'TOAreflectance_band_05', 'TOAreflectance_band_06', 'TOAreflectance_band_07', 'TOAreflectance_band_08', 'TOAreflectance_band_09', 'TOAreflectance_band_10', 'TOAreflectance_band_11', 'TOAreflectance_band_12', 'TOAreflectance_band_13', 'TOAreflectance_band_14', 'TOAreflectance_band_15', 'TOAreflectance_band_16', 'TOAreflectance_band_17', 'TOAreflectance_band_18', 'TOAreflectance_band_19', 'TOAreflectance_band_20', 'TOAreflectance_band_21', 'TOAreflectance_band_22']
    M = ~np.isnan(nc['MOD04_3K_Corrected_Optical_Depth_Land_1'][:])
    if M.sum() == 0:
        nc.close()
        continue

    this_MODIS = {}
    for var in variables:
        this_MODIS[var] = nc[var][:][M]
    this_MODIS['GMTED2010'] = GMTED2010[M]

    smoothvars = ['MOD04_3K_Surface_Reflectance_Land_0', 'MOD04_3K_Surface_Reflectance_Land_1', 'MOD04_3K_Surface_Reflectance_Land_2', 'MOD04_3K_Corrected_Optical_Depth_Land_0', 'MOD04_3K_Corrected_Optical_Depth_Land_1', 'MOD04_3K_Corrected_Optical_Depth_Land_2', 'MOD04_3K_Corrected_Optical_Depth_Land_wav2p1']
    kernel = Gaussian2DKernel(x_stddev=12, y_stddev=12, x_size=4 * 12 + 1, y_size=4 * 12 + 1)
    for smoothvar in smoothvars:
        vardata = np.array(nc[smoothvar]).astype(float)
        nanMASK = np.isnan(vardata)
        this_MODIS[smoothvar + '_smooth'] = convolve(vardata, kernel, boundary='extend', preserve_nan=True)[M]
    nc.close()
    this_MODIS = pd.DataFrame(this_MODIS).add_prefix('MODIS_')
    granule_meantime = np.mean(this_MODIS['MODIS_Scan_Start_Time'])
    AERONET_timeMASK = np.abs(AERONETdata['AERONET_timestamp'] - granule_meantime) < 900

    this_AERONET = AERONETdata[AERONET_timeMASK]

    for ii, row in this_AERONET.iterrows():
        MODISdistance = dist(row['AERONET_lat'], row['AERONET_lon'], this_MODIS['MODIS_lat'], this_MODIS['MODIS_lon'])
        distMASK = MODISdistance < 5
        if distMASK.sum() == 0:
            continue
        MODISrows = pd.DataFrame(this_MODIS[distMASK])
        MODISrows['overpassID'] = overpassID
        AERONETrows = pd.DataFrame(row.to_frame().T)
        AERONETrows['overpassID'] = overpassID
        overpassID += 1

        if MODISdf is None:
            MODISdf = pd.DataFrame(MODISrows).reset_index()
        else:
            MODISdf = pd.concat((MODISdf, MODISrows), ignore_index=True)
        if AERONETdf is None:
            AERONETdf = pd.DataFrame(AERONETrows).reset_index()
        else:
            AERONETdf = pd.concat((AERONETdf, AERONETrows), ignore_index=True)

AERONETdf.to_hdf('DRAGON2011_AERONETdf.h5', key='df', mode='w')
MODISdf.to_hdf('DRAGON2011_MODISdf.h5', key='df', mode='w')
