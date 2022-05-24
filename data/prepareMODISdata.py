# #################################################################################
# MODIS aerosol optical depth spatial downscaling and post-process correction
# for the DRAGON campaign 2011
# ---
#
# Script to prepare MODIS data
#
# CODE VERSION 24 May 2022.
#
#   * Developed by: Finnish Meteorological Institute and University of Eastern Finland
#   * Contact info: Antti Lipponen (antti.lipponen@fmi.fi)
#
# #################################################################################

import numpy as np
from glob import glob
import os
from datetime import datetime
from netCDF4 import Dataset
from pyresample import geometry, kd_tree
from geotiepoints import modis5kmto1km, modis1kmto500m, modis1kmto250m
from imageio import imwrite


outputpath = 'preparedData'
os.makedirs(outputpath, exist_ok=True)


# Define region of interest
extent = [315000, 4285000, 315000 + 120000, 4285000 + 120000]
width = (extent[2] - extent[0]) / 250
height = (extent[3] - extent[1]) / 250

area_def = geometry.AreaDefinition('DRAGON2011',
                                   'DRAGON campaign 2011 region of interest, Baltimore and Washington DC',
                                   'UTM18',
                                   '+proj=utm +zone=18 +datum=WGS84 +units=m +no_defs',
                                   width,
                                   height,
                                   extent)


# Load MODIS data
MODIS_aerosol_files = glob(os.path.join('MODIS', 'M?D04_3K.*.hdf'))
for MODISfile in MODIS_aerosol_files:
    print(os.path.split(MODISfile)[-1])
    # dimensions(sizes):
    # Cell_Along_Swath:mod04(680)
    # Cell_Across_Swath:mod04(451)
    # Solution_2_Land:mod04(3)
    # Solution_3_Land:mod04(3)
    # Solution_1_Land:mod04(2)
    # MODIS_Band_Land:mod04(7)
    # QA_Byte_Land:mod04(5)
    # Solution_Ocean:mod04(2)
    # MODIS_Band_Ocean:mod04(7)
    # Solution_Index:mod04(9)
    # QA_Byte_Ocean:mod04(5)
    #
    # variables(dimensions):
    # >f4 Longitude(Cell_Along_Swath:mod04, Cell_Across_Swath:mod04)
    # >f4 Latitude(Cell_Along_Swath:mod04, Cell_Across_Swath:mod04)
    # >f8 Scan_Start_Time(Cell_Along_Swath:mod04, Cell_Across_Swath:mod04)
    # >i2 Solar_Zenith(Cell_Along_Swath:mod04, Cell_Across_Swath:mod04)
    # >i2 Solar_Azimuth(Cell_Along_Swath:mod04, Cell_Across_Swath:mod04)
    # >i2 Sensor_Zenith(Cell_Along_Swath:mod04, Cell_Across_Swath:mod04)
    # >i2 Sensor_Azimuth(Cell_Along_Swath:mod04, Cell_Across_Swath:mod04)
    # >i2 Scattering_Angle(Cell_Along_Swath:mod04, Cell_Across_Swath:mod04)
    # >i2 Glint_Angle(Cell_Along_Swath:mod04, Cell_Across_Swath:mod04)
    # >i2 Land_Ocean_Quality_Flag(Cell_Along_Swath:mod04, Cell_Across_Swath:mod04)
    # >i2 Land_sea_Flag(Cell_Along_Swath:mod04, Cell_Across_Swath:mod04)
    # >i2 Wind_Speed_Ncep_Ocean(Cell_Along_Swath:mod04, Cell_Across_Swath:mod04)
    # >i2 Optical_Depth_Land_And_Ocean(Cell_Along_Swath:mod04, Cell_Across_Swath:mod04)
    # >i2 Image_Optical_Depth_Land_And_Ocean(Cell_Along_Swath:mod04, Cell_Across_Swath:mod04)
    # >i2 Aerosol_Type_Land(Cell_Along_Swath:mod04, Cell_Across_Swath:mod04)
    # >i2 Fitting_Error_Land(Cell_Along_Swath:mod04, Cell_Across_Swath:mod04)
    # >i2 Surface_Reflectance_Land(Solution_2_Land:mod04, Cell_Along_Swath:mod04, Cell_Across_Swath:mod04)
    # >i2 Corrected_Optical_Depth_Land(Solution_3_Land:mod04, Cell_Along_Swath:mod04, Cell_Across_Swath:mod04)
    # >i2 Corrected_Optical_Depth_Land_wav2p1(Cell_Along_Swath:mod04, Cell_Across_Swath:mod04)
    # >i2 Optical_Depth_Ratio_Small_Land(Cell_Along_Swath:mod04, Cell_Across_Swath:mod04)
    # >i2 Number_Pixels_Used_Land(Solution_1_Land:mod04, Cell_Along_Swath:mod04, Cell_Across_Swath:mod04)
    # >i2 Mean_Reflectance_Land(MODIS_Band_Land:mod04, Cell_Along_Swath:mod04, Cell_Across_Swath:mod04)
    # >i2 STD_Reflectance_Land(MODIS_Band_Land:mod04, Cell_Along_Swath:mod04, Cell_Across_Swath:mod04)
    # >f4 Mass_Concentration_Land(Cell_Along_Swath:mod04, Cell_Across_Swath:mod04)
    # >i2 Aerosol_Cloud_Fraction_Land(Cell_Along_Swath:mod04, Cell_Across_Swath:mod04)
    # int8 Quality_Assurance_Land(Cell_Along_Swath:mod04, Cell_Across_Swath:mod04, QA_Byte_Land:mod04)
    # >i2 Solution_Index_Ocean_Small(Solution_Ocean:mod04, Cell_Along_Swath:mod04, Cell_Across_Swath:mod04)
    # >i2 Solution_Index_Ocean_Large(Solution_Ocean:mod04, Cell_Along_Swath:mod04, Cell_Across_Swath:mod04)
    # >i2 Effective_Optical_Depth_Best_Ocean(MODIS_Band_Ocean:mod04, Cell_Along_Swath:mod04, Cell_Across_Swath:mod04)
    # >i2 Effective_Optical_Depth_Average_Ocean(MODIS_Band_Ocean:mod04, Cell_Along_Swath:mod04, Cell_Across_Swath:mod04)
    # >i2 Optical_Depth_Small_Best_Ocean(MODIS_Band_Ocean:mod04, Cell_Along_Swath:mod04, Cell_Across_Swath:mod04)
    # >i2 Optical_Depth_Small_Average_Ocean(MODIS_Band_Ocean:mod04, Cell_Along_Swath:mod04, Cell_Across_Swath:mod04)
    # >i2 Optical_Depth_Large_Best_Ocean(MODIS_Band_Ocean:mod04, Cell_Along_Swath:mod04, Cell_Across_Swath:mod04)
    # >i2 Optical_Depth_Large_Average_Ocean(MODIS_Band_Ocean:mod04, Cell_Along_Swath:mod04, Cell_Across_Swath:mod04)
    # >f4 Mass_Concentration_Ocean(Solution_Ocean:mod04, Cell_Along_Swath:mod04, Cell_Across_Swath:mod04)
    # >i2 Aerosol_Cloud_Fraction_Ocean(Cell_Along_Swath:mod04, Cell_Across_Swath:mod04)
    # >i2 Effective_Radius_Ocean(Solution_Ocean:mod04, Cell_Along_Swath:mod04, Cell_Across_Swath:mod04)
    # >f4 PSML003_Ocean(Solution_Ocean:mod04, Cell_Along_Swath:mod04, Cell_Across_Swath:mod04)
    # >i2 Asymmetry_Factor_Best_Ocean(MODIS_Band_Ocean:mod04, Cell_Along_Swath:mod04, Cell_Across_Swath:mod04)
    # >i2 Asymmetry_Factor_Average_Ocean(MODIS_Band_Ocean:mod04, Cell_Along_Swath:mod04, Cell_Across_Swath:mod04)
    # >i2 Backscattering_Ratio_Best_Ocean(MODIS_Band_Ocean:mod04, Cell_Along_Swath:mod04, Cell_Across_Swath:mod04)
    # >i2 Backscattering_Ratio_Average_Ocean(MODIS_Band_Ocean:mod04, Cell_Along_Swath:mod04, Cell_Across_Swath:mod04)
    # >i2 Angstrom_Exponent_1_Ocean(Cell_Along_Swath:mod04, Cell_Across_Swath:mod04)
    # >i2 Angstrom_Exponent_2_Ocean(Cell_Along_Swath:mod04, Cell_Across_Swath:mod04)
    # >i2 Least_Squares_Error_Ocean(Solution_Ocean:mod04, Cell_Along_Swath:mod04, Cell_Across_Swath:mod04)
    # >i2 Optical_Depth_Ratio_Small_Ocean_0.55micron(Cell_Along_Swath:mod04, Cell_Across_Swath:mod04)
    # >i2 Optical_Depth_by_models_ocean(Solution_Index:mod04, Cell_Along_Swath:mod04, Cell_Across_Swath:mod04)
    # >i2 Number_Pixels_Used_Ocean(Cell_Along_Swath:mod04, Cell_Across_Swath:mod04)
    # >i2 Mean_Reflectance_Ocean(MODIS_Band_Ocean:mod04, Cell_Along_Swath:mod04, Cell_Across_Swath:mod04)
    # >i2 STD_Reflectance_Ocean(MODIS_Band_Ocean:mod04, Cell_Along_Swath:mod04, Cell_Across_Swath:mod04)
    # int8 Quality_Assurance_Ocean(Cell_Along_Swath:mod04, Cell_Across_Swath:mod04, QA_Byte_Ocean:mod04)
    # >i2 Topographic_Altitude_Land(Cell_Along_Swath:mod04, Cell_Across_Swath:mod04)
    # >i2 BowTie_Flag(Cell_Along_Swath:mod04, Cell_Across_Swath:mod04)
    dataset = Dataset(MODISfile)
    MOD04_3K = {}
    MOD04_3K['Scan_Start_Time_units'] = dataset['Scan_Start_Time'].units
    for key in ['Scan_Start_Time', 'Latitude', 'Longitude', 'Solar_Zenith', 'Solar_Azimuth', 'Sensor_Zenith', 'Sensor_Azimuth', 'Scattering_Angle', 'Glint_Angle', 'Land_Ocean_Quality_Flag', 'Land_sea_Flag', 'Aerosol_Type_Land', 'Fitting_Error_Land', 'Surface_Reflectance_Land', 'Corrected_Optical_Depth_Land', 'Corrected_Optical_Depth_Land_wav2p1', 'Optical_Depth_Ratio_Small_Land', 'Number_Pixels_Used_Land', 'Mean_Reflectance_Land', 'STD_Reflectance_Land', 'Mass_Concentration_Land', 'Aerosol_Cloud_Fraction_Land', 'Quality_Assurance_Land', 'Topographic_Altitude_Land', 'BowTie_Flag']:
        valid_range = dataset[key].valid_range
        MOD04_3K[key] = np.array(dataset[key][:])
        MOD04_3K[key][(MOD04_3K[key] < valid_range[0]) | (MOD04_3K[key] > valid_range[1])] = np.nan
    dataset.close()

    MODISfilename = os.path.split(MODISfile)[-1]
    satellite = MODISfilename[:3]
    granuleID = MODISfilename[9:22]

    if os.path.isfile(os.path.join(outputpath, 'MODIS_DRAGON2011_{satellite}_{granuleID}.nc'.format(satellite=satellite, granuleID=granuleID))):
        continue

    MOD02QKMfile = glob(os.path.join('MODIS', '{satellite}02QKM.{granuleID}.061.*.hdf'.format(satellite=satellite, granuleID=granuleID)))
    MOD02HKMfile = glob(os.path.join('MODIS', '{satellite}02HKM.{granuleID}.061.*.hdf'.format(satellite=satellite, granuleID=granuleID)))
    MOD021KMfile = glob(os.path.join('MODIS', '{satellite}021KM.{granuleID}.061.*.hdf'.format(satellite=satellite, granuleID=granuleID)))
    if len(MOD02QKMfile) != 1 or len(MOD02HKMfile) != 1 or len(MOD021KMfile) != 1:
        print('==========================================================')
        print('!= 1 files found')
        print(MOD02QKMfile)
        print(MOD02HKMfile)
        print(MOD021KMfile)
        print('==========================================================')
        continue

    # dimensions(sizes):
    # 10*nscans:MODIS_SWATH_Type_L1B(2040)
    # Max_EV_frames:MODIS_SWATH_Type_L1B(1354)
    # Band_250M:MODIS_SWATH_Type_L1B(2)
    # 40*nscans:MODIS_SWATH_Type_L1B(8160)
    # 4*Max_EV_frames:MODIS_SWATH_Type_L1B(5416)
    # Band_250M(2)
    # number of emissive bands(16)
    # detectors per 1km band(10)
    # number of scans(204)
    # number of 250m bands(2)
    # detectors per 250m band(40)
    # number of 500m bands(5)
    # detectors per 500m band(20)
    # number of 1km reflective bands(15)
    # variables(dimensions):
    # >f4 Latitude(10*nscans:MODIS_SWATH_Type_L1B, Max_EV_frames:MODIS_SWATH_Type_L1B)
    # >f4 Longitude(10*nscans:MODIS_SWATH_Type_L1B, Max_EV_frames:MODIS_SWATH_Type_L1B)
    # >u2 EV_250_RefSB(Band_250M:MODIS_SWATH_Type_L1B, 40*nscans:MODIS_SWATH_Type_L1B, 4*Max_EV_frames:MODIS_SWATH_Type_L1B)
    # uint8 EV_250_RefSB_Uncert_Indexes(Band_250M:MODIS_SWATH_Type_L1B, 40*nscans:MODIS_SWATH_Type_L1B, 4*Max_EV_frames:MODIS_SWATH_Type_L1B)
    # >f4 Band_250M(Band_250M)
    # uint8 Noise in Thermal Detectors(number of emissive bands, detectors per 1km band)
    # uint8 Change in relative responses of thermal detectors(number of emissive bands, detectors per 1km band)
    # int8 DC Restore Change for Thermal Bands(number of scans, number of emissive bands, detectors per 1km band)
    # int8 DC Restore Change for Reflective 250m Bands(number of scans, number of 250m bands, detectors per 250m band)
    # int8 DC Restore Change for Reflective 500m Bands(number of scans, number of 500m bands, detectors per 500m band)
    # int8 DC Restore Change for Reflective 1km Bands(number of scans, number of 1km reflective bands, detectors per 1km band)
    dataset = Dataset(MOD02QKMfile[0])
    MOD02QKM = {}
    for key in ['Latitude', 'Longitude', 'EV_250_RefSB']:
        valid_range = dataset[key].valid_range
        MOD02QKM[key] = np.array(dataset[key][:]).astype(float)
        MOD02QKM[key][(MOD02QKM[key] < valid_range[0]) | (MOD02QKM[key] > valid_range[1])] = np.nan
    reflectance_scales = dataset['EV_250_RefSB'].reflectance_scales
    reflectance_offsets = dataset['EV_250_RefSB'].reflectance_offsets
    for ii in range(len(reflectance_scales)):
        MOD02QKM['EV_250_RefSB'][ii] = (MOD02QKM['EV_250_RefSB'][ii] - reflectance_offsets[ii]) * reflectance_scales[ii]
    dataset.close()
    MOD02QKM['Longitude'], MOD02QKM['Latitude'] = modis1kmto250m(MOD02QKM['Longitude'], MOD02QKM['Latitude'])

    # dimensions(sizes):
    # 10*nscans:MODIS_SWATH_Type_L1B(2040)
    # Max_EV_frames:MODIS_SWATH_Type_L1B(1354)
    # Band_500M:MODIS_SWATH_Type_L1B(5)
    # 20*nscans:MODIS_SWATH_Type_L1B(4080)
    # 2*Max_EV_frames:MODIS_SWATH_Type_L1B(2708)
    # Band_250M:MODIS_SWATH_Type_L1B(2)
    # Band_250M(2)
    # Band_500M(5)
    # number of emissive bands(16)
    # detectors per 1km band(10)
    # number of scans(204)
    # number of 250m bands(2)
    # detectors per 250m band(40)
    # number of 500m bands(5)
    # detectors per 500m band(20)
    # number of 1km reflective bands(15)
    # variables(dimensions):
    # >f4 Latitude(10*nscans:MODIS_SWATH_Type_L1B, Max_EV_frames:MODIS_SWATH_Type_L1B)
    # >f4 Longitude(10*nscans:MODIS_SWATH_Type_L1B, Max_EV_frames:MODIS_SWATH_Type_L1B)
    # >u2 EV_500_RefSB(Band_500M:MODIS_SWATH_Type_L1B, 20*nscans:MODIS_SWATH_Type_L1B, 2*Max_EV_frames:MODIS_SWATH_Type_L1B)
    # uint8 EV_500_RefSB_Uncert_Indexes(Band_500M:MODIS_SWATH_Type_L1B, 20*nscans:MODIS_SWATH_Type_L1B, 2*Max_EV_frames:MODIS_SWATH_Type_L1B)
    # >u2 EV_250_Aggr500_RefSB(Band_250M:MODIS_SWATH_Type_L1B, 20*nscans:MODIS_SWATH_Type_L1B, 2*Max_EV_frames:MODIS_SWATH_Type_L1B)
    # uint8 EV_250_Aggr500_RefSB_Uncert_Indexes(Band_250M:MODIS_SWATH_Type_L1B, 20*nscans:MODIS_SWATH_Type_L1B, 2*Max_EV_frames:MODIS_SWATH_Type_L1B)
    # int8 EV_250_Aggr500_RefSB_Samples_Used(Band_250M:MODIS_SWATH_Type_L1B, 20*nscans:MODIS_SWATH_Type_L1B, 2*Max_EV_frames:MODIS_SWATH_Type_L1B)
    # >f4 Band_250M(Band_250M)
    # >f4 Band_500M(Band_500M)
    # uint8 Noise in Thermal Detectors(number of emissive bands, detectors per 1km band)
    # uint8 Change in relative responses of thermal detectors(number of emissive bands, detectors per 1km band)
    # int8 DC Restore Change for Thermal Bands(number of scans, number of emissive bands, detectors per 1km band)
    # int8 DC Restore Change for Reflective 250m Bands(number of scans, number of 250m bands, detectors per 250m band)
    # int8 DC Restore Change for Reflective 500m Bands(number of scans, number of 500m bands, detectors per 500m band)
    # int8 DC Restore Change for Reflective 1km Bands(number of scans, number of 1km reflective bands, detectors per 1km band)
    dataset = Dataset(MOD02HKMfile[0])
    MOD02HKM = {}
    for key in ['Latitude', 'Longitude', 'EV_500_RefSB']:
        valid_range = dataset[key].valid_range
        MOD02HKM[key] = np.array(dataset[key][:]).astype(float)
        MOD02HKM[key][(MOD02HKM[key] < valid_range[0]) | (MOD02HKM[key] > valid_range[1])] = np.nan
    reflectance_scales = dataset['EV_500_RefSB'].reflectance_scales
    reflectance_offsets = dataset['EV_500_RefSB'].reflectance_offsets
    for ii in range(len(reflectance_scales)):
        MOD02HKM['EV_500_RefSB'][ii] = (MOD02HKM['EV_500_RefSB'][ii] - reflectance_offsets[ii]) * reflectance_scales[ii]
    dataset.close()
    MOD02HKM['Longitude'], MOD02HKM['Latitude'] = modis1kmto500m(MOD02HKM['Longitude'], MOD02HKM['Latitude'])

    # dimensions(sizes):
    # 2*nscans:MODIS_SWATH_Type_L1B(408)
    # 1KM_geo_dim:MODIS_SWATH_Type_L1B(271)
    # Band_1KM_RefSB:MODIS_SWATH_Type_L1B(15)
    # 10*nscans:MODIS_SWATH_Type_L1B(2040)
    # Max_EV_frames:MODIS_SWATH_Type_L1B(1354)
    # Band_1KM_Emissive:MODIS_SWATH_Type_L1B(16)
    # Band_250M:MODIS_SWATH_Type_L1B(2)
    # Band_500M:MODIS_SWATH_Type_L1B(5)
    # Band_250M(2)
    # Band_500M(5)
    # Band_1KM_RefSB(15)
    # Band_1KM_Emissive(16)
    # number of emissive bands(16)
    # detectors per 1km band(10)
    # number of scans(204)
    # number of 250m bands(2)
    # detectors per 250m band(40)
    # number of 500m bands(5)
    # detectors per 500m band(20)
    # number of 1km reflective bands(15)
    # variables(dimensions):
    # >f4 Latitude(2*nscans:MODIS_SWATH_Type_L1B, 1KM_geo_dim:MODIS_SWATH_Type_L1B)
    # >f4 Longitude(2*nscans:MODIS_SWATH_Type_L1B, 1KM_geo_dim:MODIS_SWATH_Type_L1B)
    # >u2 EV_1KM_RefSB(Band_1KM_RefSB:MODIS_SWATH_Type_L1B, 10*nscans:MODIS_SWATH_Type_L1B, Max_EV_frames:MODIS_SWATH_Type_L1B)
    # uint8 EV_1KM_RefSB_Uncert_Indexes(Band_1KM_RefSB:MODIS_SWATH_Type_L1B, 10*nscans:MODIS_SWATH_Type_L1B, Max_EV_frames:MODIS_SWATH_Type_L1B)
    # >u2 EV_1KM_Emissive(Band_1KM_Emissive:MODIS_SWATH_Type_L1B, 10*nscans:MODIS_SWATH_Type_L1B, Max_EV_frames:MODIS_SWATH_Type_L1B)
    # uint8 EV_1KM_Emissive_Uncert_Indexes(Band_1KM_Emissive:MODIS_SWATH_Type_L1B, 10*nscans:MODIS_SWATH_Type_L1B, Max_EV_frames:MODIS_SWATH_Type_L1B)
    # >u2 EV_250_Aggr1km_RefSB(Band_250M:MODIS_SWATH_Type_L1B, 10*nscans:MODIS_SWATH_Type_L1B, Max_EV_frames:MODIS_SWATH_Type_L1B)
    # uint8 EV_250_Aggr1km_RefSB_Uncert_Indexes(Band_250M:MODIS_SWATH_Type_L1B, 10*nscans:MODIS_SWATH_Type_L1B, Max_EV_frames:MODIS_SWATH_Type_L1B)
    # int8 EV_250_Aggr1km_RefSB_Samples_Used(Band_250M:MODIS_SWATH_Type_L1B, 10*nscans:MODIS_SWATH_Type_L1B, Max_EV_frames:MODIS_SWATH_Type_L1B)
    # >u2 EV_500_Aggr1km_RefSB(Band_500M:MODIS_SWATH_Type_L1B, 10*nscans:MODIS_SWATH_Type_L1B, Max_EV_frames:MODIS_SWATH_Type_L1B)
    # uint8 EV_500_Aggr1km_RefSB_Uncert_Indexes(Band_500M:MODIS_SWATH_Type_L1B, 10*nscans:MODIS_SWATH_Type_L1B, Max_EV_frames:MODIS_SWATH_Type_L1B)
    # int8 EV_500_Aggr1km_RefSB_Samples_Used(Band_500M:MODIS_SWATH_Type_L1B, 10*nscans:MODIS_SWATH_Type_L1B, Max_EV_frames:MODIS_SWATH_Type_L1B)
    # >i2 Height(2*nscans:MODIS_SWATH_Type_L1B, 1KM_geo_dim:MODIS_SWATH_Type_L1B)
    # >i2 SensorZenith(2*nscans:MODIS_SWATH_Type_L1B, 1KM_geo_dim:MODIS_SWATH_Type_L1B)
    # >i2 SensorAzimuth(2*nscans:MODIS_SWATH_Type_L1B, 1KM_geo_dim:MODIS_SWATH_Type_L1B)
    # >u2 Range(2*nscans:MODIS_SWATH_Type_L1B, 1KM_geo_dim:MODIS_SWATH_Type_L1B)
    # >i2 SolarZenith(2*nscans:MODIS_SWATH_Type_L1B, 1KM_geo_dim:MODIS_SWATH_Type_L1B)
    # >i2 SolarAzimuth(2*nscans:MODIS_SWATH_Type_L1B, 1KM_geo_dim:MODIS_SWATH_Type_L1B)
    # uint8 gflags(2*nscans:MODIS_SWATH_Type_L1B, 1KM_geo_dim:MODIS_SWATH_Type_L1B)
    # >u2 EV_Band26(10*nscans:MODIS_SWATH_Type_L1B, Max_EV_frames:MODIS_SWATH_Type_L1B)
    # uint8 EV_Band26_Uncert_Indexes(10*nscans:MODIS_SWATH_Type_L1B, Max_EV_frames:MODIS_SWATH_Type_L1B)
    # >f4 Band_250M(Band_250M)
    # >f4 Band_500M(Band_500M)
    # >f4 Band_1KM_RefSB(Band_1KM_RefSB)
    # >f4 Band_1KM_Emissive(Band_1KM_Emissive)
    # uint8 Noise in Thermal Detectors(number of emissive bands, detectors per 1km band)
    # uint8 Change in relative responses of thermal detectors(number of emissive bands, detectors per 1km band)
    # int8 DC Restore Change for Thermal Bands(number of scans, number of emissive bands, detectors per 1km band)
    # int8 DC Restore Change for Reflective 250m Bands(number of scans, number of 250m bands, detectors per 250m band)
    # int8 DC Restore Change for Reflective 500m Bands(number of scans, number of 500m bands, detectors per 500m band)
    # int8 DC Restore Change for Reflective 1km Bands(number of scans, number of 1km reflective bands, detectors per 1km band)
    dataset = Dataset(MOD021KMfile[0])
    MOD021KM = {}
    for key in ['Latitude', 'Longitude', 'EV_1KM_RefSB']:
        valid_range = dataset[key].valid_range
        MOD021KM[key] = np.array(dataset[key][:]).astype(float)
        MOD021KM[key][(MOD021KM[key] < valid_range[0]) | (MOD021KM[key] > valid_range[1])] = np.nan
    reflectance_scales = dataset['EV_1KM_RefSB'].reflectance_scales
    reflectance_offsets = dataset['EV_1KM_RefSB'].reflectance_offsets
    for ii in range(len(reflectance_scales)):
        MOD021KM['EV_1KM_RefSB'][ii] = (MOD021KM['EV_1KM_RefSB'][ii] - reflectance_offsets[ii]) * reflectance_scales[ii]
    dataset.close()
    MOD021KM['Longitude'], MOD021KM['Latitude'] = modis5kmto1km(MOD021KM['Longitude'], MOD021KM['Latitude'])

    MOD04_3K_swath_def = geometry.SwathDefinition(lons=MOD04_3K['Longitude'], lats=MOD04_3K['Latitude'])
    MOD02QKM_swath_def = geometry.SwathDefinition(lons=MOD02QKM['Longitude'], lats=MOD02QKM['Latitude'])
    MOD02HKM_swath_def = geometry.SwathDefinition(lons=MOD02HKM['Longitude'], lats=MOD02HKM['Latitude'])
    MOD021KM_swath_def = geometry.SwathDefinition(lons=MOD021KM['Longitude'], lats=MOD021KM['Latitude'])

    resampled_reflectances = {}
    for ii in range(len(MOD02QKM['EV_250_RefSB'])):
        resampled_reflectances['EV_250_RefSB_{ii}'.format(ii=ii + 1)] = kd_tree.resample_nearest(MOD02QKM_swath_def, MOD02QKM['EV_250_RefSB'][ii], area_def, radius_of_influence=250 * 3, epsilon=0.5, fill_value=np.nan)
    for ii in range(len(MOD02HKM['EV_500_RefSB'])):
        resampled_reflectances['EV_500_RefSB_{ii}'.format(ii=ii + 1)] = kd_tree.resample_nearest(MOD02HKM_swath_def, MOD02HKM['EV_500_RefSB'][ii], area_def, radius_of_influence=500 * 3, epsilon=0.5, fill_value=np.nan)
    for ii in range(len(MOD021KM['EV_1KM_RefSB'])):
        resampled_reflectances['EV_1KM_RefSB_{ii}'.format(ii=ii + 1)] = kd_tree.resample_nearest(MOD021KM_swath_def, MOD021KM['EV_1KM_RefSB'][ii], area_def, radius_of_influence=1000 * 3, epsilon=0.5, fill_value=np.nan)

    # MOD04_3K
    resampled_MOD04_3K = {}
    for kk in MOD04_3K.keys():
        if kk == 'Scan_Start_Time_units':
            continue
        if len(MOD04_3K[kk].shape) == 3 and (MOD04_3K[kk].shape[0] < MOD04_3K[kk].shape[1] and MOD04_3K[kk].shape[0] < MOD04_3K[kk].shape[2]):
            for jj in range(len(MOD04_3K[kk])):
                resampled_MOD04_3K[kk + '_{jj}'.format(jj=jj)] = kd_tree.resample_nearest(MOD04_3K_swath_def, MOD04_3K[kk][jj], area_def, radius_of_influence=3000 * 3, epsilon=0.5, fill_value=np.nan)
        elif len(MOD04_3K[kk].shape) == 3 and (MOD04_3K[kk].shape[2] < MOD04_3K[kk].shape[0] and MOD04_3K[kk].shape[2] < MOD04_3K[kk].shape[1]):
            for jj in range(MOD04_3K[kk].shape[2]):
                resampled_MOD04_3K[kk + '_{jj}'.format(jj=jj)] = kd_tree.resample_nearest(MOD04_3K_swath_def, MOD04_3K[kk][:, :, jj], area_def, radius_of_influence=3000 * 3, epsilon=0.5, fill_value=np.nan)
        else:
            resampled_MOD04_3K[kk] = kd_tree.resample_nearest(MOD04_3K_swath_def, MOD04_3K[kk], area_def, radius_of_influence=3000 * 3, epsilon=0.5, fill_value=np.nan)

    RGBIMG = np.stack((resampled_reflectances['EV_250_RefSB_1'], resampled_reflectances['EV_500_RefSB_2'], resampled_reflectances['EV_500_RefSB_1']), axis=2)
    imwrite(os.path.join(outputpath, 'IMG_MODIS_DRAGON2011_{satellite}_{granuleID}.png'.format(satellite=satellite, granuleID=granuleID)), (np.clip(RGBIMG, a_min=0, a_max=1) * 255).astype(np.uint8))

    # write the netCDF file
    ncout = Dataset(os.path.join(outputpath, 'MODIS_DRAGON2011_{satellite}_{granuleID}.nc'.format(satellite=satellite, granuleID=granuleID)), 'w', format='NETCDF4')
    # Add some attributes
    ncout.History = 'File generated on {} (UTC) by {}'.format(datetime.utcnow().strftime('%c'), os.path.basename(__file__))
    ncout.original_MOD02QKM_file = os.path.split(MOD02QKMfile[0])[-1]
    ncout.original_MOD02HKM_file = os.path.split(MOD02HKMfile[0])[-1]
    ncout.original_MOD021KM_file = os.path.split(MOD021KMfile[0])[-1]
    ncout.original_MOD04_3KM_file = os.path.split(MODISfile)[-1]

    # create dimensions
    ncout.createDimension('rows', resampled_MOD04_3K['Latitude'].shape[0])
    ncout.createDimension('columns', resampled_MOD04_3K['Latitude'].shape[1])
    # save coordinates
    ncout_latitude = ncout.createVariable('MOD04_3K_lat', 'f4', ('rows', 'columns'), zlib=True, complevel=3)
    ncout_latitude[:] = resampled_MOD04_3K['Latitude']
    ncout_latitude.standard_name = 'latitude'
    ncout_latitude.units = "degrees_north"
    ncout_longitude = ncout.createVariable('MOD04_3K_lon', 'f4', ('rows', 'columns'), zlib=True, complevel=3)
    ncout_longitude[:] = resampled_MOD04_3K['Longitude']
    ncout_longitude.standard_name = 'longitude'
    ncout_longitude.units = "degrees_east"

    MOD02QKMLatitude = kd_tree.resample_nearest(MOD02QKM_swath_def, MOD02QKM['Latitude'], area_def, radius_of_influence=250 * 3, epsilon=0.5, fill_value=np.nan)
    MOD02QKMLongitude = kd_tree.resample_nearest(MOD02QKM_swath_def, MOD02QKM['Longitude'], area_def, radius_of_influence=250 * 3, epsilon=0.5, fill_value=np.nan)
    ncout_latitude = ncout.createVariable('lat', 'f4', ('rows', 'columns'), zlib=True, complevel=3)
    ncout_latitude[:] = MOD02QKMLatitude
    ncout_latitude.standard_name = 'latitude'
    ncout_latitude.units = "degrees_north"
    ncout_longitude = ncout.createVariable('lon', 'f4', ('rows', 'columns'), zlib=True, complevel=3)
    ncout_longitude[:] = MOD02QKMLongitude
    ncout_longitude.standard_name = 'longitude'
    ncout_longitude.units = "degrees_east"

    ncout_d = ncout.createVariable('Scan_Start_Time', np.float64, ('rows', 'columns'), zlib=True, complevel=3)
    ncout_d[:] = resampled_MOD04_3K['Scan_Start_Time']

    for k in resampled_MOD04_3K.keys():
        if k == 'Scan_Start_Time' or k == 'Scan_Start_Time_units':
            continue
        ncout_d = ncout.createVariable('MOD04_3K_{k}'.format(k=k), np.float32, ('rows', 'columns'), zlib=True, complevel=3, least_significant_digit=3)
        ncout_d[:] = resampled_MOD04_3K[k]

    for ii in range(len(MOD02QKM['EV_250_RefSB'])):
        ncout_d = ncout.createVariable('TOAreflectance_band_{ii:02d}'.format(ii=ii + 1), np.float32, ('rows', 'columns'), zlib=True, complevel=3, least_significant_digit=3)
        ncout_d[:] = resampled_reflectances['EV_250_RefSB_{ii}'.format(ii=ii + 1)]
    for ii in range(len(MOD02HKM['EV_500_RefSB'])):
        ncout_d = ncout.createVariable('TOAreflectance_band_{ii:02d}'.format(ii=ii + 3), np.float32, ('rows', 'columns'), zlib=True, complevel=3, least_significant_digit=3)
        ncout_d[:] = resampled_reflectances['EV_500_RefSB_{ii}'.format(ii=ii + 1)]
    for ii in range(len(MOD021KM['EV_1KM_RefSB'])):
        ncout_d = ncout.createVariable('TOAreflectance_band_{ii:02d}'.format(ii=ii + 8), np.float32, ('rows', 'columns'), zlib=True, complevel=3, least_significant_digit=3)
        ncout_d[:] = resampled_reflectances['EV_1KM_RefSB_{ii}'.format(ii=ii + 1)]

    ncout.close()
