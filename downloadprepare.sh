#!/bin/bash

# #################################################################################
# MODIS aerosol optical depth spatial downscaling and post-process correction
# for the DRAGON campaign 2011
# ---
#
# Script to download and prepare MODIS + AERONET data
# For MODIS downloads, you need to get and add NASA Earthdata credentials to your home directory .netrc file
#  More info: https://wiki.earthdata.nasa.gov/display/EL/How+To+Access+Data+With+cURL+And+Wget
#
# CODE VERSION 24 May 2022.
#
#   * Developed by: Finnish Meteorological Institute and University of Eastern Finland
#   * Contact info: Antti Lipponen (antti.lipponen@fmi.fi)
#
# #################################################################################

cd data
cd AERONET
./downloadAERONET.sh
cd ..
cd MODIS
./downloadMODIS.sh
cd ..
cd GMTED2010
./downloadGMTED2010.sh
cd ..
./prepareGMTED2010.sh
python prepareMODISdata.py
python prepareAERONETdata.py
cd ..
