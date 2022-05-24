#!/bin/bash

# #################################################################################
# MODIS aerosol optical depth spatial downscaling and post-process correction
# for the DRAGON campaign 2011
# ---
#
# Script to reproject and grid the surface elevation GMTED2010 data.
#
# CODE VERSION 24 May 2022.
#
#   * Developed by: Finnish Meteorological Institute and University of Eastern Finland
#   * Contact info: Antti Lipponen (antti.lipponen@fmi.fi)
#
# #################################################################################

gdalwarp -t_srs "EPSG:32618" -te 315000 4285000 435000 4405000 -tr 250 250 -r near -of "netCDF" "GMTED2010/30N090W_20101117_gmted_mea300.tif" "GMTED2010/GMTED2010.nc"
