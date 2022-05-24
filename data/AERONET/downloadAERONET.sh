#!/bin/bash
# #################################################################################
# MODIS aerosol optical depth spatial downscaling and post-process correction
# for the DRAGON campaign 2011
# ---
#
# Script to download and unpack AERONET data
#
# CODE VERSION 24 May 2022.
#
#   * Developed by: Finnish Meteorological Institute and University of Eastern Finland
#   * Contact info: Antti Lipponen (antti.lipponen@fmi.fi)
#
# #################################################################################

curl "https://aeronet.gsfc.nasa.gov/Site_Lists/dragon_2011_locations_2011_lev20.txt" -O
curl "https://aeronet.gsfc.nasa.gov/data_push/DRAGON/DRAGON-USA_AOT_Level2_All_Points.tar.gz" -O
tar -xzf "DRAGON-USA_AOT_Level2_All_Points.tar.gz"
