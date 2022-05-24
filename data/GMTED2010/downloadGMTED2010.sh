#!/bin/bash
# #################################################################################
# MODIS aerosol optical depth spatial downscaling and post-process correction
# for the DRAGON campaign 2011
# ---
#
# Script to download GMTED2010 data
#
# CODE VERSION 24 May 2022.
#
#   * Developed by: Finnish Meteorological Institute and University of Eastern Finland
#   * Contact info: Antti Lipponen (antti.lipponen@fmi.fi)
#
# #################################################################################

curl -O "https://edcintl.cr.usgs.gov/downloads/sciweb1/shared/topo/downloads/GMTED/Global_tiles_GMTED/300darcsec/mea/W090/30N090W_20101117_gmted_mea300.tif"
