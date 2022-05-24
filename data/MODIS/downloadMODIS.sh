#!/bin/bash
# #################################################################################
# MODIS aerosol optical depth spatial downscaling and post-process correction
# for the DRAGON campaign 2011
# ---
#
# Script to download MODIS data
#
# CODE VERSION 24 May 2022.
#
#   * Developed by: Finnish Meteorological Institute and University of Eastern Finland
#   * Contact info: Antti Lipponen (antti.lipponen@fmi.fi)
#
# #################################################################################
xargs -n 1 -P 4 curl -b "cookies" -c "cookies" -L --netrc-file "${HOME}/.netrc" -g -O < FILELIST
