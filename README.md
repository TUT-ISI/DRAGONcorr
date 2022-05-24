# MODIS aerosol optical depth spatial downscaling and post-process correction for the DRAGON campaign 2011

[![DOI](https://zenodo.org/badge/495721657.svg)](https://zenodo.org/badge/latestdoi/495721657)

Scripts and codes to download and prepare the necessary data + run the spatial downscaling and post-process correction of MODIS aerosol optical depth (AOD) data for the DRAGON campaign 2011.

* Codes are developed by: Finnish Meteorological Institute and University of Eastern Finland
* Contact info: Antti Lipponen (antti.lipponen@fmi.fi)


## Python environment
To create a Python environment for the codes using Conda, first clone the repository, next go to the main directory and run:
```
conda env create -f environment.yml
conda activate DRAGONenv
```

## Running the code

First, you need to download and pre-process the MODIS and AERONET data. To download and pre-process:
```
./downloadprepare.sh
```
As the datasets are relatively large, the download may take hours.

To run the actual downscaling and post-process correction model training and evaluation, run:
```
python runCorrection.py
```

## License

```
MIT License

Copyright (c) 2022 Finnish Meteorological Institute and University of Eastern Finland

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```
