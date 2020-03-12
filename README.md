# ICLR Remote Sensing Challenge - Feature Engineering Approach

A feature engineering approach to solving the ICLR remote sensing challenge. The scripts are all based on the start notebook available [here](https://github.com/sentinel-hub/cv4a-iclr-2020-starter-notebooks). These notebooks have been modified to implement various vegetation indices and other remote sensing indices available on [this page](https://github.com/sentinel-hub/custom-scripts#sentinel-2).

Currently, this repo is not implemented as a pip installable. Outside of more standard statistical/preprocessing packages and built-ins, it also depends on `tifffile`, `tqdm`, and `eolearn`.

## Usage

Clone this repo locally and then do the following:
1. Change the `ROOT_DATA_DIR` in the `iclr_challenge_preprocessing.py` file.
2. Run `iclr_challenge_preprocessing.py` which will split all the data into patches and assign into data types compatible with the [EOLearn](https://eo-learn.readthedocs.io/en/latest/index.html) library.
3. Change the `ROOT_DATA_DIR` in the `iclr_challenge_classification.py` file to point to the data directory where you preprocessed in step 2.
4. Change the `SCRIPT_DATA_DIR` to the path to this repo.
5. Change the `MODEL_NUMBER` whenever you run a model. `iclr_challenge_classification.py` will output a submission CSV in the correct format and append `MODEL_NUMBER` in the file name.

## Remote sensing indices used

The base notebook uses NDVI. This index is used for vegetation detection but many alternative measures have been proposed. The classification script implements the following other indices that are modifications of NDVI or use a different overall approach for vegetation detection:
* Normalized difference vegetation index (NDVI)
* Corrected transform vegetation index (CTVI)
* Global environment monitoring index (GEMI)
* Atmospherically resistant vegetation index (ARVI)
* Wide dynamic range vegetation index (WDRV)
* Enchanced vegetation index (EVI)
* Modified triangular vegetation index (MTVI)
* Adjusted transformed soil adjusted vegetation index (ATSAVI)
* Green normalized difference vegetation index (GNDVI)
* Nonlinear vegetation index (NVI)
* Renormalized difference vegetation index (RDVI)

Many of these indices correct for issues with NDVI which suffers in certain frequency ranges. `ARVI` corrects for aerosol levels in the atmosphere which results in light scattering effects. It appears that this may be a problem with some regions in Ethopia looking at a heat map for aerosal levels in the area but it is unclear if this index is really necessary.

Additionally, the following infrared indices:
* False color infrared (FCI)
* Infrared agriculture display (IGD)

A few other miscellaneous transforms used in some remote sensing work for crop detection:
* Barren soil index (BSI)
* Chlorophyll absorption ratio index 2 (CARI2)
* Modified simple ratio (MSR)

Note that some of these indices do produce a divide-by-zero error in some cases. All such cases are globally replaced with 0's in the classification script.

Preprocessing also includes steps to remove clouds from the image.