"""
Creates all EOPatches for images. Based on: https://github.com/sentinel-hub/cv4a-iclr-2020-starter-notebooks/blob/master/cv4a-crop-challenge-to-eolearn.ipynb
The major modifications are to separate out each channel instead of pooling them together in one data feature.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import datetime
import tifffile as tiff
import glob

from tqdm.auto import tqdm
from eolearn.core import EOPatch, FeatureType

ROOT_DATA_DIR = '/home/snamjoshi/Documents/datasets/iclr_challenge_data'
bands = ['B01', 'B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B08', 'B8A', 'B09', 'B11', 'B12', 'CLD']

""" Functions """
def load_file(fp):
    """Takes a PosixPath object or string filepath
    and returns np array"""
    return tiff.imread(fp.__str__())

def load_image(tile, date):
    img = list()
    for band in bands:
        file_name = f"{ROOT_DATA_DIR}/{tile}/{date}/{int(tile)}_{band}_{date}.tif"
        img.append(load_file(file_name))
    return np.dstack(img)

def load_timeseries(tile, dates):
    tstack = list()
    with tqdm(dates, total=len(dates), desc=f"reading images for tile {tile}") as pbar:
        for date in pbar:
            tstack.append(load_image(tile, date.strftime("%Y%m%d")))
    return np.stack(tstack)

def load_label(tile):
    return tiff.imread(f'{ROOT_DATA_DIR}/{tile}/{tile[1]}_label.tif')

def load_field_id(tile):
    return tiff.imread(f'{ROOT_DATA_DIR}/{tile}/{tile[1]}_field_id.tif')

def save_as_eopatch(data, label, field_id, dates, folder):
    """
    Creates an EOPatch and adds data:
    * S2 L2A bands will be stored in DATA feature under name `S2-BANDS-L2A`
    * Individual bands will be stored in DATA features under their respective channel names
    * sen2cor cloud probabilities will be stored in DATA feature under name `CLOUD_PROB`
    * labels will be stored in MASK_TIMELESS under name `CROP_ID`
    * field ids  will be stored in MASK_TIMELESS under name `FIELD_ID`
    * dates of observations will be added to timestamps

    EOPatch is saved to specified folder.
    """
    eopatch = EOPatch()

    # Bands
    eopatch.add_feature(FeatureType.DATA, 'B01', data[..., 0][..., np.newaxis])
    eopatch.add_feature(FeatureType.DATA, 'B02', data[..., 1][..., np.newaxis])
    eopatch.add_feature(FeatureType.DATA, 'B03', data[..., 2][..., np.newaxis])
    eopatch.add_feature(FeatureType.DATA, 'B04', data[..., 3][..., np.newaxis])
    eopatch.add_feature(FeatureType.DATA, 'B05', data[..., 4][..., np.newaxis])
    eopatch.add_feature(FeatureType.DATA, 'B06', data[..., 5][..., np.newaxis])
    eopatch.add_feature(FeatureType.DATA, 'B07', data[..., 6][..., np.newaxis])
    eopatch.add_feature(FeatureType.DATA, 'B08', data[..., 7][..., np.newaxis])
    eopatch.add_feature(FeatureType.DATA, 'B8A', data[..., 8][..., np.newaxis])
    eopatch.add_feature(FeatureType.DATA, 'B09', data[..., 9][..., np.newaxis])
    eopatch.add_feature(FeatureType.DATA, 'B11', data[..., 10][..., np.newaxis])
    eopatch.add_feature(FeatureType.DATA, 'B12', data[..., 11][..., np.newaxis])
    eopatch.add_feature(FeatureType.DATA, 'S2-BANDS-L2A', data[..., :12])

    eopatch.add_feature(FeatureType.DATA, 'CLOUD_PROB', data[..., -1][..., np.newaxis])
    eopatch.add_feature(FeatureType.MASK_TIMELESS, 'CROP_ID', label[..., np.newaxis])
    eopatch.add_feature(FeatureType.MASK_TIMELESS, 'FIELD_ID', field_id[..., np.newaxis])
    eopatch.timestamp = dates

    eopatch.save(folder)

    return eopatch

def get_field_count(eopatch):
    return len(np.unique(eopatch.mask_timeless['FIELD_ID']))-1

""" Get list of dates """
# Get a list of dates that an observation from Sentinel-2 is provided for from the currently downloaded imagery
tile_dates = {}
for f in glob.glob(f'{ROOT_DATA_DIR}/**/*.tif', recursive = True):
    if len(f.split('/')) != len(f'{ROOT_DATA_DIR}'.split('/')) + 3:
        continue
    tile_id = f.split('/')[-3]
    date = datetime.datetime.strptime(f.split('/')[-2], '%Y%m%d')
    if tile_dates.get(tile_id, None) is None:
        tile_dates[tile_id] = []
    tile_dates[tile_id].append(date)

for tile_id, dates in tile_dates.items():
    tile_dates[tile_id] = list(set(tile_dates[tile_id]))

# All tiles have the same timestamps
for tile in tile_dates.keys():
    assert sorted(tile_dates['00']) == sorted(tile_dates[tile])

# Load a tile
selected_tile = list(tile_dates.keys())[0]
dates = sorted(tile_dates[selected_tile])
timeseries = load_timeseries(selected_tile, dates)

print(f'Tile {selected_tile} imagery has shape [n_times, height, width, bands]: {timeseries.shape}')

""" Convert to EOPatches """
h, w = timeseries.shape[1:3]
h_split = np.arange(0, h+1, int(h/5))
w_split = np.arange(0, w+1, int(w/4))

# Split and save EOPatch tiles
eopatch_dir = f'{ROOT_DATA_DIR}/eopatches'

for tile in sorted(tile_dates.keys()):
    dates = sorted(tile_dates[tile])
    timeseries = load_timeseries(tile, dates)
    labels = load_label(tile)
    fields = load_field_id(tile)

    pbar = tqdm(total=(len(h_split)-1)*(len(w_split)-1), desc="Extracting and saving EOPatches")
    for hidx in range(0, len(h_split)-1):
        for widx in range(0, len(w_split)-1):
            eop_folder = f'{eopatch_dir}/eopatch-{tile}-{hidx}-{widx}'
            data = timeseries[:, h_split[hidx]:h_split[hidx+1], w_split[widx]:w_split[widx+1], :]
            label = labels[h_split[hidx]:h_split[hidx+1], w_split[widx]:w_split[widx+1]]
            field = fields[h_split[hidx]:h_split[hidx+1], w_split[widx]:w_split[widx+1]]

            save_as_eopatch(data, label, field, dates, eop_folder)
            pbar.update(1)