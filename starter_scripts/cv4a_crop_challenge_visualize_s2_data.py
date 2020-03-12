import datetime
import matplotlib.pyplot as plt
import tifffile as tiff
import numpy as np
import glob
from tqdm.notebook import tqdm
from skimage import exposure

DATA_PATH = '/home/snamjoshi/Documents/datasets/iclr_challenge_data/00/20190606/'

""" Functions """
def load_file(fp):
    """Takes a PosixPath object or string filepath
    and returns np array"""
    return tiff.imread(fp.__str__())

def load_image(date):
    img = list()
    for band in bands:
        file_name = f"data/{selected_tile}/{date}/{int(selected_tile)}_{band}_{date}.tif"
        img.append(load_file(file_name))
    return np.dstack(img)

def load_timeseries():
    tstack = list()
    with tqdm(dates, total=len(dates), desc="reading images") as pbar:
        for date in pbar:
            tstack.append(load_image(date.strftime("%Y%m%d")))
    return np.stack(tstack)

# Get a list of dates that an observation from Sentinel-2 is provided for from the currently downloaded imagery
tile_dates = {}
for f in glob.glob(DATA_PATH + '*.tif', recursive = True):
    if len(f.split('/')) != 4:
        continue
    tile_id = f.split('/')[1]
    date = datetime.datetime.strptime(f.split('/')[2], '%Y%m%d')
    if tile_dates.get(tile_id, None) is None:
        tile_dates[tile_id] = []
    tile_dates[tile_id].append(date)

for tile_id, dates in tile_dates.items():
    tile_dates[tile_id] = list(set(tile_dates[tile_id]))

""" Time series loading and band visualizations """
selected_tile = list(tile_dates.keys())[0]
dates = sorted(tile_dates[selected_tile])

bands = ['B01', 'B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B08', 'B8A', 'B09', 'B11', 'B12', 'CLD']
timeseries = load_timeseries()
print(timeseries.shape)

""" Visualize band combinations """
# List of Sentinel-2 bands in the dataset

rgb = np.array(['B04','B03', 'B02'])
rgb_idx = [bands.index(b) for b in rgb]
rgb_stack = timeseries[:,:,:,rgb_idx]

false_color = np.array(['B08','B04', 'B03'])
false_color_idx = [bands.index(b) for b in false_color]
false_color_stack = timeseries[:,:,:,false_color_idx]

perc_rgb = np.percentile(rgb_stack.mean(2), (10, 90))
perc_false_color = np.percentile(np.median(false_color_stack,axis=2), (2, 98))

red = timeseries[:,:,:,bands.index("B04")]
nir = timeseries[:,:,:,bands.index("B08")]
ndvi = (red-nir) / (red+nir)

fig,axs = plt.subplots(len(dates),3,figsize=(15,len(dates)*8))

with tqdm(range(len(dates)),total=len(dates)) as pbar:
    for date_idx in pbar:

        ax = axs[date_idx,0]
        rgb_img = rgb_stack[date_idx]
        rgb_img = exposure.rescale_intensity(rgb_img, in_range=(perc_rgb[0],perc_rgb[1]))
        ax.imshow(rgb_img)
        ax.axis('off')
        dt = dates[date_idx].strftime("%Y-%m-%d")
        ax.set_title(f"RGB {dt} {rgb}")

        ax = axs[date_idx,1]
        false_color_img = false_color_stack[date_idx]
        false_color_img = exposure.rescale_intensity(false_color_img, in_range=(perc_false_color[0],perc_false_color[1]))
        ax.imshow(false_color_img)
        ax.axis('off')
        dt = dates[date_idx].strftime("%Y-%m-%d")
        ax.set_title(f"False-Color {dt} {false_color}")

        ax = axs[date_idx,2]
        ndvi_img = ndvi[date_idx]
        ax.imshow(ndvi_img, cmap="Greens_r")
        ax.axis('off')
        dt = dates[date_idx].strftime("%Y-%m-%d")
        ax.set_title(f"NDVI {dt} (B08-B04)/(B08+B04)")

fig.tight_layout()