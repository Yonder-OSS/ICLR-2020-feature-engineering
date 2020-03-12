import tifffile as tiff
import datetime
import matplotlib.pyplot as plt

DATA_PATH = '/home/snamjoshi/Documents/datasets/iclr_challenge_data/'

""" Functions """
def load_file(fp):
    """Takes a PosixPath object or string filepath
    and returns np array"""

    return tiff.imread(fp.__str__())

# List of Sentinel-2 bands in the dataset
bands = ['B01', 'B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B08', 'B8A', 'B09', 'B11', 'B12', 'CLD']

# Sample file to load:
file_name = DATA_PATH + "00/20190825/0_B03_20190825.tif"
band_data = load_file(file_name)

fig = plt.figure(figsize = (7, 7))
plt.imshow(band_data, vmin = 0, vmax = 0.15)
plt.show()