import tifffile as tiff
import datetime
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import log_loss

DATA_PATH = '/home/snamjoshi/Documents/datasets/iclr_challenge_data/'
OUT_PATH = '/home/snamjoshi/Documents/git_repos/iclr_challenge/'

""" Functions """
def load_file(fp):
    """Takes a PosixPath object or string filepath
    and returns np array"""

    return tiff.imread(fp.__str__())

def load_rgb(data_path, tile, date):
    r = load_file(f"{data_path}/{tile}/{date}/{tile[1]}_B04_{date}.tif")
    g = load_file(f"{data_path}/{tile}/{date}/{tile[1]}_B03_{date}.tif")
    b = load_file(f"{data_path}/{tile}/{date}/{tile[1]}_B02_{date}.tif")
    arr = np.dstack((r, g, b))
    print(max(g.flatten()))
    return arr

""" Data loading """
# List of dates that an observation from Sentinel-2 is provided in the training dataset
dates = [datetime.datetime(2019, 6, 6, 8, 10, 7),
         datetime.datetime(2019, 7, 1, 8, 10, 4),
         datetime.datetime(2019, 7, 6, 8, 10, 8),
         datetime.datetime(2019, 7, 11, 8, 10, 4),
         datetime.datetime(2019, 7, 21, 8, 10, 4),
         datetime.datetime(2019, 8, 5, 8, 10, 7),
         datetime.datetime(2019, 8, 15, 8, 10, 6),
         datetime.datetime(2019, 8, 25, 8, 10, 4),
         datetime.datetime(2019, 9, 9, 8, 9, 58),
         datetime.datetime(2019, 9, 19, 8, 9, 59),
         datetime.datetime(2019, 9, 24, 8, 9, 59),
         datetime.datetime(2019, 10, 4, 8, 10),
         datetime.datetime(2019, 11, 3, 8, 10)]

bands = ['B01', 'B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B08', 'B8A', 'B09', 'B11', 'B12', 'CLD']

# Sample file to load:
file_name = DATA_PATH + "00/20190825/0_B03_20190825.tif"
band_data = load_file(file_name)

# Quick way to see an RGB image. Can mess with the scaling factor to change brightness (3 in this example)
fig, ax = plt.subplots(figsize = (12, 18))
ax.imshow(load_rgb(DATA_PATH, '01', '20190825') * 3)
plt.tight_layout()
plt.show()

""" Preprocessing: get data for each pixel in fields """
# Read in the labels, find pixel locations of all fields, then store the pixel locations, field ID, and label
# Not super efficient but  ¯\_(ツ)_/¯

row_locs = []
col_locs = []
field_ids = []
labels = []
tiles = []
for tile in range(4):
    fids = DATA_PATH + f'0{tile}/{tile}_field_id.tif'
    labs = DATA_PATH + f'0{tile}/{tile}_label.tif'
    fid_arr = load_file(fids)
    lab_arr = load_file(labs)
    for row in range(len(fid_arr)):
        for col in range(len(fid_arr[0])):
            if fid_arr[row][col] != 0:
                row_locs.append(row)
                col_locs.append(col)
                field_ids.append(fid_arr[row][col])
                labels.append(lab_arr[row][col])
                tiles.append(tile)

df = pd.DataFrame({
    'fid':field_ids,
    'label':labels,
    'row_loc': row_locs,
    'col_loc':col_locs,
    'tile':tiles
})

print(df.shape)
print(df.groupby('fid').count().shape)
df.head()

# Loop through all images, sampling different image bands to get the values for each pixel in each field
# Sample the bands at different dates as columns in our new dataframe
col_names = []
col_values = []

for tile in range(4): # 1) For each tile
    print('Tile: ', tile)
    for d in dates: # 2) For each date
        print(str(d))
        d = ''.join(str(d.date()).split('-')) # Nice date string
        t = '0' + str(tile)
        for b in bands: # 3) For each band
            col_name = d + '_' + b

            if tile == 0:
                # If the column doesn't exist, create it and populate with 0s
                df[col_name] = 0

            # Load im
            im = load_file(f"{DATA_PATH}/{t}/{d}/{t[1]}_{b}_{d}.tif")

            # Going four levels deep. Each second on the outside is four weeks in this loop
            # If we die here, there's no waking up.....
            vals = []
            for row, col in df.loc[df.tile == tile][['row_loc', 'col_loc']].values: # 4) For each location of a pixel in a field
                vals.append(im[row][col])
            df.loc[df.tile == tile, col_name] = vals
    df.head()

df.to_csv(OUT_PATH + 'sampled_data.csv', index = False)

""" Some simple models """
# Load the data
df = pd.read_csv(OUT_PATH + 'sampled_data.csv')
print(df.shape)
df.head()

# Split into train and test sets
train = df.loc[df.label != 0]
test =  df.loc[df.label == 0]
train.shape, test.shape

# Split train into train and val, so that we can score our models locally (test is the test set)
# Splitting on field ID since we want to predict for unseen FIELDS
val_field_ids = train.groupby('fid').mean().reset_index()['fid'].sample(frac = 0.3).values
tr = train.loc[~train.fid.isin(val_field_ids)].copy()
val = train.loc[train.fid.isin(val_field_ids)].copy()

# Split into X and Y for modelling
X_train, y_train = tr[tr.columns[5:]], tr['label']
X_val, y_val = val[val.columns[5:]], val['label']

# Predicting on pixels
## Fit model (takes a few minutes since we have plenty of rows)
model = RandomForestClassifier(n_estimators = 500)
model.fit(X_train.fillna(0), y_train)

## Get predicted probabilities
preds = model.predict_proba(X_val.fillna(0))

## Add to val dataframe as columns
for i in range(7):
    val[str(i + 1)] = preds[:,i]

## Get 'true' vals as columns in val
for c in range(1, 8):
    val['true'+str(c)] = (val['label'] == c).astype(int)

pred_cols = [str(i) for i in range(1, 8)]
true_cols = ['true' + str(i) for i in range(1, 8)]

## Score
print('Pixel score:', log_loss(val[true_cols], val[pred_cols])) # Note this isn't the score you'd get for a submission!!
print('Field score:', log_loss(val.groupby('fid').mean()[true_cols], val.groupby('fid').mean()[pred_cols])) # This is what we'll compare to

## Inspect visually
val[['label'] + true_cols + pred_cols].head()

# Predicting on fields (grouping first)
## Group
train_grouped = tr.groupby('fid').mean().reset_index()
val_grouped = val.groupby('fid').mean().reset_index()
X_train, y_train = train_grouped[train_grouped.columns[5:]], train_grouped['label']
X_val, y_val = val_grouped[train_grouped.columns[5:]], val_grouped['label']

## Fit model
model = RandomForestClassifier(n_estimators=500)
model.fit(X_train.fillna(0), y_train)

## Get predicted probabilities
preds = model.predict_proba(X_val.fillna(0))

## Add to val_grouped dataframe as columns
for i in range(7):
    val_grouped[str(i+1)] = preds[:,i]

## Get 'true' vals as columns in val
for c in range(1, 8):
    val_grouped['true'+str(c)] = (val_grouped['label'] == c).astype(int)

pred_cols = [str(i) for i in range(1, 8)]
true_cols = ['true' + str(i) for i in range(1, 8)]
val_grouped[['label'] + true_cols + pred_cols].head()

## Already grouped, but just to double check:
print('Field score:', log_loss(val_grouped.groupby('fid').mean()[true_cols], val_grouped.groupby('fid').mean()[pred_cols]))

""" Making a submission """
# Group test as we did for val
test_grouped = test.groupby('fid').mean().reset_index()
preds = model.predict_proba(test_grouped[train_grouped.columns[5:]])

prob_df = pd.DataFrame({
    'Field_ID':test_grouped['fid'].values
})
for c in range(1, 8):
    prob_df['Crop_ID_' + str(c)] = preds[:,c - 1]
prob_df.head()

# Check the sample submission and compare
ss = pd.read_csv('SampleSubmission.csv')
ss.head()

# Merge the two, to get all the required field IDs
ss = pd.merge(ss['Field_ID'], prob_df, how = 'left', on = 'Field_ID')
print(ss.isna().sum()['Crop_ID_1']) # Missing fields
# Fill in a low but non-zero val for the missing rows:
ss = ss.fillna(1 / 7) # There are 34 missing fields
ss.to_csv('starter_nb_submission.csv', index = False)