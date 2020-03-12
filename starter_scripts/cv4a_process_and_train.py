import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import datetime
import tifffile as tiff
import glob

from tqdm.auto import tqdm

from eolearn.core import EOPatch, FeatureType, FeatureTypeSet, OverwritePermission, LinearWorkflow, LoadTask
from eolearn.core import MapFeatureTask, ZipFeatureTask, EOTask
from eolearn.features import SimpleFilterTask, LinearInterpolation, ValueFilloutTask, NormalizedDifferenceIndexTask

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import log_loss
from sklearn.utils import shuffle
from sklearn.metrics import accuracy_score
from sklearn import metrics

from skimage.morphology import disk, binary_dilation, binary_erosion

ROOT_DATA_DIR = '/home/snamjoshi/Documents/datasets/iclr_challenge_data'
bands = ['B01', 'B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B08', 'B8A', 'B09', 'B11', 'B12', 'CLD']

dates = [datetime.datetime(2019, 6, 6, 0, 0),
 datetime.datetime(2019, 7, 1, 0, 0),
 datetime.datetime(2019, 7, 6, 0, 0),
 datetime.datetime(2019, 7, 11, 0, 0),
 datetime.datetime(2019, 7, 21, 0, 0),
 datetime.datetime(2019, 8, 5, 0, 0),
 datetime.datetime(2019, 8, 15, 0, 0),
 datetime.datetime(2019, 8, 25, 0, 0),
 datetime.datetime(2019, 9, 9, 0, 0),
 datetime.datetime(2019, 9, 19, 0, 0),
 datetime.datetime(2019, 9, 24, 0, 0),
 datetime.datetime(2019, 10, 4, 0, 0),
 datetime.datetime(2019, 11, 3, 0, 0)]

""" Functions """
def to_mask(prob, threshold):
    """Converts a probability float array to binary mask array."""
    return prob > threshold

def cloud_coverage(mask):
    """Estimates cloud coverage (fraction of cloudy pixels) from a cloud mask"""
    return np.count_nonzero(mask.squeeze(),axis=(1,2))[...,np.newaxis]/mask[0,...,0].size

def post_process_mask(masks, operation, element):
    """Apply skimage operation over all masks (timeframes)."""
    pp_masks = np.asarray([operation(mask.squeeze(), element) for mask in masks], dtype=np.bool)
    return pp_masks[...,np.newaxis]

def count_fields(field_ids, crop_ids, train=True):
    """
    Counts number of (train/test) fields in an eopatch.

    Training fields have crop_id, while test fields don't.
    """
    crop_mask = (crop_ids.squeeze()==0)
    if train:
        crop_mask = (crop_ids.squeeze()!=0)

    field_count = np.count_nonzero(np.unique(field_ids.squeeze()[crop_mask]))

    return np.array([field_count])

def get_all_field_count(eopatch):
    return np.count_nonzero(np.unique(eop.mask_timeless['FIELD_ID']))

def get_cloud_coverage(eopatch):
    return eopatch.scalar['CLOUD_COVERAGE'].squeeze()

def get_field_count(eopatch, train=True):
    feature = 'TEST_FIELDS_COUNT'
    if train:
        feature = 'TRAIN_FIELDS_COUNT'

    return eopatch.scalar_timeless[feature][0]

def remove_arg(arg, kwargs):
    if arg in kwargs:
        kwargs_without_arg = kwargs.copy()
        del kwargs_without_arg[arg]
        return kwargs_without_arg

    return kwargs

def profile_plot(ax, date, values, std=None, **kwargs):
    """
    Adds profile plot to axes object.
    """
    kwargs_without_label = remove_arg('label', kwargs)
    kwargs_without_alpha = remove_arg('alpha', kwargs)

    ax.plot(date, values, **kwargs_without_alpha)
    if std is not None:
        ax.fill_between(date, values - std, values + std, **kwargs_without_label)

""" Classes """
class ValidDataFractionPredicate:
    def __init__(self, threshold):
        self.threshold = threshold

    def __call__(self, coverage):
        return coverage < self.threshold

class SampleValid(EOTask):
    """
    The task samples pixels with a value in given timeless feature different from no valid data value.
    """

    def __init__(self, feature, fraction=1.0, no_data_value=0, sample_features=...):
        """ Task to sample pixels from a reference timeless raster mask, excluding a no valid data value

        :param feature:  Reference feature used to select points to be sampled
        :param fraction: Fraction of valid points to be sampled
        :param no_data_value: Value of non-valid points to be ignored
        """
        self.feature_type, self.feature_name, self.new_feature_name = next(
            self._parse_features(feature, new_names=True,
                                 default_feature_type=FeatureType.MASK_TIMELESS,
                                 allowed_feature_types={FeatureType.MASK_TIMELESS},
                                 rename_function='{}_SAMPLED'.format)())
        self.fraction = fraction
        self.no_data_value = no_data_value
        self.sample_features = self._parse_features(sample_features)

    def execute(self, in_eopatch, seed=None):
        eopatch = in_eopatch.__copy__()

        mask = eopatch[self.feature_type][self.feature_name].squeeze()

        if mask.ndim != 2:
            raise ValueError('Invalid shape of sampling reference map.')

        np.random.seed(seed)

        rows, cols = np.where(mask != self.no_data_value)
        sampled = np.random.rand(len(rows)) > (1.0 - self.fraction)
        rows = rows[sampled]
        cols = cols[sampled]

        for feature_type, feature_name in self.sample_features(eopatch):

            if feature_type in FeatureTypeSet.RASTER_TYPES.intersection(FeatureTypeSet.SPATIAL_TYPES):

                if feature_type.is_time_dependent():
                    sampled_data = eopatch[feature_type][feature_name][:, rows, cols, :]
                else:
                    sampled_data = eopatch[feature_type][feature_name][rows, cols, :]

                # here a copy of sampled array is returned and assigned to feature of a shallow copy
                # orig_eopatch[feature_type][feature_name] remains unmodified
                eopatch[feature_type][feature_name] = sampled_data[..., np.newaxis, :]

        new_mask = np.ones_like(mask) * self.no_data_value
        new_mask[rows, cols] = mask[rows, cols]
        eopatch[self.feature_type][self.new_feature_name] = new_mask[..., np.newaxis]

        eopatch[FeatureType.SCALAR_TIMELESS]['SAMPLED_ROWS'] = rows
        eopatch[FeatureType.SCALAR_TIMELESS]['SAMPLED_COLS'] = cols

        return eopatch

class ComposeTask(EOTask):
    """Composes several tasks together.
    """

    def __init__(self, tasks):
        self.tasks = tasks

    def execute(self, eopatch):
        for t in self.tasks:
            eopatch = t(eopatch)
        return eopatch

""" Clouds """
# convert cloud probas to cloud masks
cloud_mask = MapFeatureTask((FeatureType.DATA, 'CLOUD_PROB'),
                            (FeatureType.MASK, 'CLOUD'),
                            to_mask,
                            threshold=50)

# apply erosion to cloud mask
cloud_erosion = MapFeatureTask((FeatureType.MASK, 'CLOUD'),
                               (FeatureType.MASK, 'CLOUD'),
                               post_process_mask,
                               operation=binary_erosion,
                               element=disk(2))

# apply cloud dilation to cloud mask
cloud_dilation = MapFeatureTask((FeatureType.MASK, 'CLOUD'),
                                (FeatureType.MASK, 'CLOUD'),
                                post_process_mask,
                                operation=binary_dilation,
                                element=disk(10))

# compute cloud coverage based
add_cc = MapFeatureTask((FeatureType.MASK, 'CLOUD'),
                        (FeatureType.SCALAR, 'CLOUD_COVERAGE'),
                        cloud_coverage)

# filter out frames with valid data ratio higher than threshold
valid_data_predicate = ValidDataFractionPredicate(0.3)
filter_task = SimpleFilterTask((FeatureType.SCALAR, 'CLOUD_COVERAGE'), valid_data_predicate)

# count train fields
count_train_fields = ZipFeatureTask({FeatureType.MASK_TIMELESS: ['FIELD_ID', 'CROP_ID']},
                                    (FeatureType.SCALAR_TIMELESS, 'TRAIN_FIELDS_COUNT'),
                                    count_fields,
                                    train=True)

# count test fields
count_test_fields = ZipFeatureTask({FeatureType.MASK_TIMELESS: ['FIELD_ID', 'CROP_ID']},
                                   (FeatureType.SCALAR_TIMELESS, 'TEST_FIELDS_COUNT'),
                                   count_fields,
                                   train=False)

# define a valid data mask used in interpolation
valid_data = MapFeatureTask((FeatureType.MASK, 'CLOUD'),
                            (FeatureType.MASK, 'VALID_DATA'),
                            np.logical_not)

# ndvi
ndvi_task = NormalizedDifferenceIndexTask((FeatureType.DATA, 'S2-BANDS-L2A'),
                                          (FeatureType.DATA, 'NDVI'),
                                          [7,3])

# interpolation
lin_interp = LinearInterpolation((FeatureType.DATA, 'S2-BANDS-L2A'),
                                 mask_feature=(FeatureType.MASK, 'VALID_DATA'),
                                 copy_features=[(FeatureType.DATA, 'NDVI')], # comment this out if NDVI is not used
                                 resample_range=dates)

# sample only fields
sample = SampleValid((FeatureType.MASK_TIMELESS, 'FIELD_ID'), fraction=1.0, no_data_value=0)

# if invalid data are at beginning or end of time-series, pad with nearest value
fillout = ValueFilloutTask((FeatureType.DATA, 'S2-BANDS-L2A'))

workflow = ComposeTask([cloud_mask,
                        #cloud_erosion,
                        cloud_dilation,
                        add_cc,
                        #filter_task,
                        count_train_fields,
                        count_test_fields])

preprocess = ComposeTask([valid_data, ndvi_task, lin_interp, fillout])

""" Data inspection """
eop_name = 'eopatch-00-3-1'
eop = workflow.execute(EOPatch.load(f'{ROOT_DATA_DIR}/eopatches/{eop_name}'))

time_idx = -1

fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(15,15))

axs[0,0].imshow(np.moveaxis(eop.data['S2-BANDS-L2A'][time_idx,...,[3,2,1]],0,-1)*3.5)
axs[0,0].set_title(f'RGB {eop_name} {eop.timestamp[time_idx]}')
axs[0,1].imshow(eop.mask['CLOUD'][time_idx].squeeze())
axs[0,1].set_title(f'Post-proc. cloud mask')

axs[1,0].imshow(eop.mask_timeless['CROP_ID'].squeeze())
axs[1,0].set_title(f'CROP_ID')
axs[1,1].imshow(eop.mask_timeless['FIELD_ID'].squeeze())
axs[1,1].set_title(f'FIELD_ID')

plt.tight_layout()
plt.show

""" Processed data """
eop_linear = preprocess(eop)
eop_linear

time_idx = -1

fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(15, 20))

axs[0,0].imshow(np.moveaxis(eop.data['S2-BANDS-L2A'][time_idx,...,[3,2,1]],0,-1)*3.5)
axs[0,0].set_title(f'Original RGB {eop_name} {eop.timestamp[time_idx]}')
axs[0,1].imshow(eop.mask['CLOUD'][time_idx].squeeze())
axs[0,1].set_title(f'Original cloud mask')
axs[1,0].imshow(np.moveaxis(eop_linear.data['S2-BANDS-L2A'][time_idx,...,[3,2,1]],0,-1)*3.5)
axs[1,0].set_title(f'Linearly interpolated RGB {eop_name} {eop.timestamp[time_idx]}')
axs[1,1].imshow(eop_linear.data['NDVI'][time_idx].squeeze())
axs[1,1].set_title(f'Linearly interpolated NDVI {eop_name} {eop.timestamp[time_idx]}')

plt.tight_layout()
plt.show()

""" Process patches and sample all pixels with reference """
eopatch_names = glob.glob(f'{ROOT_DATA_DIR}/eopatches/*')

info = []

pbar = tqdm(total=len(eopatch_names))
for eopatch in eopatch_names:
    eop = EOPatch.load(eopatch, lazy_loading=True)
    eop = workflow(eop)

    name = eopatch.split('/')[-1]
    for idx, date in enumerate(eop.timestamp):
        info.append({'eopatch':name, 'date':date,
                     'field_count':get_all_field_count(eop),
                     'cloud_coverage':get_cloud_coverage(eop)[idx],
                     'n_train':get_field_count(eop),
                     'n_test':get_field_count(eop, train=False),
                    })

    # sample and save sample eopatch to disk
    sampled_eop = sample(eop)
    sampled_eop.save(eopatch.replace('eopatches','sampled-eopatches'),
                     overwrite_permission=OverwritePermission.OVERWRITE_PATCH)

    pbar.update()

""" Merge sampled pixels to numpy arrays """
sampled_eopatch_names = glob.glob(f'{ROOT_DATA_DIR}/sampled-eopatches/*')

features = []
crop_id = []
field_id = []

pbar = tqdm(total=len(sampled_eopatch_names))
for eopatch in sampled_eopatch_names:
    eop = EOPatch.load(eopatch, lazy_loading=True)
    eop_proc =  preprocess(eop)

    ftr_arr = np.moveaxis(eop_proc.data['S2-BANDS-L2A'].squeeze(),1,0)
    ndvi_arr = np.moveaxis(eop_proc.data['NDVI'].squeeze(),1,0)
    cld_arr = np.moveaxis(eop.data['CLOUD_PROB'].squeeze(),1,0)

    features.append(np.concatenate((ftr_arr, ndvi_arr[...,np.newaxis], cld_arr[...,np.newaxis]), axis=-1))
    field_id.append(eop.mask_timeless['FIELD_ID'].squeeze())
    crop_id.append(eop.mask_timeless['CROP_ID'].squeeze())

    pbar.update()

features = np.concatenate(features)
field_id = np.concatenate(field_id)
crop_id = np.concatenate(crop_id)

features.shape, field_id.shape, crop_id.shape

len(np.unique(field_id[crop_id==0])), len(np.unique(field_id[crop_id!=0]))

""" Visualize normalized difference vegetation index """
ndvi = (features[...,7]-features[...,3])/(features[...,7]+features[...,3])
np.unique(crop_id, return_counts=True)

fig, axs = plt.subplots(figsize=(20,7))

crop_code = 1
profile_plot(axs, eop.timestamp, np.mean(ndvi[crop_id==crop_code],axis=0), np.std(ndvi[crop_id==crop_code],axis=0),
             label=f'Crop {crop_code}', alpha=0.5)
crop_code = 2
profile_plot(axs, eop.timestamp, np.mean(ndvi[crop_id==crop_code],axis=0), np.std(ndvi[crop_id==crop_code],axis=0),
             label=f'Crop {crop_code}', alpha=0.5)
crop_code = 3
profile_plot(axs, dates, np.mean(ndvi[crop_id==crop_code],axis=0), np.std(ndvi[crop_id==crop_code],axis=0),
             label=f'Crop {crop_code}', alpha=0.5)
crop_code = 4
profile_plot(axs, dates, np.mean(ndvi[crop_id==crop_code],axis=0), np.std(ndvi[crop_id==crop_code],axis=0),
             label=f'Crop {crop_code}', alpha=0.5)
crop_code = 5
profile_plot(axs, dates, np.mean(ndvi[crop_id==crop_code],axis=0), np.std(ndvi[crop_id==crop_code],axis=0),
             label=f'Crop {crop_code}', alpha=0.5)
crop_code = 6
profile_plot(axs, dates, np.mean(ndvi[crop_id==crop_code],axis=0), np.std(ndvi[crop_id==crop_code],axis=0),
             label=f'Crop {crop_code}', alpha=0.5)
crop_code = 7
profile_plot(axs, dates, np.mean(ndvi[crop_id==crop_code],axis=0), np.std(ndvi[crop_id==crop_code],axis=0),
             label=f'Crop {crop_code}', alpha=0.5)
axs.grid()
axs.legend()
axs.set_ylim(0,1);

""" Inspect other info """
# df with all eopatches and all timeframes
df = pd.DataFrame(info)

# df with eopatches with reference info and aggregated over all timeframes
patchdf = df.groupby(by='eopatch').mean()
patchdf = patchdf.loc[(patchdf.n_train!=0) | (patchdf.n_test!=0)]
patchdf['ratio'] = patchdf.n_test/patchdf.n_train

df.loc[df.field_count>0].hist('cloud_coverage',bins=100);
patchdf.hist('ratio',bins=20);

""" Train a model """
X = features[crop_id>0]
y = crop_id[crop_id>0]
fid = field_id[crop_id>0]

X_test = features[crop_id==0]
y_test = crop_id[crop_id==0]
fid_test = field_id[crop_id==0]

fields, px_count = np.unique(fid, return_counts=True)

plt.hist(px_count, bins=100, range=(0,100));
plt.title('Size of training fields in pixels');
plt.xlabel('Size in pixels');
plt.ylabel('Number of fields');

# Split training data into 90/10
n_fields = len(np.unique(fid))
train_frac = 0.9

unq_field_ids = np.unique(fid)

random_state = 7
np.random.seed(random_state)
train_fields = np.random.choice(unq_field_ids, int(n_fields*train_frac), replace=False)
val_fields = unq_field_ids[~np.in1d(unq_field_ids, train_fields)]

len(train_fields), len(val_fields)

X_train = X[np.in1d(fid, train_fields)]
y_train = y[np.in1d(fid, train_fields)]

X_val = X[np.in1d(fid, val_fields)]
y_val = y[np.in1d(fid, val_fields)]

fid_train = fid[np.in1d(fid, train_fields)]
fid_val = fid[np.in1d(fid, val_fields)]

y_train.shape, y_val.shape, y.shape

# Train RF model on pixels and get scores on cross-validation split
X_train, y_train, fid_train = shuffle(X_train, y_train, fid_train, random_state=random_state)
np.unique(y_train, return_counts=True)
rf = RandomForestClassifier(n_estimators=500, random_state=random_state)

rf.fit(X_train.reshape(X_train.shape[0], X_train.shape[1]*X_train.shape[2]), y_train)

preds = rf.predict_proba(X_val.reshape(X_val.shape[0], X_val.shape[1]*X_val.shape[2]))

val_results = pd.DataFrame(preds,columns=['1','2','3','4','5','6','7'])

for c in range(1, 8):
    val_results['true'+str(c)] = (y_val == c).astype(int)

val_results['fid'] = field_id[np.in1d(field_id, val_fields)]

pred_cols = [str(i) for i in range(1, 8)]
true_cols = ['true'+str(i) for i in range(1, 8)]

# Score
print('Pixel score:', log_loss(val_results[true_cols], val_results[pred_cols])) # Note this isn't the score you'd get for a submission!!
print('Field score:', log_loss(val_results.groupby('fid').mean()[true_cols],
                               val_results.groupby('fid').mean()[pred_cols])) # This is what we'll compare to

rf_preds =  rf.predict(X_val.reshape(X_val.shape[0], X_val.shape[1]*X_val.shape[2]))

accuracy_score(y_val, rf_preds)

unq_lbls = np.unique(y_val)

f1_scores = metrics.f1_score(y_val, rf_preds, labels=unq_lbls, average=None)
recall = metrics.recall_score(y_val, rf_preds, labels=unq_lbls, average=None)
precision = metrics.precision_score(y_val, rf_preds, labels=unq_lbls, average=None)

for idx, class_id in enumerate(unq_lbls):
    print(f'        * class {class_id:5d} = {f1_scores[idx]*100:2.1f} | {recall[idx]*100:2.1f} | {precision[idx]*100:2.1f} | {np.count_nonzero(y_val==class_id):8d}')

tr = pd.DataFrame(X_train.reshape(X_train.shape[0], X_train.shape[1]*X_train.shape[2]))
tr['label'] = y_train
tr['fid'] = fid_train
tr.head()

val = pd.DataFrame(X_val.reshape(X_val.shape[0], X_val.shape[1]*X_val.shape[2]))
val['label'] = y_val
val['fid'] = fid_val
val.head()

# Train per field using mean values
# Group
train_grouped = tr.groupby('fid').mean().reset_index()
val_grouped = val.groupby('fid').mean().reset_index()

X_train, y_train = train_grouped[train_grouped.columns[1:-2]], train_grouped['label']
X_val, y_val = val_grouped[val_grouped.columns[1:-2]], val_grouped['label']

np.unique(y_train, return_counts=True)

# Predicting on fields (grouping first)

# Fit model
model = RandomForestClassifier(n_estimators=500, random_state=random_state)
model.fit(X_train.fillna(-1), y_train)

# Get predicted probabilities
preds = model.predict_proba(X_val.fillna(-1))

# Add to val_grouped dataframe as columns
for i in range(7):
  val_grouped[str(i+1)] = preds[:,i]

# Get 'true' vals as columns in val
for c in range(1, 8):
  val_grouped['true'+str(c)] = (val_grouped['label'] == c).astype(int)

pred_cols = [str(i) for i in range(1, 8)]
true_cols = ['true'+str(i) for i in range(1, 8)]
val_grouped[['label']+true_cols+pred_cols].head()

# Already grouped, but just to double check:
print('Field score:', log_loss(val_grouped.groupby('fid').mean()[true_cols],
                               val_grouped.groupby('fid').mean()[pred_cols]))

test = pd.DataFrame(X_test.reshape(X_test.shape[0], X_test.shape[1]*X_test.shape[2]))
test['fid'] = fid_test
test.head()

# Group test as we did for val
test_grouped = test.groupby('fid').mean().reset_index()
preds = model.predict_proba(test_grouped[test_grouped.columns[1:-1]])

prob_df = pd.DataFrame({
    'Field_ID':test_grouped['fid'].values
})
for c in range(1, 8):
    prob_df['Crop_ID_'+str(c)] = preds[:,c-1]
prob_df.head()

# Save test results to .csv file for submission
sample_sub_path = f'{ROOT_DATA_DIR}/SampleSubmission.csv'
ss = pd.read_csv(sample_sub_path)
ss.head()

# Merge the two, to get all the required field IDs
ss = pd.merge(ss['Field_ID'], prob_df, how='left', on='Field_ID')
print(ss.isna().sum()['Crop_ID_1']) # Missing fields
# Fill in a low but non-zero val for the missing rows:
ss = ss.fillna(1/7) # There are 34 missing fields
ss.to_csv(f'{ROOT_DATA_DIR}/starter_nb_submission.csv', index=False)

pd.read_csv(f'{ROOT_DATA_DIR}/starter_nb_submission.csv').head()