"""
Preprocessing, feature engineering, and classification with random forests. Based on: https://github.com/sentinel-hub/cv4a-iclr-2020-starter-notebooks/blob/master/cv4a-process-and-train.ipynb
The major modifications are the addition of various indices calculated from the satellite bands.
Some features produce a divide-by-zero error. This were dealt with by converting to 0 before passing to the model.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import datetime
import math
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
SCRIPT_DATA_DIR = '/home/snamjoshi/Documents/git_repos/iclr_challenge'
MODEL_NUMBER = 'M01_031220'
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

def false_color_infrared(B08, B04, B03):
    """
    Applies a false color infrared transform to the patches.
    Expects bands B08, B04, and B03 as inputs.
    Source: https://github.com/sentinel-hub/custom-scripts/tree/master/sentinel-2/false_color_infrared
    """
    gain = 2.5
    fci = np.array(list(map(lambda a : gain * a, [B08, B04, B03])))
    fci = fci[:, :, :, :, 0]
    fci = np.transpose(fci, (1, 2, 3, 0))
    return fci

def atmospherically_resistant_vegetation_index(B09, B04, B02):
    """
    Corrects atmospheric scattering effects for areas with high aerosal content.
    Expects bands B09, B04, and B02 as inputs.
    Source: https://github.com/sentinel-hub/custom-scripts/tree/master/sentinel-2/arvi
    """
    y = 0.106
    arvi = (B09 - B04 - y * (B04 - B02)) / (B09 + B04 - y * (B04 - B02))
    return arvi

def barren_soil_index(B02, B04, B08, B11, B12):
    """
    Calculates the bare soil index based on the red channel. Vegetation is in green and bare ground in red.
    Expects bands B02, B04, B08, B11, and B12 as inputs.
    Source: https://github.com/sentinel-hub/custom-scripts/tree/master/sentinel-2/barren_soil
    """
    val = 2.5 * ((B11 + B04) - (B08 + B02)) / ((B11 + B04) + (B08 + B02))
    bsi = np.array([2.5 * val, B08, B11])
    bsi = bsi[:, :, :, :, 0]
    bsi = np.transpose(bsi, (1, 2, 3, 0))
    return bsi

def infrared_agriculture_display(B04, B08, B02):
    """
    Applies an infrared transform to the patches to highlight agriculture. Similar to false color infrared.
    Expects bands B08, B04, and B03 as inputs.
    Source: https://github.com/sentinel-hub/custom-scripts/tree/master/sentinel-2/infrared_agriculture_display
    """
    gain = 2.5
    iad = np.array(list(map(lambda a : gain * a, [B04, B08, B02])))
    iad = iad[:, :, :, :, 0]
    iad = np.transpose(iad, (1, 2, 3, 0))
    return iad

# def ndvi_natural_colors(B04, B03, B02, B8A):
#     TODO: Finish converting Javascript code. Still need to fix indexing in to_RGB and figure out ternary function
#     """
#     Calculates NDVI on pixels classified as vegetation and natural colors of surface reflectance otherwise. Indicates live green vegetation.
#     Expects bands B04, B03, B02 and B8A as inputs.
#     Source: https://github.com/sentinel-hub/custom-scripts/tree/master/sentinel-2/ndvi-on-vegetation-natural_colours
#     """
#     natural_color = np.array(list(map(lambda a : gain * a, [B04, B03, B02])))
#     ndvi_color_map = {
#         "-1.0": "0x000000",
#         "-0.2": "0xA50026",
#         "0.0": "0xD73027",
#         "0.1": "0xF46D43",
#         "0.2": "0xFDAE61",
#         "0.3": "0xFEE08B",
#         "0.4": "0xFFFFBF",
#         "0.5": "0xD9EF8B",
#         "0.6": "0xA6D96A",
#         "0.7": "0x66BD63",
#         "0.8": "0x1A9850",
#         "0.9": "0x006837"
#     }

#     def index(x, y):
#         return (x - y) / (x + y)

#     def zero_fill_right_shift(val, n):
#         return (val >> n) if val >= 0 else ((val + 0x100000000) >> n)

#     def to_RGB(val):
#         rgb = map(lambda x : (x & 0xFF / 0xFF), [zero_fill_right_shift(val, 16), zero_fill_right_shift(val, 8), val])
#         return rgb

#     def find_color(col_val_pairs, val):
#         n = len(col_val_pairs)
#         for i in range(1, n):
#             if val <= col_val_pairs[i][0]:
#                 return to_RGB(col_val_pairs[i - 1][1])
#         return to_RGB(cloud_coverage[n - 1][1])

#     findcolor(nvdi_color_map, index(B8A, B04))

def corrected_transform_vegetation_index(B04, B03):
    """
    Transforms NDVIs histogram into a normal distribution using absolute values to account for cases where NDVI < -0.5
    Expects bands B04 and B03 as inputs.
    Source: https://github.com/sentinel-hub/custom-scripts/blob/master/sentinel-2/indexdb/id_244.js
    """
    ctvi = (((B04 - B03) / (B04 + B03)) + 0.5) / np.abs(((B04 - B03) / (B04 + B03)) + 0.5) * np.sqrt(np.abs((((B04 - B03) / (B04 + B03))) + 0.5))
    return ctvi

def global_environment_monitoring_index(B08, B04):
    """
    Vegetation index that reduces atmospheric effects.
    Expects bands B08 and B04 as inputs.
    Source: https://github.com/sentinel-hub/custom-scripts/blob/master/sentinel-2/indexdb/id_25.js
    """
    gemi = ((2.0 * (np.power(B08, 2.0) - np.power(B04, 2.0)) + 1.5 * B08 + 0.5 * B04) / (B08 + B04 + 0.5) * (1.0 - 0.25 * (2.0 * (np.power(B08, 2.0) - np.power(B04, 2.0)) + 1.5 * B08 + 0.5 * B04) / (B08 + B04 + 0.5)) - (B04 - 0.125) / (1.0 - B04))
    return gemi

def wide_dynamic_range_vegetation_index(B08, B04):
    """
    Enhanced dynamic range NDVI.
    Expects bands B08 and B04 as input.
    Source: https://github.com/sentinel-hub/custom-scripts/blob/master/sentinel-2/indexdb/id_125.js
    """
    wdrvi = (0.1 * B08 - B04) / (0.1 * B08 + B04)
    return wdrvi

def enhanced_vegetation_index(B08, B04, B02):
    """
    Modified version of NDVI.
    Expects bands B08, B04, and B02 as input.
    Source: https://github.com/sentinel-hub/custom-scripts/blob/master/sentinel-2/indexdb/id_16.js
    """
    evi = 2.5 * (B08 - B04) / ((B08 + 6.0 * B04 - 7.5 * B02) + 1.0)
    return evi

def modified_triangular_vegetation_index(B08, B04, B03):
    """
    Modified version of the triangular vegetation index.
    Expects bands B08, B04, and B03 as inputs.
    Source: https://github.com/sentinel-hub/custom-scripts/blob/master/sentinel-2/indexdb/id_46.js
    """
    mtvi = 1.2 * (1.2 * (B08 - B03) - 2.5 * (B04 - B03))
    return mtvi

def chlorophyll_absorption_ratio_index(B05, B04, B03):
    """
    CARI2 for chloropyll absorption.
    Expects bands B05, B04, and B03 as inputs.
    Source: https://github.com/sentinel-hub/custom-scripts/blob/master/sentinel-2/indexdb/id_250.js
    """
    a = 0.567
    cari2 = (np.abs(((B05 - B03) / 150.0 * B04 + B04 + B03 - (a * B03))) / np.power((np.power(a, 2.0) + 1.0), 0.5)) * (B05 / B04)
    return cari2

def adjusted_transformed_soil_adjusted_vegetation_index(B08, B04):
    """
    Expects bands B08 and B04 as inputs.
    Source: https://github.com/sentinel-hub/custom-scripts/blob/master/sentinel-2/indexdb/id_209.js
    """
    atsavi =  1.22 * (B08 - 1.22 * B04 - 0.03) / (1.22 * B08 + B04 - 1.22 * 0.03 + 0.08 * (1.0 + np.power(1.22, 2.0)))
    return atsavi

def green_normalized_difference_vegetation_index(B08, B03):
    """
    Expects bands B08 and B03 as inputs.
    Source: https://github.com/sentinel-hub/custom-scripts/blob/master/sentinel-2/indexdb/id_28.js
    """
    gndvi = (B08 - B03) / (B08 + B03)
    return gndvi

def modified_simple_ratio(B08, B04, B01):
    """
    Expects bands B08, B04, and B01 as inputs.
    Source: https://github.com/sentinel-hub/custom-scripts/blob/master/sentinel-2/indexdb/id_167.js
    """
    msr = (B08 - B01) / (B04 - B01)
    return msr

def nonlinear_vegetation_index(B08, B04):
    """
    Expects bands B08 and B04 as inputs.
    Source: https://github.com/sentinel-hub/custom-scripts/blob/master/sentinel-2/indexdb/id_111.js
    """
    nvi = (np.power(B08, 2.0) - B04) / (np.power(B08, 2.0) + B04)
    return nvi

def renormalized_difference_vegetation_index(B08, B04):
    """
    Expects bands B08 and B04 as inputs.
    Source: https://github.com/sentinel-hub/custom-scripts/blob/master/sentinel-2/indexdb/id_76.js
    """
    rdvi = (B08 - B04) / np.sqrt(B08 + B04) * 0.5
    return rdvi

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

""" Clouds tasks """
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

# define a valid data mask used in interpolation
valid_data = MapFeatureTask((FeatureType.MASK, 'CLOUD'),
                            (FeatureType.MASK, 'VALID_DATA'),
                            np.logical_not)

""" Other tasks """
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

# ndvi
ndvi_task = NormalizedDifferenceIndexTask((FeatureType.DATA, 'S2-BANDS-L2A'),
                                          (FeatureType.DATA, 'NDVI'),
                                          [7,3])

# interpolation
lin_interp = LinearInterpolation((FeatureType.DATA, 'S2-BANDS-L2A'),
                                 mask_feature=(FeatureType.MASK, 'VALID_DATA'),
                                 copy_features=[(FeatureType.DATA, 'NDVI'),
                                                (FeatureType.DATA, 'FCI'),
                                                (FeatureType.DATA, 'ARVI'),
                                                (FeatureType.DATA, 'BSI'),
                                                (FeatureType.DATA, 'IAD'),
                                                (FeatureType.DATA, 'CTVI'),
                                                (FeatureType.DATA, 'GEMI'),
                                                (FeatureType.DATA, 'WDRVI'),
                                                (FeatureType.DATA, 'EVI'),
                                                (FeatureType.DATA, 'MTVI'),
                                                (FeatureType.DATA, 'CARI2'),
                                                (FeatureType.DATA, 'ATSAVI'),
                                                (FeatureType.DATA, 'GNDVI'),
                                                (FeatureType.DATA, 'MSR'),
                                                (FeatureType.DATA, 'NVI'),
                                                (FeatureType.DATA, 'RDVI')],
                                 resample_range=dates)

# add false color infrared feature
false_color_infrared_task = ZipFeatureTask({FeatureType.DATA: ['B08', 'B04', 'B03']},
                                           (FeatureType.DATA, 'FCI'),
                                           false_color_infrared)

atmospherically_resistant_vegetation_index_task = ZipFeatureTask({FeatureType.DATA: ['B09', 'B04', 'B02']},
                                                                 (FeatureType.DATA, 'ARVI'),
                                                                 atmospherically_resistant_vegetation_index)

barren_soil_index_task = ZipFeatureTask({FeatureType.DATA: ['B02', 'B04', 'B08', 'B11', 'B12']},
                                        (FeatureType.DATA, 'BSI'),
                                        barren_soil_index)

infrared_agriculture_display_task = ZipFeatureTask({FeatureType.DATA: ['B04', 'B08', 'B02']},
                                                   (FeatureType.DATA, 'IAD'),
                                                   infrared_agriculture_display)

corrected_transform_vegetation_index_task = ZipFeatureTask({FeatureType.DATA: ['B04', 'B03']},
                                                           (FeatureType.DATA, 'CTVI'),
                                                           corrected_transform_vegetation_index)

global_environment_monitoring_index_task = ZipFeatureTask({FeatureType.DATA: ['B08', 'B04']},
                                                          (FeatureType.DATA, 'GEMI'),
                                                          global_environment_monitoring_index)

wide_dynamic_range_vegetation_index_task = ZipFeatureTask({FeatureType.DATA: ['B08', 'B04']},
                                                          (FeatureType.DATA, 'WDRVI'),
                                                          wide_dynamic_range_vegetation_index)

enhanced_vegetation_index_task = ZipFeatureTask({FeatureType.DATA: ['B08', 'B04', 'B02']},
                                                (FeatureType.DATA, 'EVI'),
                                                enhanced_vegetation_index)

modified_triangular_vegetation_index_task = ZipFeatureTask({FeatureType.DATA: ['B08', 'B04', 'B03']},
                                                           (FeatureType.DATA, 'MTVI'),
                                                           modified_triangular_vegetation_index)

chlorophyll_absorption_ratio_index_task = ZipFeatureTask({FeatureType.DATA: ['B05', 'B04', 'B03']},
                                                          (FeatureType.DATA, 'CARI2'),
                                                          chlorophyll_absorption_ratio_index)

adjusted_transformed_soil_adjusted_vegetation_index_task = ZipFeatureTask({FeatureType.DATA: ['B08', 'B04']},
                                                                          (FeatureType.DATA, 'ATSAVI'),
                                                                          adjusted_transformed_soil_adjusted_vegetation_index)

green_normalized_difference_vegetation_index_task = ZipFeatureTask({FeatureType.DATA: ['B08', 'B03']},
                                                                   (FeatureType.DATA, 'GNDVI'),
                                                                   green_normalized_difference_vegetation_index)

modified_simple_ratio_task = ZipFeatureTask({FeatureType.DATA: ['B08', 'B04', 'B01']},
                                            (FeatureType.DATA, 'MSR'),
                                            modified_simple_ratio)

nonlinear_vegetation_index_task = ZipFeatureTask({FeatureType.DATA: ['B08', 'B04']},
                                                 (FeatureType.DATA, 'NVI'),
                                                 nonlinear_vegetation_index)

renormalized_difference_vegetation_index_task = ZipFeatureTask({FeatureType.DATA: ['B08', 'B04']},
                                                               (FeatureType.DATA, 'RDVI'),
                                                               renormalized_difference_vegetation_index)

# sample only fields
sample = SampleValid((FeatureType.MASK_TIMELESS, 'FIELD_ID'), fraction=1.0, no_data_value=0)

# if invalid data are at beginning or end of time-series, pad with nearest value
fillout = ValueFilloutTask((FeatureType.DATA, 'S2-BANDS-L2A'))

# Create workflow and preprocessing
workflow = ComposeTask([cloud_mask,
                        cloud_dilation,
                        add_cc,
                        count_train_fields,
                        count_test_fields])

preprocess = ComposeTask([valid_data,
                          ndvi_task,
                          false_color_infrared_task,
                          corrected_transform_vegetation_index_task,
                          global_environment_monitoring_index_task,
                          infrared_agriculture_display_task,
                          barren_soil_index_task,
                          atmospherically_resistant_vegetation_index_task,
                          wide_dynamic_range_vegetation_index_task,
                          enhanced_vegetation_index_task,
                          modified_triangular_vegetation_index_task,
                          chlorophyll_absorption_ratio_index_task,
                          adjusted_transformed_soil_adjusted_vegetation_index_task,
                          green_normalized_difference_vegetation_index_task,
                          modified_simple_ratio_task,
                          nonlinear_vegetation_index_task,
                          renormalized_difference_vegetation_index_task,
                          lin_interp,
                          fillout])

""" Data inspection """
eop_name = 'eopatch-00-3-1'
eop = workflow.execute(EOPatch.load(f'{ROOT_DATA_DIR}/eopatches/{eop_name}'))

# time_idx = -1

# fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(15,15))

# axs[0,0].imshow(np.moveaxis(eop.data['S2-BANDS-L2A'][time_idx,...,[3,2,1]],0,-1)*3.5)
# axs[0,0].set_title(f'RGB {eop_name} {eop.timestamp[time_idx]}')
# axs[0,1].imshow(eop.mask['CLOUD'][time_idx].squeeze())
# axs[0,1].set_title(f'Post-proc. cloud mask')

# axs[1,0].imshow(eop.mask_timeless['CROP_ID'].squeeze())
# axs[1,0].set_title(f'CROP_ID')
# axs[1,1].imshow(eop.mask_timeless['FIELD_ID'].squeeze())
# axs[1,1].set_title(f'FIELD_ID')

# plt.tight_layout()
# plt.show()

""" Processed data """
eop_linear = preprocess(eop)
eop_linear

# time_idx = -1

# fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(15, 20))

# axs[0,0].imshow(np.moveaxis(eop.data['S2-BANDS-L2A'][time_idx,...,[3,2,1]],0,-1)*3.5)
# axs[0,0].set_title(f'Original RGB {eop_name} {eop.timestamp[time_idx]}')
# axs[0,1].imshow(eop.mask['CLOUD'][time_idx].squeeze())
# axs[0,1].set_title(f'Original cloud mask')
# axs[1,0].imshow(np.moveaxis(eop_linear.data['S2-BANDS-L2A'][time_idx,...,[3,2,1]],0,-1)*3.5)
# axs[1,0].set_title(f'Linearly interpolated RGB {eop_name} {eop.timestamp[time_idx]}')
# # axs[1,1].imshow(eop_linear.data['FCI'][time_idx].squeeze())
# # axs[1,1].set_title(f'False Color Infrared {eop_name} {eop.timestamp[time_idx]}')
# # axs[1,1].imshow(eop_linear.data['ARVI'][time_idx].squeeze())
# # axs[1,1].set_title(f'Atmospherically resistant vegetation index {eop_name} {eop.timestamp[time_idx]}')
# # axs[1,1].imshow(eop_linear.data['BSI'][time_idx].squeeze())
# # axs[1,1].set_title(f'Barren Soil Index {eop_name} {eop.timestamp[time_idx]}')
# # axs[1,1].imshow(eop_linear.data['IAD'][time_idx].squeeze())
# # axs[1,1].set_title(f'Infrared Agricultural Display {eop_name} {eop.timestamp[time_idx]}')
# # axs[1,1].imshow(eop_linear.data['CTVI'][time_idx].squeeze())
# # axs[1,1].set_title(f'Corrected transform vegetation index {eop_name} {eop.timestamp[time_idx]}')
# axs[1,1].imshow(eop_linear.data['GEMI'][time_idx].squeeze())
# axs[1,1].set_title(f'Global environment monitoring index {eop_name} {eop.timestamp[time_idx]}')

# plt.tight_layout()
# plt.show()

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
    arvi_arr = np.moveaxis(eop_proc.data['ARVI'].squeeze(),1,0)
    bsi_arr = np.moveaxis(eop_proc.data['BSI'].squeeze(),1,0)
    ctvi_arr = np.moveaxis(eop_proc.data['CTVI'].squeeze(),1,0)
    fci_arr = np.moveaxis(eop_proc.data['FCI'].squeeze(),1,0)
    gemi_arr = np.moveaxis(eop_proc.data['GEMI'].squeeze(),1,0)
    iad_arr = np.moveaxis(eop_proc.data['IAD'].squeeze(),1,0)
    wdrvi_arr = np.moveaxis(eop_proc.data['WDRVI'].squeeze(),1,0)
    evi_arr = np.moveaxis(eop_proc.data['EVI'].squeeze(),1,0)
    mtvi_arr = np.moveaxis(eop_proc.data['MTVI'].squeeze(),1,0)
    cari2_arr = np.moveaxis(eop_proc.data['CARI2'].squeeze(),1,0)
    atsavi_arr = np.moveaxis(eop_proc.data['ATSAVI'].squeeze(),1,0)
    gndvi_arr = np.moveaxis(eop_proc.data['GNDVI'].squeeze(),1,0)
    msr_arr = np.moveaxis(eop_proc.data['MSR'].squeeze(),1,0)
    nvi_arr = np.moveaxis(eop_proc.data['NVI'].squeeze(),1,0)
    rdvi_arr = np.moveaxis(eop_proc.data['RDVI'].squeeze(),1,0)
    cld_arr = np.moveaxis(eop.data['CLOUD_PROB'].squeeze(),1,0)

    features.append(np.concatenate((ftr_arr,
                                    ndvi_arr[...,np.newaxis],
                                    arvi_arr[...,np.newaxis],
                                    bsi_arr,
                                    ctvi_arr[...,np.newaxis],
                                    fci_arr,
                                    gemi_arr[...,np.newaxis],
                                    iad_arr,
                                    wdrvi_arr[..., np.newaxis],
                                    evi_arr[..., np.newaxis],
                                    mtvi_arr[..., np.newaxis],
                                    cari2_arr[..., np.newaxis],
                                    atsavi_arr[..., np.newaxis],
                                    gndvi_arr[..., np.newaxis],
                                    msr_arr[..., np.newaxis],
                                    nvi_arr[..., np.newaxis],
                                    rdvi_arr[..., np.newaxis],
                                    cld_arr[...,np.newaxis]), axis=-1))
    field_id.append(eop.mask_timeless['FIELD_ID'].squeeze())
    crop_id.append(eop.mask_timeless['CROP_ID'].squeeze())

    pbar.update()

features = np.concatenate(features)
features[np.isinf(features)] = 0
field_id = np.concatenate(field_id)
crop_id = np.concatenate(crop_id)

#features.shape, field_id.shape, crop_id.shape

#len(np.unique(field_id[crop_id==0])), len(np.unique(field_id[crop_id!=0]))

""" Visualize normalized difference vegetation index """
# ndvi = (features[...,7]-features[...,3])/(features[...,7]+features[...,3])
# np.unique(crop_id, return_counts=True)

# fig, axs = plt.subplots(figsize=(20,7))

# crop_code = 1
# profile_plot(axs, eop.timestamp, np.mean(ndvi[crop_id==crop_code],axis=0), np.std(ndvi[crop_id==crop_code],axis=0),
#              label=f'Crop {crop_code}', alpha=0.5)
# crop_code = 2
# profile_plot(axs, eop.timestamp, np.mean(ndvi[crop_id==crop_code],axis=0), np.std(ndvi[crop_id==crop_code],axis=0),
#              label=f'Crop {crop_code}', alpha=0.5)
# crop_code = 3
# profile_plot(axs, dates, np.mean(ndvi[crop_id==crop_code],axis=0), np.std(ndvi[crop_id==crop_code],axis=0),
#              label=f'Crop {crop_code}', alpha=0.5)
# crop_code = 4
# profile_plot(axs, dates, np.mean(ndvi[crop_id==crop_code],axis=0), np.std(ndvi[crop_id==crop_code],axis=0),
#              label=f'Crop {crop_code}', alpha=0.5)
# crop_code = 5
# profile_plot(axs, dates, np.mean(ndvi[crop_id==crop_code],axis=0), np.std(ndvi[crop_id==crop_code],axis=0),
#              label=f'Crop {crop_code}', alpha=0.5)
# crop_code = 6
# profile_plot(axs, dates, np.mean(ndvi[crop_id==crop_code],axis=0), np.std(ndvi[crop_id==crop_code],axis=0),
#              label=f'Crop {crop_code}', alpha=0.5)
# crop_code = 7
# profile_plot(axs, dates, np.mean(ndvi[crop_id==crop_code],axis=0), np.std(ndvi[crop_id==crop_code],axis=0),
#              label=f'Crop {crop_code}', alpha=0.5)
# axs.grid()
# axs.legend()
# axs.set_ylim(0,1);

""" Inspect other info """
# df with all eopatches and all timeframes
# df = pd.DataFrame(info)

# # df with eopatches with reference info and aggregated over all timeframes
# patchdf = df.groupby(by='eopatch').mean()
# patchdf = patchdf.loc[(patchdf.n_train!=0) | (patchdf.n_test!=0)]
# patchdf['ratio'] = patchdf.n_test/patchdf.n_train

# df.loc[df.field_count>0].hist('cloud_coverage',bins=100);
# patchdf.hist('ratio',bins=20);

""" Train the model """
X = features[crop_id>0]
y = crop_id[crop_id>0]
fid = field_id[crop_id>0]

X_test = features[crop_id==0]
y_test = crop_id[crop_id==0]
fid_test = field_id[crop_id==0]

fields, px_count = np.unique(fid, return_counts=True)

# plt.hist(px_count, bins=100, range=(0,100));
# plt.title('Size of training fields in pixels');
# plt.xlabel('Size in pixels');
# plt.ylabel('Number of fields');

# Split training data into 90/10
n_fields = len(np.unique(fid))
train_frac = 0.9

unq_field_ids = np.unique(fid)

random_state = 7
np.random.seed(random_state)
train_fields = np.random.choice(unq_field_ids, int(n_fields*train_frac), replace=False)
val_fields = unq_field_ids[~np.in1d(unq_field_ids, train_fields)]

#len(train_fields), len(val_fields)

X_train = X[np.in1d(fid, train_fields)]
y_train = y[np.in1d(fid, train_fields)]

X_val = X[np.in1d(fid, val_fields)]
y_val = y[np.in1d(fid, val_fields)]

fid_train = fid[np.in1d(fid, train_fields)]
fid_val = fid[np.in1d(fid, val_fields)]

#y_train.shape, y_val.shape, y.shape

# Train RF model on pixels and get scores on cross-validation split
X_train, y_train, fid_train = shuffle(X_train, y_train, fid_train, random_state=random_state)
#np.unique(y_train, return_counts=True)
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
#tr.head()

val = pd.DataFrame(X_val.reshape(X_val.shape[0], X_val.shape[1]*X_val.shape[2]))
val['label'] = y_val
val['fid'] = fid_val
#val.head()

# Train per field using mean values
# Group
train_grouped = tr.groupby('fid').mean().reset_index()
val_grouped = val.groupby('fid').mean().reset_index()

X_train, y_train = train_grouped[train_grouped.columns[1:-2]], train_grouped['label']
X_val, y_val = val_grouped[val_grouped.columns[1:-2]], val_grouped['label']

#np.unique(y_train, return_counts=True)

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
#test.head()

# Group test as we did for val
test_grouped = test.groupby('fid').mean().reset_index()
preds = model.predict_proba(test_grouped[test_grouped.columns[1:-1]])

prob_df = pd.DataFrame({
    'Field_ID':test_grouped['fid'].values
})
for c in range(1, 8):
    prob_df['Crop_ID_'+str(c)] = preds[:,c-1]
#prob_df.head()

# Save test results to .csv file for submission
sample_sub_path = f'{SCRIPT_DATA_DIR}/SampleSubmission.csv'
ss = pd.read_csv(sample_sub_path)
#ss.head()

# Merge the two, to get all the required field IDs
ss = pd.merge(ss['Field_ID'], prob_df, how='left', on='Field_ID')
print(ss.isna().sum()['Crop_ID_1']) # Missing fields
# Fill in a low but non-zero val for the missing rows:
ss = ss.fillna(1/7) # There are 34 missing fields
ss.to_csv(f'{SCRIPT_DATA_DIR}/submissions/starter_nb_submission_' + MODEL_NUMBER + '.csv', index=False)

#pd.read_csv(f'{SCRIPT_DATA_DIR}/submissions/starter_nb_submission_' + MODEL_NUMBER + '.csv').head()
