'''import tensorflow as tf
import numpy as np

# Define features to compute stats for
FEATURES = [
    'elevation', 'th', 'vs', 'tmmn', 'tmmx', 'sph',
    'pr', 'pdsi', 'NDVI', 'population', 'erc',
    'PrevFireMask', 'FireMask'
]

# TFRecord schema (adjust shapes as needed)
feature_description = {
    f: tf.io.FixedLenFeature([], tf.float32) for f in FEATURES
}

# Path to TFRecords (wildcard allowed)
tfrecord_files = tf.io.gfile.glob("/Volumes/StudyNProjects/UnitecFolder/Thesis_Project/WildFireDetection/archive/*.tfrecord")

# Storage
feature_data = {k: [] for k in FEATURES}

# Parse each record
def parse_fn(example_proto):
    return tf.io.parse_single_example(example_proto, feature_description)

raw_dataset = tf.data.TFRecordDataset(tfrecord_files)
parsed_dataset = raw_dataset.map(parse_fn)

# Collect values
for example in parsed_dataset.take(10000):  # Limit for speed; increase if needed
    for k in FEATURES:
        val = example[k].numpy()
        feature_data[k].append(val)

# Compute stats
DATA_STATS = {}
for k, vals in feature_data.items():
    vals_np = np.array(vals)
    DATA_STATS[k] = (
        round(np.min(vals_np), 4),
        round(np.max(vals_np), 4),
        round(np.mean(vals_np), 4),
        round(np.std(vals_np), 4)
    )

# Print for pasting
print("DATA_STATS = {")
for k, v in DATA_STATS.items():
    print(f"    '{k}': {v},")
print("}")'''

import tensorflow as tf

# Load one TFRecord file
raw_dataset = tf.data.TFRecordDataset("/Volumes/StudyNProjects/UnitecFolder/Thesis_Project/WildFireDetection/archive/next_day_wildfire_spread_eval_00.tfrecord")

# Take one record and decode it
for raw_record in raw_dataset.take(1):
    example = tf.train.Example()
    example.ParseFromString(raw_record.numpy())
    print("Available keys in this record:")
    for key in example.features.feature.keys():
        print(key)
       

'''
tmmn
NDVI
FireMask
population
elevation
vs
pdsi
pr
tmmx
sph
th
PrevFireMask
erc
'''
'''
import tensorflow as tf

tfrecord_file = '/Volumes/StudyNProjects/UnitecFolder/Thesis_Project/Applied Project/archive/next_day_wildfire_spread_train_00.tfrecord'
raw_dataset = tf.data.TFRecordDataset(tfrecord_file)

# Take one example
for raw_record in raw_dataset.take(1):
    example = tf.train.Example()
    example.ParseFromString(raw_record.numpy())
    feature_vals = example.features.feature['tmmn'].float_list.value
    print("tmmn length:", len(feature_vals))  # <- Check this number

import glob

print(glob.glob("/Volumes/StudyNProjects/UnitecFolder/Thesis_Project/WildFireDetection/Training_ds/*.tfrecord"))
print(glob.glob("/Volumes/StudyNProjects/UnitecFolder/Thesis_Project/WildFireDetection/Testing_ds/*.tfrecord"))
print(glob.glob("/Volumes/StudyNProjects/UnitecFolder/Thesis_Project/WildFireDetection/Validation_ds/*.tfrecord"))
    '''