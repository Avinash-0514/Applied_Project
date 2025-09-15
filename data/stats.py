import tensorflow as tf
import numpy as np
import os
import pandas as pd

# TFRecord path
tfrecord_path = '/Volumes/StudyNProjects/UnitecFolder/Thesis_Project/Applied Project/archive/*.tfrecord'
raw_dataset = tf.data.TFRecordDataset(tf.io.gfile.glob(tfrecord_path))
'''
# Define the features you want to parse
feature_description = {
        'tmmn': tf.io.FixedLenFeature([4096], tf.float32),
        'NDVI': tf.io.FixedLenFeature([4096], tf.float32),
        'FireMask': tf.io.FixedLenFeature([4096], tf.float32),
        'population': tf.io.FixedLenFeature([4096], tf.float32),
        'elevation': tf.io.FixedLenFeature([4096], tf.float32),
        'vs': tf.io.FixedLenFeature([4096], tf.float32),
        'pdsi': tf.io.FixedLenFeature([4096], tf.float32),
        'pr': tf.io.FixedLenFeature([4096], tf.float32),
        'tmmx': tf.io.FixedLenFeature([4096], tf.float32),
        'sph': tf.io.FixedLenFeature([4096], tf.float32),
        'th': tf.io.FixedLenFeature([4096], tf.float32),
        'PrevFireMask': tf.io.FixedLenFeature([4096], tf.float32),
        'erc': tf.io.FixedLenFeature([4096], tf.float32),
}

def _parse_function(example_proto):
    return tf.io.parse_single_example(example_proto, feature_description)

parsed_dataset = raw_dataset.map(_parse_function)

# Initialize dictionary to collect all feature values
all_feature_values = {feature: [] for feature in feature_description.keys()}

# Collect values
for record in parsed_dataset:
    for feature in feature_description.keys():
        all_feature_values[feature].append(record[feature].numpy())

# statistics
stats_list = []

for feature_name, values in all_feature_values.items():
    values = np.concatenate(values, axis=0)

    mean = np.mean(values)
    std = np.std(values)
    min_val = np.min(values)
    max_val = np.max(values)

    stats_list.append({
        'Feature': feature_name,
        'Min': round(min_val, 4),
        'Max': round(max_val, 4),
        'Mean': round(mean, 4),
        'Std': round(std, 4)
    })

# Save to CSV
df = pd.DataFrame(stats_list)
csv_path = 'feature_statistics.csv'
df.to_csv(csv_path, index=False)

print(f"Feature statistics saved to {csv_path}")
'''
# Define feature schema (all 64x64 -> flattened 4096)
feature_description = {
    'tmmn': tf.io.FixedLenFeature([4096], tf.float32),
    'NDVI': tf.io.FixedLenFeature([4096], tf.float32),
    'FireMask': tf.io.FixedLenFeature([4096], tf.float32),
    'population': tf.io.FixedLenFeature([4096], tf.float32),
    'elevation': tf.io.FixedLenFeature([4096], tf.float32),
    'vs': tf.io.FixedLenFeature([4096], tf.float32),
    'pdsi': tf.io.FixedLenFeature([4096], tf.float32),
    'pr': tf.io.FixedLenFeature([4096], tf.float32),
    'tmmx': tf.io.FixedLenFeature([4096], tf.float32),
    'sph': tf.io.FixedLenFeature([4096], tf.float32),
    'th': tf.io.FixedLenFeature([4096], tf.float32),
    'PrevFireMask': tf.io.FixedLenFeature([4096], tf.float32),
    'erc': tf.io.FixedLenFeature([4096], tf.float32),
}

def _parse_function(example_proto):
    return tf.io.parse_single_example(example_proto, feature_description)

parsed_dataset = raw_dataset.map(_parse_function)

# Compute statistics
stats_list = []

for feature_name in feature_description.keys():
    feature_values = []

    # Collect all values for this feature
    for record in parsed_dataset:
        feature_values.append(record[feature_name].numpy())

    feature_values = np.concatenate(feature_values, axis=0)  # shape: (N*4096,)

    # Reshape to 64x64 if needed (optional)
    # feature_values = feature_values.reshape(-1, 64, 64)

    # Basic stats
    mean = np.mean(feature_values)
    std = np.std(feature_values)
    min_val = np.min(feature_values)
    max_val = np.max(feature_values)

    stat_entry = {
        'Feature': feature_name,
        'Count': feature_values.size,
        'Min': round(float(min_val), 4),
        'Max': round(float(max_val), 4),
        'Mean': round(float(mean), 4),
        'Std': round(float(std), 4),
    }

    # For FireMask features: compute fire pixel ratio (nonzero fraction)
    if feature_name in ['FireMask', 'PrevFireMask']:
        fire_ratio = np.mean(feature_values > 0)
        stat_entry['FirePixelRatio'] = round(float(fire_ratio), 4)

    stats_list.append(stat_entry)

# Save to CSV
df = pd.DataFrame(stats_list)
csv_path = 'feature_statistics.csv'
df.to_csv(csv_path, index=False)

print(f"Feature statistics saved to {csv_path}")