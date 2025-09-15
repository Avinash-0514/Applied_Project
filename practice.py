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
'''
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
    
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# Pretend we have 2 input features
INPUT_FEATURES = ["temp", "veg"]
PREV_FIRE = "prev_fire"
LABEL = "label"

def _normalize(x, name):
    return tf.cast(x, tf.float32) / 255.0

def _rescale(x, name):
    return tf.cast(x, tf.float32)

def _parse_fn_ensemble_simple_mock(parsed):
    # Prev fire mask
    prev_fire = _normalize(parsed[PREV_FIRE], PREV_FIRE)
    prev_fire = tf.expand_dims(prev_fire, -1)

    # Build inputs
    branch_inputs = []
    for feat in INPUT_FEATURES:
        feat_norm = _normalize(parsed[feat], feat)
        inp = tf.concat([tf.expand_dims(feat_norm, -1), prev_fire], axis=-1)
        branch_inputs.append(inp)

    # Label mask
    label = _rescale(parsed[LABEL], LABEL)
    label = tf.expand_dims(label, -1)
    return tuple(branch_inputs), label

# Create dummy example (64x64 grids)
example = {
    "temp": tf.constant(np.random.randint(0,255,(64,64)), dtype=tf.float32),
    "veg": tf.constant(np.random.randint(0,255,(64,64)), dtype=tf.float32),
    "prev_fire": tf.constant(np.random.randint(0,255,(64,64)), dtype=tf.float32),
    "label": tf.constant(np.random.choice([0,1], size=(64,64), p=[0.95,0.05]), dtype=tf.float32) # 5% fire pixels
}

(inputs, label) = _parse_fn_ensemble_simple_mock(example)

# Visualize
plt.figure(figsize=(12,4))

plt.subplot(1,3,1)
plt.title("Temp + PrevFire (branch1)")
plt.imshow(inputs[0][:,:,0], cmap="hot")
plt.colorbar()

plt.subplot(1,3,2)
plt.title("Prev Fire (extra channel)")
plt.imshow(inputs[0][:,:,1], cmap="gray")
plt.colorbar()

plt.subplot(1,3,3)
plt.title("Label (Fire vs No Fire)")
plt.imshow(label[:,:,0], cmap="Reds")
plt.colorbar()

plt.tight_layout()
plt.show()
'''
'''
#plot Single Features each
import tensorflow as tf
import matplotlib.pyplot as plt

# Path to your TFRecord
tfrecord_path = "/Volumes/StudyNProjects/UnitecFolder/Thesis_Project/WildFireDetection/archive/next_day_wildfire_spread_train_06.tfrecord"

# Define feature description for parsing
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

# Parsing function
def _parse_function(example_proto):
    example = tf.io.parse_single_example(example_proto, feature_description)
    # Reshape each 4096 vector into 64x64
    for k in example.keys():
        example[k] = tf.reshape(example[k], (64, 64))
    return example

# Load dataset
raw_dataset = tf.data.TFRecordDataset([tfrecord_path])
parsed_dataset = raw_dataset.map(_parse_function)

# Take one example
for record in parsed_dataset.take(1):
    example = record

# Features list
input_features = ['tmmn','NDVI','population','elevation','vs','pdsi','pr','tmmx','sph','th','erc']

# Setup plot
fig, axes = plt.subplots(3, 5, figsize=(15, 9))
axes = axes.flatten()

# Plot 11 features
for i, feat in enumerate(input_features):
    axes[i].imshow(example[feat], cmap='viridis')
    axes[i].set_title(feat)
    axes[i].axis('off')

# Plot PrevFireMask
axes[11].imshow(example['PrevFireMask'], cmap='gray')
axes[11].set_title("PrevFireMask")
axes[11].axis('off')

# Plot FireMask (Label)
axes[12].imshow(example['FireMask'], cmap='Reds')
axes[12].set_title("FireMask (Label)")
axes[12].axis('off')

plt.tight_layout()
plt.show()
'''

import glob
import tensorflow as tf
tfrecord_files = glob.glob("/Volumes/StudyNProjects/UnitecFolder/Thesis_Project/WildFireDetection/archive/*.tfrecord")

total_samples = 0
for file in tfrecord_files:
    ds = tf.data.TFRecordDataset([file])
    total_samples += sum(1 for _ in ds)

print(f"Total samples in all TFRecords: {total_samples}")