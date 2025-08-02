import tensorflow as tf



from google.colab import drive
drive.mount('/content/drive')

#Dataset 
training ='https://drive.google.com/drive/folders/122m2iIPFwSUPCquhp6BprdnulBjiRWGN?usp=drive_link'
testing = 'https://drive.google.com/drive/folders/1QQBxvlvfRtJ279xT-4_bffO8YkwLxD8K?usp=drive_link'
validation ='https://drive.google.com/drive/folders/1urhY51whQ-OIqmIVaufVyeOZ7rkhuGWA?usp=drive_link'

#Batch Configuration
BATCH_SIZE =0
EPOCHS =0

# Input Features Names
input_features = ['tmmn','NDVI','FireMask','population','elevation','vs','pdsi','pr','tmmx','sph','th','PrevFireMask','erc']
prev_fire = 'PrevFireMask'
curr_fire = 'FireMask'
selected_features = ''

# Feature Stats like Min, Max, Mean and std
wildfire_stats = {
    'tmmn': (-444.693, 716.6276, 281.8673, 18.0986),
    'NDVI': (-9567.0, 9966.0, 5297.4717, 2186.9038),
    'FireMask': (-1.0, 1.0, -0.0128, 0.1868),
    'population': (0.0, 27103.605, 29.8874, 214.3112),
    'elevation': (-45.0, 4203.0, 904.5699, 846.5071),
    'vs': (-82.6531, 103.2201, 3.6543, 1.3117),
    'pdsi': (-152.9079, 80.9965, -0.74, 2.4769),
    'pr': (-167.4483, 136.8156, 0.3348, 1.5888),
    'tmmx': (0.0, 1229.8488, 297.742, 19.0823),
    'sph': (-0.129, 0.0855, 0.0065, 0.0037),
    'th': (-505870.1, 37735.63, 154.127, 3163.1426),
    'PrevFireMask': (-1.0, 1.0, -0.0029, 0.1408),
    'erc': (-1196.0886, 2470.8823, 53.6251, 25.2632)
}

# Normalisation Function and Rescaling functions

def _normalise(x,key):
    minimum_val, maximum_val, mean, standard_dev = wildfire_stats[key]
    x = tf.clip_by_value(x,minimum_val, maximum_val)
    return (x-mean)/standard_dev

def _rescale(x, key):
    minimum_val, maximum_val, *_ = wildfire_stats[key]
    x = tf.clip_by_value(x, minimum_val, maximum_val)
    return (x - minimum_val) / (maximum_val - minimum_val)

# Dataset preprocessing functions

def get_mkcnn_dataset(pattern_type, batch_size = BATCH_SIZE):
    
    dataset = tf.data.Dataset.list_files(pattern_type)
    dataset = dataset.interleave(tf.data.TFRecordDataset,cycle_length=4)
    parse_function = get_dynamic_parser(selected_features)
    dataset= dataset.map(parse_function, num_parallel_calls=tf.data.AUTOTUNE)
    dataset= dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    return dataset

def get_dynamic_parser(selected_features):

    def _mkcnn_parse_function_ensemble(example_proto):
        
        wildFire_feature = [input_features]+[prev_fire,curr_fire]
        wildFire_ft_disc = {f: tf.io.FixedLenFeature([64,64], tf.float32) for f in wildFire_feature}
        wildFire_ft_parse = tf.io.parse_single_example(example_proto,wildFire_ft_disc)
        
        # Normalise previous Fire Mask    
        prev_fire_input = _normalise(wildFire_ft_parse[prev_fire],prev_fire)
        prev_fire_input = tf.expand_dims(prev_fire, -1)

        selected_branch_inputs=[]
        for features in selected_features:
            feature_Normalise = _normalise(wildFire_ft_parse[features],features)
            inputs_feat = tf.concat([tf.expand_dims(feature_Normalise,-1),prev_fire_input],axis=-1)
            selected_branch_inputs.append(inputs_feat)
        
        label = _rescale(wildFire_ft_parse[curr_fire],curr_fire)
        label = tf.expand_dims(label,-1)

        return tuple (selected_branch_inputs),label
 
    return _mkcnn_parse_function_ensemble

def dataset_split_function(training, testing, validation,batch_size = BATCH_SIZE, isFire = False):

    def _sep_fire_filter(tuple_input_values, label):
            cnt = tf.reduce_sum(tf.cast(label > 0.5, tf.int32))
            return cnt > 50
    
    dataset_fire = get_mkcnn_dataset(train_pt,batch_size=batch_size)
    dataset_non_fire = get_mkcnn_dataset(train_pt, batch_size=batch_size)
    if isFire:
            fire_ds = fire_ds.filter(_sep_fire_filter)
    dataset_fire = dataset_fire.take(700// batch_size)
    dataset_non_fire = dataset_non_fire.take(300// batch_size)
    training_dataset = dataset_fire.concatenate(dataset_non_fire).shuffle(1000//batch_size).repeat()
    validation_dataset = get_mkcnn_dataset(validation_pt,batch_size=batch_size)
    testing_dataset = get_mkcnn_dataset(testing_pt,batch_size=batch_size)
    return training_dataset, validation_dataset,testing_dataset
