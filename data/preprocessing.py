import tensorflow as tf
from common import (prev_fire,prev_fire,curr_fire,wildfire_stats,BATCH_SIZE)
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
'''
def get_mkcnn_dataset(pattern_type, batch_size = BATCH_SIZE):
    
    dataset = tf.data.Dataset.list_files(pattern_type)
    dataset = dataset.interleave(tf.data.TFRecordDataset,cycle_length=4)
    parse_function = get_dynamic_parser(selected_features)
    dataset= dataset.map(parse_function, num_parallel_calls=tf.data.AUTOTUNE)
    dataset= dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    return dataset
'''
'''
Old
def get_mkcnn_dataset(pattern_type, batch_size=BATCH_SIZE, selected_features=None):
    if selected_features is None:
        raise ValueError("selected_features must be provided.")

    dataset = tf.data.Dataset.list_files(pattern_type)
    dataset = dataset.interleave(tf.data.TFRecordDataset, cycle_length=4)
    parse_function = get_dynamic_parser(selected_features)
    dataset = dataset.map(parse_function, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    return dataset
'''
def get_mkcnn_dataset(pattern_type,multi_input,prev_feat_append_flag, batch_size=BATCH_SIZE,selected_features=None):
    if selected_features is None:
        raise ValueError("selected_features must be provided.")

    dataset = tf.data.Dataset.list_files(pattern_type)
    dataset = dataset.interleave(tf.data.TFRecordDataset, cycle_length=4)
    parse_function = get_dynamic_parser(selected_features,prev_feat_append_flag, multi_input=multi_input)
    dataset = dataset.map(parse_function, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    return dataset

'''
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
'''
'''
old
def get_dynamic_parser(selected_features):
    def _mkcnn_parse_function_ensemble(example_proto):
        wildFire_features = selected_features + [prev_fire, curr_fire]
        feature_description = {
            f: tf.io.FixedLenFeature([64, 64], tf.float32) for f in wildFire_features
        }
        parsed = tf.io.parse_single_example(example_proto, feature_description)

        # Normalize previous fire mask
        prev_fire_input = _normalise(parsed[prev_fire], prev_fire)
        prev_fire_input = tf.expand_dims(prev_fire_input, -1)

        # Stack each selected feature with previous fire input
        selected_branch_inputs = []
        for feat in selected_features:
            norm_feat = _normalise(parsed[feat], feat)
            feat_input = tf.expand_dims(norm_feat, -1)
            combined_input = tf.concat([feat_input, prev_fire_input], axis=-1)
            selected_branch_inputs.append(combined_input)

        # Prepare label (current fire mask)
        label = _rescale(parsed[curr_fire], curr_fire)
        label = tf.expand_dims(label, -1)

        return tuple(selected_branch_inputs), label

    return _mkcnn_parse_function_ensemble

'''
def get_dynamic_parser(selected_features,prev_feat_append_flag, multi_input=False, prev_fire="prev_fire", curr_fire="curr_fire"):
    def _mkcnn_parse_function_ensemble(example_proto):

        prev_fire = "PrevFireMask"
        curr_fire = "FireMask"
        # Add prev_fire and curr_fire to the feature set
        wildFire_features = selected_features + [prev_fire, curr_fire]

        # TFRecord feature description
        feature_description = {
            f: tf.io.FixedLenFeature([64, 64], tf.float32) for f in wildFire_features
        }
        parsed = tf.io.parse_single_example(example_proto, feature_description)

        # Normalize previous fire mask
        prev_fire_input = _normalise(parsed[prev_fire], prev_fire)
        prev_fire_input = tf.expand_dims(prev_fire_input, -1)

        # Process features
        branch_inputs = []
        #prev_feat_append_flag = True
        if prev_feat_append_flag:

            for feat in selected_features:
                # Scenario 1: Concat Previous fire with every Feature branch(2* feature Channels)
                norm_feat = _normalise(parsed[feat], feat)
                feat_input = tf.expand_dims(norm_feat, -1)

                # Each branch: feature + prev_fire
                combined_input = tf.concat([feat_input, prev_fire_input], axis=-1)
                branch_inputs.append(combined_input)
        else:
            for feat in selected_features:
                # Scenario 2: keeping Previous Fire as a separate Branch(Feature + 1 Extra Branch)
                norm_feat = _normalise(parsed[feat],feat)
                feat_input = tf.expand_dims(norm_feat,-1)
                branch_inputs.append(feat_input)
            branch_inputs.append(prev_fire_input)

        # Label (current fire mask)
        label = _rescale(parsed[curr_fire], curr_fire)
        label = tf.expand_dims(label, -1)

        # Return depending on mode
        if multi_input:
            # Multiple inputs (tuple of tensors)
            return tuple(branch_inputs), label
        else:
            # Single input (stack all feature branches into one tensor)
            final_input = tf.concat(branch_inputs, axis=-1)
            return final_input, label

    return _mkcnn_parse_function_ensemble




'''
def dataset_split_function(training, testing, validation,batch_size = BATCH_SIZE, isFire = False):
    
    def _sep_fire_filter(tuple_input_values, label):
            cnt = tf.reduce_sum(tf.cast(label > 0.5, tf.int32))
            return cnt > 50
    
    dataset_fire = get_mkcnn_dataset(training,batch_size=batch_size)
    dataset_non_fire = get_mkcnn_dataset(training, batch_size=batch_size)
    if isFire:
            fire_ds = fire_ds.filter(_sep_fire_filter)
    dataset_fire = dataset_fire.take(700// batch_size)
    dataset_non_fire = dataset_non_fire.take(300// batch_size)
    training_dataset = dataset_fire.concatenate(dataset_non_fire).shuffle(1000//batch_size).repeat()
    validation_dataset = get_mkcnn_dataset(validation,batch_size=batch_size)
    testing_dataset = get_mkcnn_dataset(testing,batch_size=batch_size)
    return training_dataset, validation_dataset,testing_dataset
'''
def dataset_split_function(training, testing, validation,prev_feat_append_flag, batch_size,selected_features, multiple_input,isFire=False):
    if selected_features is None:
        raise ValueError("You must pass selected_features.")

    def _sep_fire_filter(inputs, label):
        cnt = tf.reduce_sum(tf.cast(label > 0.5, tf.int32))
        return cnt > 50

    dataset_fire = get_mkcnn_dataset(training, multiple_input,prev_feat_append_flag,batch_size=batch_size,selected_features=selected_features)
    dataset_non_fire = get_mkcnn_dataset(training, multiple_input,prev_feat_append_flag,batch_size=batch_size, selected_features=selected_features)

    if isFire:
        dataset_fire = dataset_fire.filter(_sep_fire_filter)

    dataset_fire = dataset_fire.take(700 // batch_size)
    dataset_non_fire = dataset_non_fire.take(300 // batch_size)

    training_dataset = dataset_fire.concatenate(dataset_non_fire).shuffle(1000).repeat()
    validation_dataset = get_mkcnn_dataset(validation, multiple_input,prev_feat_append_flag,batch_size=batch_size, selected_features=selected_features)
    testing_dataset = get_mkcnn_dataset(testing, multiple_input,prev_feat_append_flag,batch_size=batch_size, selected_features=selected_features)

    return training_dataset, validation_dataset, testing_dataset

