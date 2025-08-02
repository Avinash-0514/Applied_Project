import tensorflow as tf
from tensorflow.keras import layers, Model
from data.preprocessing import(get_mkcnn_dataset, get_dynamic_parser,dataset_split_function,input_features)


def build_multi_kernel_cnn_model(loss_type='wbce'):
    inp = Input(shape=(64,64, len(input_features)+1), name="mkc_input")
    # three parallel conv paths
    c3 = layers.Conv2D(32, 3, padding='same', activation='relu')(inp)
    c5 = layers.Conv2D(32, 5, padding='same', activation='relu')(inp)
    c7 = layers.Conv2D(32, 7, padding='same', activation='relu')(inp)
    x = layers.Concatenate()([c3,c5,c7])
    x = layers.MaxPooling2D()(x)

    x = layers.Conv2D(64, 3, padding='same', activation='relu')(x)
    x = layers.UpSampling2D()(x)
    x = layers.Conv2D(32, 3, padding='same', activation='relu')(x)
    x = layers.Conv2D(1, 1, activation='sigmoid')(x)

    model = Model(inp, x)
    loss_fn = weighted_bce if loss_type=='wbce' else focal_loss()
    model.compile(optimizer='adam',
                  loss=loss_fn,
                  metrics=[tf.keras.metrics.AUC(),
                           tf.keras.metrics.Precision(),
                           tf.keras.metrics.Recall(),
                           tf.keras.metrics.BinaryAccuracy(),
                           iou_metric])
    return model