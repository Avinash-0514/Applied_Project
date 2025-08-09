import tensorflow as tf
from tensorflow.keras import layers, Model, Input, backend as K
#from data.preprocessing import(get_mkcnn_dataset, get_dynamic_parser,dataset_split_function,input_features)
from common import(input_features)

def build_multi_kernel_cnn_model(num_input_channels,loss_type='wbce'):
    inp = Input(shape=(64,64, num_input_channels), name="mkc_input")
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

def iou_metric(y_true, y_pred, smooth=1e-6):
    y_true = tf.reshape(y_true, tf.shape(y_pred))
    y_true = tf.reshape(y_true, [-1])
    y_pred = tf.reshape(y_pred, [-1])
    inter = tf.reduce_sum(y_true * y_pred)
    union = tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) - inter
    return (inter + smooth) / (union + smooth)

def weighted_bce(y_true, y_pred):
    y_true = tf.reshape(y_true, tf.shape(y_pred))
    y_pred = tf.clip_by_value(y_pred, 1e-7, 1 - 1e-7)
    bce = K.binary_crossentropy(y_true, y_pred)
    weights = 1 + (20.0 - 1) * y_true
    return tf.reduce_mean(bce * weights)

def focal_loss(gamma=2.0):
    def loss(y_true, y_pred):
        y_true = tf.reshape(y_true, tf.shape(y_pred))
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1 - 1e-7)
        return tf.reduce_mean(-(y_true * (1 - y_pred) ** gamma * tf.math.log(y_pred) +
                                 (1 - y_true) * y_pred ** gamma * tf.math.log(1 - y_pred)))
    return loss
