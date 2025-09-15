import tensorflow as tf
from tensorflow.keras import layers, Model, Input, backend as K
#from data.preprocessing import(get_mkcnn_dataset, get_dynamic_parser,dataset_split_function,input_features)
from common import(input_features)


def mkcnn_model(num_input_channels,
                num_layers=3,               # how many stacked MK blocks
                base_filters=32,            # first layer filters
                kernel_sizes=[3,5,7],
                loss_type='wbce'):

    inp = Input(shape=(64,64,num_input_channels), name="mkc_input")
    x = inp

    # 🔸 Encoder with increasing filters
    for layer_idx in range(num_layers):
        filters = base_filters * (2**layer_idx)   # 32, 64, 128, ...
        convs = []
        for k in kernel_sizes:
            conv = layers.Conv2D(filters, k, padding='same', activation='relu')(x)
            convs.append(conv)
        x = layers.Concatenate(name=f"concat_L{layer_idx+1}")(convs)
        x = layers.MaxPooling2D(name=f"pool_L{layer_idx+1}")(x)

    # Bottleneck
    filters = base_filters * (2**num_layers)      # next level filters
    x = layers.Conv2D(filters, 3, padding='same', activation='relu', name="bottleneck")(x)

    # Decoder (simple version: just one upsample)
    x = layers.UpSampling2D(name="upsample")(x)
    x = layers.Conv2D(base_filters, 3, padding='same', activation='relu', name="refine")(x)

    # Output mask
    out = layers.Conv2D(1, 1, activation='sigmoid', name="mask_output")(x)

    model = Model(inp, out, name="MKCNN")
    return model
    

    
'''
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
    '''