from keras import Model
from keras.layers import Dense, Dropout, GlobalAveragePooling2D, BatchNormalization, Activation
from keras import regularizers

weight_decay = 0.1


def tune_model(base_model):
    for layer in base_model.layers:
        layer.trainable = False

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    # x = Dense(10, activation='sigmoid', kernel_regularizer=regularizers.l2(weight_decay))(x)
    predictions = Dense(1, activation='sigmoid', kernel_regularizer=regularizers.l2(weight_decay))(x)

    return Model(inputs=base_model.input, outputs=predictions)
