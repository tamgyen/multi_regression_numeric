from tensorflow.keras.layers import Input, Dense, Conv2D, Flatten, Concatenate, MaxPool1D, AveragePooling1D, \
    Dropout, BatchNormalization, MaxPooling1D, GlobalAveragePooling2D, UpSampling2D, Conv1D, LayerNormalization
from tensorflow.keras.models import Model
import tensorflow.keras.backend as bk
from tensorflow.keras import optimizers
from tensorflow.python.keras.utils.vis_utils import plot_model


# since we want to punish preds over 0 more (0 bcs dset is shifted +.5)
def closs(y_true, y_pred):
    loss = ((y_true - y_pred) ** 2) * (bk.sign(y_true - y_pred) + 0.5) ** 2
    return loss

def buildDenseAuto(INPUT_SHAPE, OPTIM, BNORM, DEPTH, WIDTH):
    input = Input(shape=INPUT_SHAPE, name='features')
    x = Dense(WIDTH, activation='relu')(input)
    if BNORM:
        x = BatchNormalization()(x)
    for i in range(0, DEPTH):
        x = Dense(WIDTH, activation='relu')(x)
    x = Dense(16, activation='relu')(x)
    backbone = x

    r1 = Dense(1, activation='linear', name='r1')(backbone)
    r2 = Dense(1, activation='linear', name='r2')(backbone)
    r3 = Dense(1, activation='linear', name='r3')(backbone)
    r4 = Dense(1, activation='linear', name='r4')(backbone)

    l1 = Dense(1, activation='sigmoid', name='l1')(backbone)
    l2 = Dense(1, activation='sigmoid', name='l2')(backbone)
    l3 = Dense(1, activation='sigmoid', name='l3')(backbone)
    l4 = Dense(1, activation='sigmoid', name='l4')(backbone)

    head = [r1, r2, r3, r4, l1, l2, l3, l4]

    model = Model(inputs=input, outputs=head)
    model.compile(optimizer=OPTIM,
                  loss={'r1': closs, 'r2': closs, 'r3': closs, 'r4': closs, 'l1': 'binary_crossentropy',
                        'l2': 'binary_crossentropy', 'l3': 'binary_crossentropy', 'l4': 'binary_crossentropy'},
                  loss_weights={'r1': 1, 'r2': 1, 'r3': 1, 'r4': 1, 'l1': .2, 'l2': .2, 'l3': .2, 'l4': .2},
                  metrics=['mae', 'accuracy'])

    print(model.summary())
    return model


def buildCnnMan(INPUT_SHAPE):
    input = Input(shape=INPUT_SHAPE, name='features')
    x = Conv1D(128, 5, strides=2, activation='relu', data_format='channels_first')(input)
    x = MaxPooling1D(3, data_format='channels_first')(x)
    x = Conv1D(64, 3, strides=2, activation='relu', data_format='channels_first')(x)
    x = MaxPooling1D(3, data_format='channels_first')(x)
    x = Conv1D(8, 3, strides=2, activation='relu', data_format='channels_first')(x)
    x = MaxPooling1D(3, data_format='channels_first')(x)
    x = Flatten()(x)
    x = Dense(32, activation='relu')(x)
    x = Dense(16, activation='relu')(x)
    backbone = x

    r1 = Dense(1, activation='linear', name='r1')(backbone)
    r2 = Dense(1, activation='linear', name='r2')(backbone)
    r3 = Dense(1, activation='linear', name='r3')(backbone)
    r4 = Dense(1, activation='linear', name='r4')(backbone)

    l1 = Dense(1, activation='sigmoid', name='l1')(backbone)
    l2 = Dense(1, activation='sigmoid', name='l2')(backbone)
    l3 = Dense(1, activation='sigmoid', name='l3')(backbone)
    l4 = Dense(1, activation='sigmoid', name='l4')(backbone)

    model = Model(inputs=input, outputs=[r1, r2, r3, r4, l1, l2, l3, l4])
    model.compile(optimizer='adam',
                  loss={'r1': closs, 'r2': closs, 'r3': closs, 'r4': closs, 'l1': 'binary_crossentropy',
                        'l2': 'binary_crossentropy', 'l3': 'binary_crossentropy', 'l4': 'binary_crossentropy'},
                  loss_weights={'r1': 1, 'r2': 1, 'r3': 1, 'r4': 1, 'l1': .2, 'l2': .2, 'l3': .2, 'l4': .2},
                  metrics=['mae', 'accuracy'])
    # print(model.summary())
    return model


def buildCnnAuto(INPUT_SHAPE, DEPTH_CNN, KCNT_CONV, KSIZE_CONV,
             KSIZE_POOL, STRIDE_CONV,BNORM):

    input = Input(shape=INPUT_SHAPE, name='features')

    for i in range(0, DEPTH_CNN):
        x = Conv1D(KCNT_CONV, KSIZE_CONV, strides=STRIDE_CONV, activation='relu', data_format='channels_first')(input)
        x = MaxPooling1D(KSIZE_POOL, data_format='channels_first')(x)
        if BNORM:
            x = BatchNormalization()(x)
    x = Flatten()(x)
    x = Dense(32, activation='relu')(x)
    x = Dense(16, activation='relu')(x)
    backbone = x

    r1 = Dense(1, activation='linear', name='r1')(backbone)
    r2 = Dense(1, activation='linear', name='r2')(backbone)
    r3 = Dense(1, activation='linear', name='r3')(backbone)
    r4 = Dense(1, activation='linear', name='r4')(backbone)

    l1 = Dense(1, activation='sigmoid', name='l1')(backbone)
    l2 = Dense(1, activation='sigmoid', name='l2')(backbone)
    l3 = Dense(1, activation='sigmoid', name='l3')(backbone)
    l4 = Dense(1, activation='sigmoid', name='l4')(backbone)

    model = Model(inputs=input, outputs=[r1, r2, r3, r4, l1, l2, l3, l4])
    model.compile(optimizer='adam',
                  loss={'r1': closs, 'r2': closs, 'r3': closs, 'r4': closs, 'l1': 'binary_crossentropy',
                        'l2': 'binary_crossentropy', 'l3': 'binary_crossentropy', 'l4': 'binary_crossentropy'},
                  loss_weights={'r1': 1, 'r2': 1, 'r3': 1, 'r4': 1, 'l1': .1, 'l2': .1, 'l3': .1, 'l4': .1},
                  metrics=['mae'])

    print(model.summary())
    return model

# model = buildCnnMan((1, 19182))
# plot_model(
#     model,
#     to_file= "dense_exp_7" + ".png",
#     show_shapes=True,
#     show_layer_names=True,
#     rankdir="TB",
#     expand_nested=False,
#     dpi=96,
# )
# print(model.summary())
