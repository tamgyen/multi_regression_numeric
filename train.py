import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow.keras.backend as bk
import modelBuilder as mb

print("Using TensorFlow version %s" % tf.__version__)

# SETTINGS:
SPLITS = np.array([.8, .1, .1])
BATCH_SIZE = 64
BUFFER_SIZE = 100
MAX_EPOCH = 2000
CB_PATIENCE = 20

DSETS = ["ds_cn", "ds_exp", "ds"]

DEPTHS = [1, 2, 3]
WIDTHS = [2, 8, 32, 128, 512]
BNORMS = [0, 1]

modelNameBase ="dense_combined_"

# parse csv to df
data = pd.read_csv("C:/Dev/Projects/turbine/data/" + DSETS[1] + ".csv")
data.drop(columns='DepMap_ID', axis=1, inplace=True)

# shuffle for split and split
data = data.sample(frac=1).reset_index(drop=True)
splitVals = (SPLITS * len(data.index)).astype(int)
dataTrain = data.iloc[0:splitVals[0], :]
dataVal = data.iloc[splitVals[0]:splitVals[0] + splitVals[1], :]
dataTest = data.iloc[splitVals[1] + splitVals[0]:len(data.index), :]

outShape = data.iloc[:, 0:8].shape[1]
inpShape = data.iloc[:, 8:].shape[1]

# init datasets
datasetTrain = tf.data.Dataset.from_tensor_slices((dataTrain.iloc[:, 0:8], dataTrain.iloc[:, 8:]))
datasetVal = tf.data.Dataset.from_tensor_slices((dataVal.iloc[:, 0:8], dataVal.iloc[:, 8:]))
datasetTest = tf.data.Dataset.from_tensor_slices((dataTest.iloc[:, 0:8], dataTest.iloc[:, 8:]))
dataset = {"train": datasetTrain, "val": datasetVal, "test": datasetTest}


# define loader
@tf.function
def loadDataVector(targ, feat):
    r1, r2, r3, r4, l1, l2, l3, l4 = tf.split(targ, num_or_size_splits=8)
    targets = {'r1': r1, 'r2': r2, 'r3': r3, 'r4': r4, 'l1': l1, 'l2': l2, 'l3': l3, 'l4': l4}
    feat = tf.reshape(feat, [1, inpShape])
    features = {'features': feat}

    return features, targets


# init pipeline
dataset["train"] = dataset["train"].map(loadDataVector, num_parallel_calls=tf.data.experimental.AUTOTUNE)
dataset['train'] = dataset['train'].shuffle(buffer_size=BUFFER_SIZE)
dataset['train'] = dataset['train'].repeat()
dataset['train'] = dataset['train'].batch(BATCH_SIZE)
dataset["val"] = dataset["val"].map(loadDataVector, num_parallel_calls=tf.data.experimental.AUTOTUNE)
dataset['val'] = dataset['val'].repeat()
dataset['val'] = dataset['val'].batch(BATCH_SIZE)
dataset["test"] = dataset["test"].map(loadDataVector, num_parallel_calls=tf.data.experimental.AUTOTUNE)
dataset['test'] = dataset['test'].batch(BATCH_SIZE)

# for tb logs
# logdir = os.path.join("logs", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
# tensorboard_callback = tf.keras.callbacks.TensorBoard(logdir, histogram_freq=1)

callbacks = [
    # tensorboard_callback,
    tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=CB_PATIENCE, verbose=1),
]

# to check dataset integrity
# for element in dataset["train"].take(1):
#     print(element)

stepsTrain = splitVals[0] // BATCH_SIZE
stepsVal = splitVals[1] // BATCH_SIZE

# arrays for logging
errors = list()
maes = list()
configs = list()
modelNames = list()

modelID = 0
# MAIN TRAIN-TEST LOOP HERE:
for BNORM in BNORMS:
    for DEPTH in DEPTHS:
        for WIDTH in WIDTHS:
            modelID += 1
            modelName = modelNameBase + str(modelID)
            modelNames.append(modelName)
            print("\ntraining " + modelName)

            model = mb.buildDenseAuto(INPUT_SHAPE=(1, inpShape),
                                      OPTIM='adam',
                                      BNORM=BNORM,
                                      DEPTH=DEPTH,
                                      WIDTH=WIDTH)

            history = model.fit(dataset["train"],
                                validation_data=dataset["val"],
                                steps_per_epoch=stepsTrain,
                                validation_steps=stepsVal,
                                epochs=MAX_EPOCH,
                                callbacks=callbacks,
                                verbose=1)

            error = model.evaluate(dataset["test"], verbose=0)
            errors.append(error)
            mae = (error[9]+error[11]+error[13]+error[15])/4
            maes.append(mae)
            config = [BNORM, DEPTH, WIDTH]
            configs.append(config)

            fig = plt.figure()
            plt.plot(history.history['loss'])
            plt.plot(history.history['val_loss'])
            plt.title(modelName + ' test_mae: ' + '%.4f' % mae)
            plt.ylabel('loss')
            plt.xlabel('Epoch')
            plt.legend(['Train', 'val'], loc='upper right')
            plt.savefig("./test_results/" + modelName + ".png")
            plt.close(fig)

            # to prevent mem leak from functional API model storage
            bk.clear_session()

print("main loop end :)")

testData = pd.concat([pd.DataFrame(data=modelNames, columns=['modelID']),
                      pd.DataFrame(data=configs, columns=['bnorm', 'depth', 'width']),
                      pd.DataFrame(data=maes, columns=['mae']),
                      pd.DataFrame(data=errors, columns=['loss', 'r1_loss', 'r2_loss', 'r3_loss', 'r4_loss', '11_loss',
                                                         'l2_loss', 'l3_loss', 'l4_loss',
                                                         'r1_mae', 'r1_acc', 'r2_mae', 'r2_acc', 'r3_mae', 'r3_acc',
                                                         'r4_mae', 'r4_acc', 'l1_mae', 'l1_acc', 'l2_mae',
                                                         'l2_acc', 'l3_mae', 'l3_acc', 'l4_mae', 'l4_acc', ])], axis=1)
testData.to_csv("./test_results/"+modelNameBase+".csv", index=None, mode='a')




# FOR TRAINING MANUALLY BUILT CNNS CNNs
# model = mb.buildCnnMan(INPUT_SHAPE=(1, inpShape))
#
# history = model.fit(dataset["train"],
#                             validation_data=dataset["val"],
#                             steps_per_epoch=stepsTrain,
#                             validation_steps=stepsVal,
#                             epochs=MAX_EPOCH,
#                             callbacks=callbacks,
#                             verbose=1)
#
# error = model.evaluate(dataset["test"], verbose=0)
# print((error[9]+error[11]+error[13]+error[15])/4)
#
# modelName= "conv1"
