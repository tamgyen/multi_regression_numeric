import pandas as pd
from sklearn.model_selection import train_test_split
import autokeras as ak
import tensorflow as tf
import numpy as np
from sklearn import preprocessing

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  try:
    tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
    logical_gpus = tf.config.experimental.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPU")
  except RuntimeError as e:
    print(e)


# CHANGED TO ONLY CN DS!!!
data = pd.read_csv("C:/Dev/Projects/turbine/data/ds_cn.csv")
data.drop(columns='DepMap_ID', axis= 1, inplace=True)


targets = data.iloc[:, 0:4]
features = data.iloc[:, 4:]
print(features.shape, targets.shape)

Y = targets.to_numpy()
Y = Y.astype('float32')
X = features.to_numpy()
X = X.astype('float32')


X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.1, random_state=1)
print(X_train.shape, X_test.shape, Y_train.shape, Y_test.shape)

search = ak.StructuredDataRegressor(max_trials=6, loss='mse')

search.fit(x=X_train, y=Y_train, epochs=25, verbose=1)

mae, _ = search.evaluate(X_test, Y_test, verbose=1)

print('MAE: %.3f' % mae)

model = search.export_model()
model.summary()
model.save('model_autosearch1.h5')