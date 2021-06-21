import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import plot_model
import autokeras as ak
import pandas as pd
from progressbar import progressbar

# def get_divisors(n):
#     for i in range(1, int(n / 2) + 1):
#         if n % i == 0:
#             yield i
#     yield n
#
# for i in get_divisors(46744):
#     print(i)

# def readCsv(datPath):
#     print("Reading csv: " + datPath)
#     df = pd.read_csv(datPath)
#     print("Loaded dataframe of shape [" + str(len(df.index)) + ", " + str(len(df.columns)) + "]\n")
#
#     return df
#
# # rows = list()
# # expPath = "C:/Dev/Projects/turbine/data/CCLE_expression.csv"
# # exp = readCsv(expPath)
# # cnPath = "C:/Dev/Projects/turbine/data/ds.csv"
# # cn = readCsv(cnPath)
# targPath = "C:/Dev/Projects/turbine/data/targets_parsed.csv"
# targh = readCsv(targPath)

# sns.displot(data=cn.iloc[:,0:5], binwidth=0.1)
# plt.show()



# l = lambda a: ((a)**2)*(np.sign(a)+0.5)**2
# c = lambda a: ((a)**2)
# A = np.arange(-3,3,.01)
# l_data = [l(x) for x in A]
# c_data = [c(x) for x in A]
#
# plt.figure(figsize=(20,10))
# plt.plot(A, l_data,'r')
# plt.plot(A, c_data,'b')
#
# plt.xlim(-3,3)
# plt.ylim(0,10)
# plt.show()


# model = load_model("model_autosearch1.h5", custom_objects=ak.CUSTOM_OBJECTS)
#
# print(model.summary())
# plot_model(
#     model,
#     to_file= "autokeras1" + ".png",
#     show_shapes=True,
#     show_layer_names=True,
#     rankdir="TB",
#     expand_nested=False,
#     dpi=96,
# )

