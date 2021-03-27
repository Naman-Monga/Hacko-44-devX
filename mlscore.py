# !pip install pycaret
# Pycaret Install kr lena and pandas too

import pandas as pd
dataset = pd.read_csv('C:/Users/Immortal Blue/PycharmProjects/Hacko-44-devX/automobile.csv')
# path of the file ^

columnNameReg = 'price'
# column name of the column on which to apply & train liner regerssion models

# ---------------------------------------------------
# --------------------REGRESSION---------------------
# ---------------------------------------------------
from pycaret.regression import *

exp_reg101 = setup(data = dataset, target = columnNameReg, data_split_shuffle=False)
cmp = compare_models()
storecmp = pull()
#print(storecmp)
# storecmp is the pandas DataFrame containing all the R2, MAE, MSE, etc. values for 17 reg models

cmplist = storecmp.values.tolist()
print(cmplist)
# cmplist contains the above df in list format please view the dataframe once to get to know which value in the list are which ones

# ---------------------------------------------------
# --------------------CLASSIFICATION-----------------
# ---------------------------------------------------

columnNameClass = 'body-style'
from pycaret.classification import *
clf1 = setup(data, target = columnNameClass, data_split_shuffle=False)
cmp2 = compare_models()
storecmp2 = pull()
#print(storecmp2)
# storecmp is the pandas DataFrame containing all the R2, MAE, MSE, etc. values for 14 classification models

cmplist2 = storecmp2.values.tolist()
print(cmplist2)
# cmplist contains the above df in list format please view the dataframe once to get to know which value in the list are which ones
