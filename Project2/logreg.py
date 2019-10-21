import numpy as np
import sklearn.linear_model as skl
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
import seaborn as sns
import tensorflow as tf
import pandas as pd
import os
from tabulate import tabulate


np.random.seed(0)

cwd = os.getcwd()   # Current working directory
filename = cwd + '/data/default of credit card clients.xls'
nanDict = {}    # To store NaN from CC data when reading with pandas?
df = pd.read_excel(filename, header=1, index_col=0,
                   skiprows=0, na_values=nanDict)  # Dataframe

# Renaming axis
df.rename(index=str,
          columns={df.columns[-1]: 'DefaultPaymentNextMonth'},
          inplace=True)

# Features and targets
X = (df.loc[:, df.columns != 'DefaultPaymentNextMonth']).values   # All vals !=
y = (df.loc[:, df.columns == 'DefaultPaymentNextMonth']).values   # All vals ==

print(df.columns)

# Train-test split
test_size = 0.2
Xtest, Xtrain, ytest, ytrain = train_test_split(
    X, y, train_size=1 - test_size, test_size=test_size)

# Scaling
scale = StandardScaler()   # Scales by (func - mean)/std.dev
Xtrain = scale.transform(Xtrain)
Xtest = scale.transform(Xtest)

df = df.drop(df[(df.BILL_AMT1 == 0) &
                (df.BILL_AMT2 == 0) &
                (df.BILL_AMT3 == 0) &
                (df.BILL_AMT4 == 0) &
                (df.BILL_AMT5 == 0) &
                (df.BILL_AMT6 == 0)].index)

df = df.drop(df[(df.PAY_AMT1 == 0) &
                (df.PAY_AMT2 == 0) &
                (df.PAY_AMT3 == 0) &
                (df.PAY_AMT4 == 0) &
                (df.PAY_AMT5 == 0) &
                (df.PAY_AMT6 == 0)].index)
