# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.2
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %%

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn import metrics
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
 
import warnings
warnings.filterwarnings('ignore')


# %%

df = pd.read_csv('./winequality/winequality-red.csv')
print(df.head())


# %%
df.info()

# %%
df.describe().T

# %%
#df.isnull().sum()

for col in df.columns:
  if df[col].isnull().sum() > 0:
    df[col] = df[col].fillna(df[col].mean())
 
df.isnull().sum().sum()

# %%
df.hist(bins=20, figsize=(10, 10))
plt.show()


# %%

plt.bar(df['quality'], df['alcohol'])
plt.xlabel('quality')
plt.ylabel('alcohol')
plt.show()


# %%
plt.figure(figsize=(12, 12))
sb.heatmap(df.corr() > 0.65, annot=True, cbar=False)
plt.show()


# %%

df['best quality'] = [1 if x > 5 else 0 for x in df.quality]


features = df.drop(['quality', 'best quality'], axis=1)
target = df['best quality']
 
xtrain, xtest, ytrain, ytest = train_test_split(
    features, target, test_size=0.2, random_state=40)
 
xtrain.shape, xtest.shape


# %%

norm = MinMaxScaler()
xtrain = norm.fit_transform(xtrain)
xtest = norm.transform(xtest)



# %%
metrics.ConfusionMatrixDisplay(models[1], xtest)
plt.show()


# %%


print(metrics.classification_report(ytest,
                                    models[1].predict(xtest)))


# %%
