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
#requried imports

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn import metrics
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
 
import warnings
warnings.filterwarnings('ignore')


# %%
#importing data

df = pd.read_csv('./winequality/winequality-red.csv')
#print(df.head())

len(df)


# %%
row_sums = df.sum(axis=1)
row_means = row_sums / df.shape[1]

#row_means.head()
df['total_score'] = row_means
df.head()

# %%
#investigation of dataset using presets

#df.info()
df.describe().T

# %%

# %%
#removing all null values from the set

#df.isnull().sum()

for col in df.columns:
  if df[col].isnull().sum() > 0:
    df[col] = df[col].fillna(df[col].mean())
 
df.isnull().sum().sum()

# %%
#histrogram to visualize the data in  a continuous format for ea. trait

df.hist(bins=20, figsize=(10, 10))
plt.show()


# %%

plt.bar(df['quality'], df['residual sugar'])
plt.xlabel('quality')
plt.ylabel('alcohol')
plt.show()


# %%
plt.bar(df['quality'], df['total_score'])
plt.xlabel('quality')
plt.ylabel('total score')
plt.show()

# %%
plt.figure(figsize=(12, 12))
sb.heatmap(df.corr() > 0.70, annot=True, cbar=False)
plt.show()


# %%

df = df.drop('total sulfur dioxide', axis=1)
df = df.drop('citric acid', axis=1)
df = df.drop('density', axis=1)



# %%
#prepparing datasets for training
df['best quality'] = [1 if x > 6 else 0 for x in df.quality]


features = df.drop(['quality', 'best quality'], axis=1)
target = df['best quality']
 
xtrain, xtest, ytrain, ytest = train_test_split(
    features, target, test_size=0.2, random_state=40)
 
xtrain.shape, xtest.shape


# %%
# normalizing data

norm = MinMaxScaler()
xtrain = norm.fit_transform(xtrain)
xtest = norm.transform(xtest)



# %%

models = [LogisticRegression(), XGBClassifier(), SVC(kernel='rbf')]
 
for i in range(3):
    models[i].fit(xtrain, ytrain)
 
    print(f'{models[i]} : ')
    print('Training Accuracy : ', metrics.roc_auc_score(ytrain, models[i].predict(xtrain)))
    print('Validation Accuracy : ', metrics.roc_auc_score(
        ytest, models[i].predict(xtest)))
    print()


# %%
from sklearn.svm import SVC

clf = SVC(random_state=0)
clf.fit(xtrain, ytrain)

predictions = clf.predict(xtest)
cm = confusion_matrix(ytest, predictions, labels=clf.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                              display_labels=clf.classes_)
disp.plot()

plt.show()

# %%
from sklearn.svm import SVC

clf = SVC(random_state=0)
clf.fit(xtest, ytest)

predictions = clf.predict(xtrain)
cm = confusion_matrix(ytrain, predictions, labels=clf.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                              display_labels=clf.classes_)
disp.plot()

plt.show()

# %%


print(metrics.classification_report(ytest,
                                    models[0].predict(xtest)))


# %%
for i in range(3):
    print(models[i])
    print(metrics.classification_report(ytest,
                                    models[i].predict(xtest)))

# %%

# %%
#importing data

df2 = pd.read_csv('./winequality/winequality-white2.csv')
#print(df2.head())

df2.describe().T


# %%
df2.isnull().sum()


# for col in df.columns:
#   if df[col].isnull().sum() > 0:
#     df[col] = df[col].fillna(df[col].mean())
 
# df.isnull().sum().sum()

# %%
df2.hist(bins=20, figsize=(10, 10), color='pink')
plt.show()

# %%
plt.figure(figsize=(12, 12))
sb.heatmap(df2.corr() > 0.6, annot=True, cbar=False)
plt.show()

# %%
df2['best quality'] = [1 if x > 6 else 0 for x in df2.quality]


features = df2.drop(['quality', 'best quality'], axis=1)
target = df2['best quality']
 
xtrain2, xtest2, ytrain2, ytest2 = train_test_split(
    features, target, test_size=0.2, random_state=40)
 
xtrain2.shape, xtest2.shape

# %%
norm = MinMaxScaler()
xtrain2 = norm.fit_transform(xtrain2)
xtest2 = norm.transform(xtest2)


# %%
clf = SVC(random_state=0)
clf.fit(xtrain2, ytrain2)

predictions2 = clf.predict(xtest2)
cm2 = confusion_matrix(ytest2, predictions2, labels=clf.classes_)
disp2 = ConfusionMatrixDisplay(confusion_matrix=cm2,
                              display_labels=clf.classes_)
disp2.plot()

plt.show()

# %%
