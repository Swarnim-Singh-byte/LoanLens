#!/usr/bin/env python
# coding: utf-8

# In[16]:


import pandas as pd
import numpy as np
import warnings # control warning messages because these libraries give lots of warnings which are not actually the error and only end up cluttering our notebook.

# now to ignore the warnings
warnings.filterwarnings('ignore')

df = pd.read_csv("../data/application_train_clean.csv")
print(df.shape)
df.head()


# In[17]:


df['AGE_YEARS'] = (-df['DAYS_BIRTH']) / 365

print(df['AGE_YEARS'].describe())


# In[18]:


df['EMPLOYMENT_YEARS'] = (-df['DAYS_EMPLOYED']) / 365

print(df['EMPLOYMENT_YEARS'].describe())


# In[19]:


df['CREDIT_INCOME_RATIO'] = df['AMT_CREDIT'] / df['AMT_INCOME_TOTAL']

print(df['CREDIT_INCOME_RATIO'].describe())


# In[20]:


df['ANNUITY_INCOME_RATIO'] = df['AMT_ANNUITY'] / df['AMT_INCOME_TOTAL']

print(df['ANNUITY_INCOME_RATIO'].describe())


# In[21]:


df['HAS_HOUSING_DATA'] = df['APARTMENTS_AVG'].notnull().astype(int)

print(df['HAS_HOUSING_DATA'].value_counts())


# In[22]:


df['HAS_BUREAU_INQUIRY_DATA'] = df['AMT_REQ_CREDIT_BUREAU_YEAR'].notnull().astype(int)

print(df['HAS_BUREAU_INQUIRY_DATA'].value_counts())


# In[23]:


df['EXT_SOURCE_1_WAS_MISSING'] = df['EXT_SOURCE_1'].isnull().astype(int)
df['EXT_SOURCE_3_WAS_MISSING'] = df['EXT_SOURCE_3'].isnull().astype(int)

print(df['EXT_SOURCE_1_WAS_MISSING'].value_counts())
print(df['EXT_SOURCE_3_WAS_MISSING'].value_counts())


# In[24]:


print(f"Original shape: (307511, 123)")
print(f"New shape: {df.shape}")
print(f"New columns added: {df.shape[1] - 123}")


# In[25]:


df.to_csv("../data/application_train_features.csv", index=False)
print("Exported successfully")
print(df.shape)


# # NOW Let's see basline model with feature engneering.
# 

# In[26]:


from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

X = df.drop(columns="TARGET")
y = df["TARGET"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("Training set:", X_train.shape)
print("Testing set :", X_test.shape)


# In[27]:


numerical_features = X_train.select_dtypes(include=["int64", "float64", "bool"]).columns.tolist()
categorical_features = X_train.select_dtypes(include=["object"]).columns.tolist()

numerical_pipeline = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_pipeline = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer(transformers=[
    ("num", numerical_pipeline, numerical_features),
    ("cat", categorical_pipeline, categorical_features)
])

preprocessor.fit(X_train)

X_train_processed = preprocessor.transform(X_train)
X_test_processed = preprocessor.transform(X_test)

print("Processed Training Shape:", X_train_processed.shape)
print("Processed Testing Shape :", X_test_processed.shape)


# In[28]:


log_reg_features = LogisticRegression(max_iter=1000, random_state=42, class_weight="balanced")
log_reg_features.fit(X_train_processed, y_train)

y_pred_features = log_reg_features.predict(X_test_processed)
y_prob_features = log_reg_features.predict_proba(X_test_processed)[:, 1]

print("Accuracy :", accuracy_score(y_test, y_pred_features))
print("Precision:", precision_score(y_test, y_pred_features))
print("Recall   :", recall_score(y_test, y_pred_features))
print("F1 Score :", f1_score(y_test, y_pred_features))
print("ROC AUC  :", roc_auc_score(y_test, y_prob_features))


# In[ ]:




