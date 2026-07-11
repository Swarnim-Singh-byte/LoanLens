#!/usr/bin/env python
# coding: utf-8

# # LoanLens - Data Preprocessing
# 
# ## Objective
# 
# The cleaned dataset cannot be used directly for machine learning because it still contains missing values, categorical variables, and features with different numerical scales.
# 
# The objective of this notebook is to build a reusable preprocessing pipeline using Scikit-Learn.
# 
# The preprocessing pipeline will:
# 
# - Split the dataset into training and testing sets.
# - Handle missing values.
# - Encode categorical variables.
# - Scale numerical features.
# - Produce transformed datasets ready for machine learning models.
# 
# Using a pipeline ensures that identical preprocessing steps are consistently applied during both training and prediction while preventing data leakage.

# In[1]:


import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split

from sklearn.impute import SimpleImputer

from sklearn.preprocessing import (
    OneHotEncoder,
    StandardScaler
)

from sklearn.compose import ColumnTransformer

from sklearn.pipeline import Pipeline


# In[2]:


df = pd.read_csv("../data/application_train_clean.csv")

print(df.shape)

df.head()


# ### Observation
# 
# The cleaned dataset was successfully loaded and will be used for all subsequent preprocessing operations.

# In[3]:


X = df.drop(columns="TARGET")

y = df["TARGET"]

print(X.shape)
print(y.shape)


# ### Observation
# 
# The target variable (`TARGET`) was separated from the predictor variables.
# 
# - **X** contains all input features.
# - **y** contains the loan default labels.

# In[7]:


X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# In[5]:


print("Training set:", X_train.shape)
print("Testing set :", X_test.shape)


# ## Train-Test Split
# 
# The dataset was divided into training and testing sets using an 80:20 ratio.
# 
# Stratified sampling was applied to preserve the original class distribution of the target variable in both datasets.
# 
# Using a separate test dataset ensures that model performance is evaluated on unseen data.

# # Feature Identification
# 
# ## Problem
# 
# Machine learning algorithms require different preprocessing techniques for numerical and categorical features.
# 
# Therefore, the predictor variables are separated into numerical and categorical groups before building the preprocessing pipeline.

# In[9]:


numerical_features = X_train.select_dtypes(
    include=["int64", "float64", "bool"]
).columns.tolist()

categorical_features = X_train.select_dtypes(
    include=["object"]
).columns.tolist()

print(f"Number of Numerical Features : {len(numerical_features)}")
print(f"Number of Categorical Features : {len(categorical_features)}")

print("\nFirst 10 Numerical Features:")
print(numerical_features[:10])

print("\nCategorical Features:")
print(categorical_features)


# ### Observation
# 
# The predictor variables were successfully separated into numerical and categorical feature groups.
# 
# This separation enables different preprocessing techniques to be applied based on the feature type.
# 
# - Numerical features will be imputed using the median and standardized.
# - Categorical features will be imputed using the most frequent category and one-hot encoded.

# # Numerical Preprocessing Pipeline
# 
# The numerical preprocessing pipeline performs the following operations:
# 
# 1. Replace missing values using the median.
# 2. Standardize numerical features using StandardScaler.

# In[11]:


numerical_pipeline = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ]
)

numerical_pipeline


# # Categorical Preprocessing Pipeline
# 
# The categorical preprocessing pipeline performs the following operations:
# 
# 1. Replace missing categorical values using the most frequent category.
# 2. Convert categorical variables into numerical representations using One-Hot Encoding.
# 
# One-Hot Encoding allows machine learning algorithms such as Logistic Regression to process categorical variables without assuming any ordinal relationship between categories.

# In[12]:


categorical_pipeline = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ]
)

categorical_pipeline


# # Column Transformer
# 
# The numerical and categorical preprocessing pipelines are combined using a ColumnTransformer.
# 
# This ensures that:
# 
# - Numerical features receive numerical preprocessing.
# - Categorical features receive categorical preprocessing.
# 
# Both transformations are applied automatically during model training and prediction.

# In[13]:


preprocessor = ColumnTransformer(
    transformers=[
        ("num", numerical_pipeline, numerical_features),
        ("cat", categorical_pipeline, categorical_features)
    ]
)

preprocessor


# In[14]:


preprocessor.fit(X_train)


# Why only **X_train**?
# 
# Because the imputer learns medians and modes from the training data only. If we fit on the full dataset, information from the test set leaks into training, which leads to overly optimistic evaluation.

# In[15]:


X_train_processed = preprocessor.transform(X_train)
X_test_processed = preprocessor.transform(X_test)


# In[16]:


print("Processed Training Shape :", X_train_processed.shape)
print("Processed Testing Shape  :", X_test_processed.shape)


# ### Observation
# 
# The preprocessing pipeline was successfully fitted using the training dataset.
# 
# The transformed datasets contain:
# 
# - No missing values.
# - Encoded categorical variables.
# - Standardized numerical variables.
# 
# The data is now ready for machine learning model training.

# # P5 – Logistic Regression Baseline
# Problem
# 
# The preprocessing pipeline has prepared the dataset for machine learning. A baseline Logistic Regression model will now be trained to establish an initial performance benchmark.
# 
# The results from this model will be used as a reference when comparing more advanced algorithms later in the project.

# In[18]:


from sklearn.linear_model import LogisticRegression


# In[19]:


log_reg = LogisticRegression(
    max_iter=1000,
    random_state=42
)


# In[20]:


log_reg.fit(X_train_processed, y_train)


# In[21]:


y_pred = log_reg.predict(X_test_processed)


# In[22]:


y_prob = log_reg.predict_proba(X_test_processed)[:, 1]


# In[32]:


from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report
)


# In[24]:


print("Accuracy :", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall   :", recall_score(y_test, y_pred))
print("F1 Score :", f1_score(y_test, y_pred))
print("ROC AUC  :", roc_auc_score(y_test, y_prob))


# In[29]:


from sklearn.metrics import ConfusionMatrixDisplay

ConfusionMatrixDisplay.from_predictions(y_test, y_pred, cmap = "Blues")


# In[33]:


print(classification_report(y_test, y_pred))


# # Conclusion
# 
# The baseline Logistic Regression model was successfully trained and evaluated using the preprocessing pipeline.
# 
# The preprocessing pipeline correctly handled:
# 
# - Missing value imputation
# - Numerical feature scaling
# - Categorical feature encoding
# 
# The obtained evaluation metrics will serve as the baseline for comparing more advanced machine learning models in the next phase of the project.
# 
# ### Model Interpretation
# 
# Although the Logistic Regression model achieved a high overall accuracy, the recall for the default class is extremely low.
# 
# This indicates that the model struggles to identify applicants who are likely to default due to the strong class imbalance in the dataset.
# 
# Future improvements will focus on addressing this imbalance using techniques such as class weighting, resampling (e.g., SMOTE), and more advanced algorithms.

# In[ ]:




