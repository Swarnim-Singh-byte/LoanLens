#!/usr/bin/env python
# coding: utf-8

# # LoanLens - Data Cleaning
# 
# ## Objective
# 
# The goal of this notebook is to clean the errors we found in the eda notebook where we made the analysis this will be a transformation notebook.
# 
# Cleaning will be performed in sequence so that we don't miss anything and preserve useful information for machine learning.
# 
# Dataset : application_train.csv
# 
# Expected Output : application_train_clean.csv
# 

# ## Cleaning Plan
# | Problem ID | Problem | Severity | Decision | Status |
# |------------|----------|----------|----------|--------|
# | P1 | Duplicate Rows | Low | Check & Remove | ⏳. |
# | P2 | Missing Values | High | Analyze column-wise | ⏳ ..|
# | P3 | Sentinel Values | High | Replace invalid values | ⏳ ...|
# | P4 | Outliers | Medium | Decide treatment | ⏳ ....|
# | P5 | Data Types | Low | Verify | ⏳..... |
# | P6 | Constant Columns | Low | Remove if needed | ⏳...... |
# | P7 | High Missing Columns | High | Decide drop/impute | ⏳....... |
# | P8 | Redundant Features | Medium | Review | ⏳ ........|
# | P9 | Target Leakage | High | Verify | ⏳......... |

# # P1 first why are duplicate rows important 
# 
# ### Why are duplicate rows important?
# 
# Duplicate records can bias statistical analysis and machine learning models by giving certain observations more importance than others.
# 
# During EDA, duplicate records were not explicitly removed, therefore they are verified before any further preprocessing.

# In[1]:


import pandas as pd
import numpy as np

pd.set_option("display.max_columns", 150)
pd.set_option("display.float_format", "{:.3f}".format)

df = pd.read_csv("../data/application_train.csv")

print(f"Dataset Shape : {df.shape}")


# In[2]:


duplicate_rows = df.duplicated().sum()

print(f"Number of duplicate rows: {duplicate_rows}")


# ### Observation
# 
# No duplicate records were found in the dataset.
# 
# ### Cleaning Decision
# 
# No duplicate removal was required. The original dataset is retained without modification.
# For a better understanding we performed it here as well it was already done in eda nootebook as well.

# # P2 - Sentinel Values
# 
# ## Problem
# 
# Some datasets use placeholder values to represent missing or unknown information instead of actual null values.
# 
# During the EDA phase, the `DAYS_EMPLOYED` feature was found to contain the value `365243`, which does not represent a realistic number of employment days.
# 
# Keeping this value would distort statistical analysis and negatively affect model training.
# 
# Therefore, these values will be replaced while preserving the information that they originally contained.
# 
# Let's see the verification below 

# In[15]:


anomaly_count = (df['DAYS_EMPLOYED'] == 365243).sum()
print(f"Anomalous rows: {anomaly_count} ({anomaly_count/len(df)*100:.2f}%)")


# In[ ]:


# create an anamoly flag is to preserve the information

df["DAYS_EMPLOYED_ANOM"] = ( df["DAYS_EMPLOYED"] == 365243 )

df["DAYS_EMPLOYED_ANOM"].value_counts()


# In[12]:


df["DAYS_EMPLOYED"] = (df["DAYS_EMPLOYED"].replace(365243, np.nan)) #replacing the values with nan.


# In[13]:


print("Remaining anomalous values:", (df["DAYS_EMPLOYED"] == 365243).sum())

df["DAYS_EMPLOYED"].describe()


# Over here when we used describe we found the employee day had exceptionally higher days which is not possible because 365243 days would mean that someone worked for 1000 years which is not possible. So, we changed the value with Nan values.

#  # P3 
#  ### Now we will go to the part where we will see does our missing vlaues means anything? why are they missing and how to deal with them? do they sing a different song than getting dropped?
# 
# #### Problem
# 
# Missing values are common in real-world datasets and can affect model performance if not handled appropriately.
# 
# However, not all missing values should be treated the same way. Some represent genuinely unavailable information, while others may carry meaningful information about an applicant.
# 
# Therefore, each feature group will be evaluated before selecting an appropriate cleaning strategy.

# ## Missing Value Strategy
# 
# | Missing Percentage | Strategy |
# |--------------------|----------|
# | 0% | No action required |
# | Less than 5% | Impute |
# | 5–30% | Review individually |
# | More than 30% | Investigate before making a decision |

# In[16]:


# Missing percentage for each column

missing_percent = (df.isnull().mean() * 100).sort_values(ascending=False)

missing_percent = missing_percent[missing_percent > 0]

missing_percent.head(20)


# In[17]:


high_missing = missing_percent[missing_percent > 50]

medium_missing = missing_percent[
    (missing_percent > 5) &
    (missing_percent <= 50)
]

low_missing = missing_percent[missing_percent <= 5]

print(f"High Missing (>50%): {len(high_missing)} columns")
print(f"Medium Missing (5-50%): {len(medium_missing)} columns")
print(f"Low Missing (<5%): {len(low_missing)} columns")


# In[21]:


print(high_missing.index.tolist( ))


# In[22]:


df[[
    "COMMONAREA_AVG",
    "COMMONAREA_MEDI",
    "COMMONAREA_MODE"
]].describe()


# In[23]:


df[[
    "COMMONAREA_AVG",
    "COMMONAREA_MEDI",
    "COMMONAREA_MODE"
]].head(10)


# ### Observation
# 
# The `COMMONAREA_AVG`, `COMMONAREA_MEDI`, and `COMMONAREA_MODE` features contain approximately **69.87% missing values**, with only **92,646** non-missing observations out of **307,511** records.
# 
# The descriptive statistics show that all three features have very similar distributions:
# 
# - Their means are nearly identical.
# - Their quartiles (25%, 50%, and 75%) are almost the same.
# - They all share the same minimum and maximum values.
# - The missing values occur in exactly the same rows.
# 
# We can also see that over here the missigness is occuring together and when the values are pressent they are pretty close to each other. This shows these columns are representing the same underlying characteristic using different statistical summaries.
# 
# #### Decision 
# At this stage, no cleaning operation will be applied to these features.
# 
# Although the columns contain a high percentage of missing values, they describe an important characteristic of the applicant's property. Since the three features represent different statistical summaries of the same attribute, they will be retained for now and evaluated later during feature engineering and feature selection.
# 
# The missing values will be handled during the preprocessing stage rather than being removed at this point.

# In[24]:


df[[
    "NONLIVINGAPARTMENTS_AVG",
    "NONLIVINGAPARTMENTS_MEDI",
    "NONLIVINGAPARTMENTS_MODE"
]].describe()


# In[25]:


df[[
    "NONLIVINGAPARTMENTS_AVG",
    "NONLIVINGAPARTMENTS_MEDI",
    "NONLIVINGAPARTMENTS_MODE"
]].head(10)


# ### Observation
# 
# The `NONLIVINGAPARTMENTS_AVG`, `NONLIVINGAPARTMENTS_MEDI`, and `NONLIVINGAPARTMENTS_MODE` features exhibit the same behaviour as the previously analysed `COMMONAREA` feature group.
# 
# The three columns:
# 
# - Have identical counts of non-missing observations.
# - Share nearly identical descriptive statistics.
# - Contain missing values in the same records.
# - Represent the same underlying attribute using different statistical summaries (Average, Median, and Mode).
# 
# This indicates that the missingness is associated with the availability of the property information rather than an issue with individual columns.
# 
# #### Cleaning Decision
# 
# No cleaning operation will be applied at this stage.
# 
# The three features will be retained because they describe the same property using different statistical summaries. Since they contain potentially useful information, they will be evaluated later during feature selection and model development.
# 
# The missing values will be handled during the preprocessing stage.

# In[26]:


pd.crosstab(
    df["FLAG_OWN_CAR"],
    df["OWN_CAR_AGE"].isna(),
    margins=True
)


# ### Observation
# 
# A cross-tabulation between **FLAG_OWN_CAR** and **OWN_CAR_AGE** shows that applicants who do not own a car (`FLAG_OWN_CAR = 'N'`) consistently have missing values for `OWN_CAR_AGE`.
# 
# This confirms that the missing values are expected and represent the absence of a car rather than missing or corrupted data.
# 
# #### Cleaning Decision
# 
# The `OWN_CAR_AGE` feature will be retained without replacing the missing values during the cleaning stage.
# 
# The missing values are meaningful because they indicate that an applicant does not own a car. Imputation will be considered later during preprocessing only if required by the selected machine learning algorithm.

# In[29]:


df["EXT_SOURCE_1"].describe()


# In[30]:


df["EXT_SOURCE_1"].head(10)


# In[31]:


print(f"Missing values: {df['EXT_SOURCE_1'].isnull().sum()}")
print(f"Missing percentage: {df['EXT_SOURCE_1'].isnull().mean()*100:.2f}%")


# ### Observation
# 
# The `EXT_SOURCE_1` feature contains **173,378** missing values (**56.38%** of the dataset).
# 
# Unlike the previously analysed property-related features, `EXT_SOURCE_1` is a single numerical feature with values ranging approximately between **0 and 1**, indicating that it represents a normalized external score.
# 
# The exact reason for the missing values cannot be determined from the dataset alone, therefore the missingness cannot be considered meaningful.
# 
# ### Cleaning Decision
# 
# The `EXT_SOURCE_1` feature will be retained despite having a high percentage of missing values.
# 
# This feature is widely recognised as one of the most informative variables for predicting loan default in the Home Credit dataset. Removing it solely because of its missing percentage could significantly reduce model performance.
# 
# The missing values will be imputed later during the preprocessing stage using the machine learning pipeline.

# In[32]:


df["OCCUPATION_TYPE"].describe()


# In[33]:


df["OCCUPATION_TYPE"].value_counts(dropna=False)


# ### Observation
# 
# The `OCCUPATION_TYPE` feature contains **18 unique occupation categories** and **96,391 missing values** (approximately **31.35%** of the dataset).
# 
# The most frequent occupation is **Laborers**, while the missing values represent the largest single category.
# 
# Since occupation is an important socioeconomic characteristic, removing the feature or deleting rows with missing values could result in a significant loss of useful information.
# 
# #### Cleaning Decision
# 
# The `OCCUPATION_TYPE` feature will be retained.
# 
# The missing values will not be replaced during the cleaning stage. Instead, categorical imputation using the **most frequent category** will be performed later as part of the preprocessing pipeline before model training.
# 
# This approach avoids data leakage and ensures a consistent preprocessing workflow.

# In[34]:


low_missing


# ### Cleaning Decision
# 
# These features will be retained without any modifications during the data cleaning stage.
# 
# Since the proportion of missing values is negligible, the missing numerical values will be imputed using the **median**, while the missing categorical values will be imputed using the **most frequent category** during the preprocessing stage.
# 
# Performing imputation within the preprocessing pipeline prevents data leakage and ensures that the same transformations are consistently applied to both the training and testing datasets.
# 
# #### Cleaning Decision
# 
# These features will be retained without any modifications during the data cleaning stage.
# 
# Since the proportion of missing values is negligible, the missing numerical values will be imputed using the **median**, while the missing categorical values will be imputed using the **most frequent category** during the preprocessing stage.
# 
# Performing imputation within the preprocessing pipeline prevents data leakage and ensures that the same transformations are consistently applied to both the training and testing datasets.   

# # P4 - Outlier Treatment
# 
# ## Problem
# 
# Several numerical features were identified as containing outliers during the exploratory data analysis phase using the Interquartile Range (IQR) method.
# 
# However, not every outlier should be removed. In financial datasets, unusually large values often represent genuine customer characteristics rather than data entry errors.
# 
# Therefore, each feature will be evaluated before deciding whether the outliers should be retained, capped, or transformed.

# ### Outlier Analysis Summary
# 
# The detailed outlier analysis was performed during the Exploratory Data Analysis (EDA) phase using the Interquartile Range (IQR) method.
# 
# The analysis identified several numerical features containing extreme values. However, these outliers were further examined before deciding whether any treatment was required.

# ## 1. Binary and Categorical Features
# 
# ### Observation
# 
# The IQR method identified several binary and categorical variables as containing outliers.
# 
# Examples include:
# 
# - TARGET
# - FLAG_EMP_PHONE
# - FLAG_WORK_PHONE
# - FLAG_EMAIL
# - REGION_RATING_CLIENT
# - REGION_RATING_CLIENT_W_CITY
# - FLAG_DOCUMENT_*
# 
# These are discrete variables where the detected "outliers" simply represent valid categories rather than erroneous observations.
# 
# ### Cleaning Decision
# 
# No treatment is required for these features.

# ## 2. DAYS_EMPLOYED
# 
# ### Observation
# 
# During the EDA phase, the extreme values in `DAYS_EMPLOYED` were found to originate from the placeholder value **365243** rather than genuine employment durations.
# 
# ### Cleaning Decision
# 
# This issue has already been resolved in Problem 2 by replacing the placeholder with missing values while preserving the anomaly using the `DAYS_EMPLOYED_ANOM` indicator feature.

# ## 3. Financial Features
# 
# ### Observation
# 
# The EDA identified outliers in the following monetary features:
# 
# - AMT_INCOME_TOTAL
# - AMT_CREDIT
# - AMT_ANNUITY
# - AMT_GOODS_PRICE
# 
# These values represent genuine financial characteristics of applicants rather than data entry errors.
# 
# ### Cleaning Decision
# 
# Most financial outliers will be retained because they are valid observations.
# 
# However, extremely high income values in `AMT_INCOME_TOTAL` will be capped at the 99th percentile to reduce the influence of extreme observations on machine learning models such as Logistic Regression.

# In[36]:


income_cap = df["AMT_INCOME_TOTAL"].quantile(0.99)

df["AMT_INCOME_TOTAL"] = df["AMT_INCOME_TOTAL"].clip(upper=income_cap)

print(df["AMT_INCOME_TOTAL"].describe())


# ## 4. Property Features
# 
# ### Observation
# 
# Property-related variables such as:
# 
# - COMMONAREA
# - LIVINGAREA
# - APARTMENTS
# - BASEMENTAREA
# - LANDAREA
# 
# also contain values identified as outliers by the IQR method.
# 
# These values correspond to legitimate differences in property characteristics and therefore should not be considered data errors.
# 
# ### Cleaning Decision
# 
# No treatment will be applied to these features.

# ## 5. Count Features
# 
# ### Observation
# 
# Features such as `CNT_CHILDREN` and `CNT_FAM_MEMBERS` contain a small number of unusually high values.
# 
# Although these observations are rare, they remain plausible and cannot be classified as erroneous records.
# 
# ### Cleaning Decision
# 
# These features will be retained without modification.

# In[38]:


df.to_csv("../data/application_train_clean.csv", index=False)


# # P5 - Data Type Verification
# 
# ## Problem
# 
# Before exporting the cleaned dataset, the data types of all features should be verified to ensure they are correctly interpreted during preprocessing and model training.
# 
# Incorrect data types can lead to errors during feature engineering and machine learning.

# In[39]:


df.info()


# In[40]:


import os

print(os.path.exists("../data/application_train_clean.csv"))


# ### Observation
# 
# The cleaned dataset contains **307,511 records** and **123 features**, including the newly created `DAYS_EMPLOYED_ANOM` indicator feature.
# 
# The dataset consists of:
# 
# - 66 floating-point numerical features (`float64`)
# - 40 integer numerical features (`int64`)
# - 16 categorical features (`object`)
# - 1 boolean feature (`DAYS_EMPLOYED_ANOM`)
# 
# No unexpected data type conversions were observed during the cleaning process.

# # Cleaning Summary
# 
# The following data cleaning operations were completed successfully:
# 
# - Checked for duplicate records (none found).
# - Identified and corrected the `DAYS_EMPLOYED` placeholder anomaly while preserving anomaly information using an indicator feature.
# - Analysed missing values and defined feature-specific handling strategies.
# - Evaluated outliers identified during EDA and retained valid observations while capping extreme income values.
# - Verified data types and exported the cleaned dataset.
# 
# The cleaned dataset is now ready for preprocessing, feature engineering, and machine learning model development.
Raw Dataset
        │
        ▼
Duplicate Check
        │
        ▼
Sentinel Value Handling
        │
        ▼
Missing Value Analysis
        │
        ▼
Outlier Treatment
        │
        ▼
Data Type Verification
        │
        ▼
Cleaned Dataset
(application_train_clean.csv)
# In[ ]:




