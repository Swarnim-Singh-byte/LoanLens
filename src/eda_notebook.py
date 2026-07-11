#!/usr/bin/env python
# coding: utf-8

# Loan Lens
# 

# First step in any ml project is eda and we will start doing that and learning it.
# Eda answers our questions what it is, what can go wrong and what went wrong in the data, what is the pattern discovering pattern is the goal here so that we can find our exactly trends of our business let's say. 
# 
# Our eda is a detective work
# 
# Don't kn0w how much it's true but a bad model on good eda beats a good model on bad eda.
# 
# The possible questions eda answers:
# 1) What do i have?
# 2) is the data healty and relevant?
# 3) what does the target looks like? but what does target means here? will see that further in the notebook.
# 4) What do all different features looks like?
# 5) How do features relate to the target?
# 6) How do features relate to each other?
# 7) What can be the possible missing values?
# 8) Do we even need this much data?

# So our first question - What do i have?
# 
# The first aim - To understand our shape and size of our data how many rows and how many columns?
# 
# Now the question is why do we need it? so the answers is we need to know few stuffs before working on it and those are -: 
# 
# For example
# 1) How many applicants(rows)?
# 2) How many pieces of info per applicant ? that means how much information do we have about our applicants.
# 3) Type of data in each column?
# 4) What does even our first few rows even looks like?
# 

# In[12]:


import sys
get_ipython().system('{sys.executable} -m pip install pandas numpy matplotlib seaborn missingno')


# In[13]:


# now how do exactly do it for that we would need to use some libraries so first let's import them

import pandas as pd #reading the files, manipulates data, filter rows and columns, create new columns, it groups data as well
import numpy as np # Pandas is made on numpy, numpy handles every crucial mathematical problems, functions everything.
import matplotlib.pyplot as plt # our skecthbook of all charts we create
import seaborn as sns # It's like pandas how pandas came from numpy this come is based on matplotlib, this higher level plotting library injust few lines.
import missingno as msno # this helps us to visualize the missing data.
import warnings # control warning messages because these libraries give lots of warnings which are not actually the error and only end up cluttering our notebook.

# now to ignore the warnings
warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', 100) #function is pandas which let's us display according to our own needs so when the dataset has many columns pandas shows only few cols and use(...) to showcase others.
pd.set_option('display.float_format', '{:.3f}'.format)

print(" 🙂‍↕️ ALL THE LIBRARIES ARE LOADED SUCCESSFULLY, YAHOO")


# In[14]:


# Now main game is on let's load our data 

df = pd.read_csv('../data/application_train.csv')

# FIrst look
print("Data loaded successfully :) ") 
print(f"Rows: {df.shape[0]: ,}") #to see how many applicants means rows we have we use shape[0], (:, is used as a formator add , after the thousand place)
print(f"Columns: {df.shape[1]: ,}") # same over here just 1 for columns
df.head()


# In[15]:


# info about our data 
df.info()


# In[16]:


df.describe()


# The describe over here shows the stats of our dataset which will further tell us about skewed, outliers etc.
# 
# Over here let's go a bit back and see a bit about what is default rate mentioned in the document and how does it matter here.
# 
# So over here the mean is actually helping a lot let's say for the target column over here we know the vlaues we have are either **0 or 1 nothing else which means 0 - repaid and  1 as - defaulted.**
# 
# Now to find how many defaulted are we would have to do something like:- 
# **(Proportion of defaults = number of 1s)**. 
# 
# But Mean is a shortcut and 0 doesn't change anything in sum so it's more like :- **Mean = (sum of all values(0+1))/(count of all values)**
# 
# NOw, this means 8.1% of the people are defaulters here, which matches are doc prediction at the same time shows that this is a **heavily imbalanced classification problem(let's call it 'hicp').**
# 
# SO, hicp means that in our dataset we have very few errors or defaulters for our model to learn for future prediction and since usually ml model ignore overall small errors we would face problems until we took so external measures. 92% people are correct with thier timing some hiccups might be there but they are paying istallments mostly on time and our model can learn it nicely but for 8% we would have use technique like SMOTE, Recall@k etc.
# 
# and one more thing mean> median(50%) shows that our data is skewed right. Which tells us whether to log-transform a feature before modelling.
# 
# 25%/75%(quartiles) the middle of where the data sits it can detect outliers when max is way beyond 75% means there are some interesting outliers.
# 
# 

# In[17]:


# Now we will try to find the missing data values.
missing = df.isnull().sum().sort_values(ascending=False)
print(missing)


# In[18]:


pd.set_option('display.max_rows', None) #when we are putting none means we are saying that don't put any range show all the rows.
# missing[missing > 0] #over here we are trying to find all the columns which has missing values from missing var we wrote earlier where is added all the null values.
missing_per = (missing/len(df)) * 100
missing_df = pd.DataFrame({'Missing Count' : missing, 'Missing %' : missing_per})
missing_df = missing_df[missing_df['Missing Count'] > 0]
missing_df


# In[19]:


msno.bar(df)


# Sometimes working with missingvalues doesn't mean that it is just noise missing values might also lead us to the fact that whether our target is getting affected by it or not. Sometimes it might me clue there is a possiblity that if the applicant doesn't have realiable house he may be a defaulter or maybe he is not. So now let's try to find that. The image above is the visual of missing values using the library missingno. Now, in the below cell well will run a loop if missingvalues of the columns says anything about the target.

# In[20]:


cols_with_missing = missing_df.index.tolist()
missingness_vs_target = {} # empty dictionary to collect the result or comparsion
for col in cols_with_missing:
    flag = df[col].isnull().astype(int) #df[col].isnull() give us the values as true false and astype(int) change them in binary numbers 0 or 1
    default_rate_missing = df.loc[flag == 1, 'TARGET'].mean()
    #we did mean here becaucse we need to know how much default rates do we have like the people who are missing are they effecting target as well?
    default_rate_present = df.loc[flag == 0, 'TARGET']. mean()
    missingness_vs_target[col] = { 'default_rate_if_missing': default_rate_missing,
                                   'default_rate_if_present': default_rate_present,
                                   'difference' : default_rate_missing - default_rate_present
                                 }
missingness_df = pd.DataFrame(missingness_vs_target).T
missingness_df.sort_values('difference', ascending=False)






# ## What I got to know
# 
# So over here we can see if we compare the default rate between **missing** and **the one has it** we can see the difference and use the overall difference rate 8.1% to judge so that we can something meaningful.
# 
# ### Key findings
# 
# **1. Credit bureau inquiry columns (AMT_REQ_CREDIT_BUREAU_*)**
# - Missing: 10.3% default rate | Present: 7.7% default rate | Difference: +2.6%
# - People with no credit bureau inquiry history default more often — likely applicants 
#   who are new to formal credit systems, which is itself a real risk signal.
# 
# **2. Housing/building data block (APARTMENTS_*, ELEVATORS_*, ENTRANCES_*, 
#    YEARS_BUILD_*, TOTALAREA_MODE, EMERGENCYSTATE_MODE, etc. — ~40 related columns)**
# - Missing: ~8.6-9.3% default rate | Present: ~6.9-7.0% default rate | Difference: ~+1.7-2.3%
# - This confirms my earlier hypothesis: applicants without documented housing/building 
#   info default somewhat more often. The pattern is small per-column, but remarkably 
#   **consistent across every column in this block** — which makes it trustworthy rather 
#   than a fluke.
# 
# **3. External credit scores (EXT_SOURCE_1, EXT_SOURCE_3)**
# - Missing: 8.5-9.3% default rate | Present: 7.5-7.8% default rate | Difference: +1.0-1.5%
# - Missing an external bureau score (a "thin credit file") is itself associated with 
#   slightly higher default risk.
# 
# ### What I'm NOT trusting
# A few columns (AMT_ANNUITY, CNT_FAM_MEMBERS, DAYS_LAST_PHONE_CHANGE) showed a large 
# *negative* difference (missing group defaulting at 0%) — but these columns only had 
# **1-12 missing rows total** out of 307,511. With so few missing rows, a 0% default rate 
# among them is very likely random chance, not a real pattern. I'm treating these as 
# statistical noise, not a genuine finding.
# 
# ### What This Means Going Forward
# 
# Missingness itself carries a real signal in several places, not just noise. My plan:
# 
# - **Housing/building block (~40 columns):** create a single `has_housing_data` flag, 
#   rather than imputing all ~40 columns individually and diluting the signal across 
#   many near-duplicate features.
# - **EXT_SOURCE_1 / EXT_SOURCE_3:** keep `EXT_SOURCE_1_was_missing` and 
#   `EXT_SOURCE_3_was_missing` as their own binary features, alongside the imputed values.
# - **Credit bureau inquiries (AMT_REQ_CREDIT_BUREAU_*):** these six columns are missing 
#   for the exact same 41,519 applicants every time — they're really one signal reported 
#   six ways. Collapse into a single `has_bureau_inquiry_data` flag, and fill the actual 
#   counts with 0 (not median), since a missing record most likely means "no inquiries," 
#   not "average inquiries."
# 
# **For today's Week 2 baseline pipeline**, I'm using a simpler median-fill approach on 
# a small feature set, just to get a working end-to-end model quickly. The missingness-flag 
# strategy above is planned for Week 3, as part of the dedicated Data Cleaning and Feature 
# Engineering steps on my roadmap.
# 
# 

# In[22]:


#finding duplicates
df.duplicated().sum()


# **Duplicate check:** NO duplicate found since df.duplicated().sum() returned 0.

# In[25]:


numeric_cols = df.select_dtypes(include = ['int64', 'float64']).columns
numeric_cols
correlations = df[numeric_cols].corr()['TARGET'].sort_values(ascending = False)
correlations


# So, over here +1 in correlation we can say it has high correlation means both of them can go up together, 0 means no relation and -1 means negative relation.
# 
# **EXT_SOURCE_3, EXT_SOURCE_2, EXT_SOURCE_1** are by far the strongest predictors 
# (-0.179, -0.160, -0.155) — higher external credit score, lower default risk. No other 
# feature comes close.
# 
# **DAYS_BIRTH** (+0.078) is the strongest positive correlation — since this is stored 
# as negative days-before-today, this means younger applicants default somewhat more.
# 
# Everything else (DAYS_EMPLOYED, loan amounts, FLAG_DOCUMENT_*, building measurements) 
# is weak, mostly under ±0.05.
# 

# In[26]:


for col in df.select_dtypes(include='object').columns:
    print(f"\n{col}")
    print(df.groupby(col)['TARGET'].mean().sort_values(ascending=False))




# ### Categorical Features vs TARGET
# 
# **Strong, trustworthy signals:**
# - Education: default rate ranges from 1.8% (Academic degree) to 10.9% (Lower secondary)
# - Occupation: 4.8% (Accountants) to 17.2% (Low-skill Laborers)
# - Housing type: 6.6% (Office apartment) to 12.3% (Rented apartment)
# 
# **Caution — small sample artifacts:**
# NAME_INCOME_TYPE shows extreme values (Maternity leave: 40%, Student: 0%) — these 
# categories likely have very few applicants, so the percentages aren't statistically 
# reliable. Same for CODE_GENDER's "XNA" category at 0%. Verified via value_counts() 
# before treating any of these as real signal.
# 
# **Weak/no signal:** WEEKDAY_APPR_PROCESS_START shows almost no variation (0.078-0.084 
# across all days) — not a useful predictor.

# In[27]:


df['NAME_INCOME_TYPE'].value_counts()


# **Confirmed via value_counts():** NAME_INCOME_TYPE's extreme percentages are small-sample 
# artifacts — Unemployed (22 people), Student (18), Businessman (10), Maternity leave (5) 
# are all tiny groups out of 307,511 total. Their default rates (36-40% or 0%) aren't 
# statistically reliable. The four main categories (Working, Commercial associate, 
# Pensioner, State servant) have solid sample sizes and their default rates are trustworthy.
# 

# In[28]:


import missingno as msno
msno.heatmap(df)


# In[29]:


anomaly_count = (df['DAYS_EMPLOYED'] == 365243).sum()
print(f"Anomalous rows: {anomaly_count} ({anomaly_count/len(df)*100:.2f}%)")


# For outliers method now

# In[34]:


numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns

outlier_summary = {}

for col in numeric_cols:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
    outlier_summary[col] = {
        'outlier_count': len(outliers),
        'outlier_pct': (len(outliers) / len(df)) * 100
    }

outlier_df = pd.DataFrame(outlier_summary).T
outlier_df.sort_values('outlier_pct', ascending=False)


# In[35]:


top_outlier_cols = outlier_df.sort_values('outlier_pct', ascending=False).head(6).index.tolist()

fig, axes = plt.subplots(2, 3, figsize=(15, 8))
axes = axes.flatten()

for i, col in enumerate(top_outlier_cols):
    axes[i].boxplot(df[col].dropna())
    axes[i].set_title(col)

plt.tight_layout()
plt.show()


# ### Outlier Check — IQR Method Across All Numeric Columns
# 
# Ran the IQR method (1.5×IQR rule) across all numeric columns. Most "high outlier %" 
# results are false positives from applying a continuous-outlier method to binary flags 
# (FLAG_*), ratings (REGION_RATING_CLIENT), or rare-event counts (DEF_30_CNT_SOCIAL_CIRCLE) 
# — these aren't real data quality issues, just the natural shape of categorical-coded data.
# 
# **Genuine outlier findings, worth addressing in cleaning:**
# - DAYS_EMPLOYED (23.5%) — confirms the known 365243 placeholder anomaly, needs fixing
# - AMT_INCOME_TOTAL (4.6%) — confirms the extreme high-income outliers found earlier, 
#   candidate for capping
# - AMT_CREDIT (2.1%), AMT_ANNUITY (2.4%), CNT_CHILDREN (1.4%) — mild, expected levels, 
#   worth light capping but not a major concern
# 
# **Clean columns (no real outlier issue):** DAYS_BIRTH, EXT_SOURCE_1/2/3 — smooth 
# distributions, no cleaning needed here.

# ## EDA Summary — Key Insights
# 
# - **Dataset:** 307,511 applicants, 122 columns, no duplicate rows.
# - **Target:** 8.1% default rate — a heavily imbalanced classification problem, 
#   requiring techniques like SMOTE and Recall@K rather than relying on accuracy alone.
# - **Strongest predictors:** EXT_SOURCE_1/2/3 (external credit scores) show the 
#   strongest correlations with TARGET (-0.155 to -0.179) — no other feature comes close. 
#   DAYS_BIRTH (+0.078) shows younger applicants default somewhat more.
# - **Categorical signals:** Education, occupation, and housing type show large, 
#   reliable default-rate spreads (e.g. Lower secondary 10.9% vs Academic degree 1.8% 
#   default). Some extreme percentages (e.g. NAME_INCOME_TYPE's "Maternity leave") are 
#   small-sample artifacts, not real signal — confirmed via value_counts().
# - **Missingness is informative, not just noise:** applicants missing housing/building 
#   data, bureau inquiry data, or EXT_SOURCE scores default at meaningfully higher rates 
#   than those with complete data. The missingno heatmap confirms these columns are 
#   missing together as structural blocks, not independently.
# - **Data quality issues found:** DAYS_EMPLOYED contains a placeholder value (365243) 
#   in 18.01% of rows — needs explicit handling before modeling. AMT_INCOME_TOTAL has 
#   an extreme outlier (117 million) and CNT_CHILDREN has an outlier (19 children) — 
#   both likely need capping.
# - **Next steps (per project roadmap):** Data Cleaning (fix DAYS_EMPLOYED, handle 
#   outliers, decide missing-value strategy) → Feature Engineering (age, employment 
#   duration, income ratios, missingness flags) → Baseline Model.

# In[ ]:




