import pandas as pd
from sqlalchemy import create_engine
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
import numpy as np
# from ydata_profiling import ProfileReport
from data_profiling import ProfileReport
import pdfkit
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
roc_auc_score, average_precision_score, confusion_matrix,
classification_report)
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from sklearn.naive_bayes import GaussianNB
import joblib
print("\N{white heavy check mark}","All libraries loaded")

print("Processing started")

# ## Step 1: Data Understanding

# **Findings:**
# - 51,500 rows × 26 columns
# - Numerical Columns:14, categorical:12
# - 9 columns with missing values (largest gap ~16%)
# - Inconsistent labels detected (e.g., `yes `, `Male`/`male`)
# - 1,500 duplicate rows

df=pd.read_csv('HR_Job_Placement_Dataset.csv')
df1=df.copy()
print(f"Shape: {df1.shape}")
print(f"Duplicates: {df1.duplicated().sum()}")
print(f"\nTarget distribution: ")
print(f"Target Column name: {df1['status'].value_counts()}")
print(f"\nMissing values:")
print(df1.isnull().sum().sort_values(ascending=False))

print("Data Set loading to dataframe completed")

print("Data Cleaning & Preprocessing started")
print("Formatting of non-categorized labels, Duplicate removal, Null Imputation, Outlier normalization/removal processing started")
# -------------------------------------------------------
# "To identify the non-categorized lavbels in the columns"
# -------------------------------------------------------
# print("Before Cleaning")
for i in df1.columns:
    if df1[i].dtype in ['object','category']:
        dt=df1[i].value_counts().reset_index()
        # print(f"\n--- {i} ---")
        # print(dt)


# print(f"\n***After Inconsistent labels Cleaning: ***")
for i in df1.columns:
    if df1[i].dtype in ['object','category']:
        df1[i]=df1[i].str.title()
        df1[i] = df1[i].str.strip()
        df1[i]=df1[i].astype('category')
        dt=df1[i].value_counts().reset_index()
        # print(f"\n--- {i} ---")
        # print(dt)  

print("Formatting of Inconsistent labels in Data Set values completed")
# ----------------------------------------------------------------------------------------------------
## Step 2: Data Cleaning & Preprocessing
# ----------------------------------------------------------------------------------------------------

# Count total duplicate rows
print(f"Duplicate rows: {df1.duplicated().sum()}")

# See the actual duplicate rows

duplicates = df1[df1.duplicated(keep=False)]  # keep=False marks ALL duplicates
print(f"\n Total no. of.Rows involved in Duplication: {duplicates.shape[0]}")

# ----------------------------------------------------------------------------------------------------
# confirmation of the duplicate is for the same person by comparing the fixed features of the person
# ----------------------------------------------------------------------------------------------------

variable_colums=['age_years','gender', 'ssc_percentage','hsc_percentage', 'degree_percentage', 'degree_specialization', 'previous_ctc_lpa',
                 'expected_ctc_lpa', 'company_tier', 'job_role_match','notice_period_days','layoff_history', 'employment_gap_months', 'relocation_willingness']

df1_Variable_dupe=df1.duplicated(subset=variable_colums, keep=False)
Variable_sample = (df1[df1_Variable_dupe].sort_values(by=variable_colums))


print("\n_____________________________________________________________________________\n")

print(f"Before Duplicates removal: {df1.shape}\n")
df1 = df1.drop_duplicates()
print(f"After Duplicates removal:  {df1.shape}")

print("Detection and removal of Duplicated rows in Data Set completed")
# ----------------------------------------------------------------------------------------------------
## STEP 3: NULL IMPUTATION
# ----------------------------------------------------------------------------------------------------
## Null Imputation:
#  Correlation:
# Target (Status) is highly correlated with technical_score,years_of_experience,skills_match_percentage,internship_experience,previous_ctc_lpa,communication_score
# Target (Status) is negatively correlated with expected_ctc_lpa, job_role_match.


# 1.Employment_gap_months
# print(f"\nSkewness: {df['employment_gap_months'].skew():.2f}")
# As skewness (1.72) is more than 1, the values are rightly skewed, hence taking median to fill values
df1['employment_gap_months']=df1['employment_gap_months'].fillna(df1['employment_gap_months'].median())

# 2.Notice_period_days           
# print(f"\nSkewness: {df['notice_period_days'].skew():.2f}")
# As skewness (1.72) is more than 1, the values are rightly skewed, hence taking median to fill values
df1['notice_period_days']=df1['notice_period_days'].fillna(df1['notice_period_days'].median())


# 3.hsc_percentage
# print(f"\nSkewness: {df['hsc_percentage'].skew():.2f}")
# As skewness (0.01) is less than 0.5, the values are approx disttributted normally, hence taking median to fill values
df1['hsc_percentage']=df1['hsc_percentage'].fillna(df1['hsc_percentage'].median())

# 4.ssc_percentage
# print(f"\nSkewness: {df['ssc_percentage'].skew():.2f}")
# As skewness (0.05) is less than 0.5, the values are approx disttributted normally, hence taking mean to fill values
df1['ssc_percentage']=df1['ssc_percentage'].fillna(df1['ssc_percentage'].mean())


# 5.job_role_match

# print(df1['job_role_match'].value_counts())
# print(f"\n{df1['job_role_match'].mode().tolist()}")
df1['job_role_match']=df1['job_role_match'].fillna(df1['job_role_match'].mode()[0])


# 6.relevant_experience

# print(df1['relevant_experience'].value_counts())
# print(f"\n{df1['relevant_experience'].mode().tolist()}")
df1['relevant_experience'].describe()['top']
df1['relevant_experience']=df1['relevant_experience'].fillna(df1['relevant_experience'].mode()[0])
print(df1['relevant_experience'].value_counts())

# 7.career_switch_willingness
print(f"Before Imputation:")
print(df1['career_switch_willingness'].value_counts())
print(f"\n{df1['career_switch_willingness'].mode().tolist()}")
df1['career_switch_willingness'].describe()['top']
df1['career_switch_willingness']=df1['career_switch_willingness'].fillna(df1['career_switch_willingness'].mode()[0])
print(f"\nAfter Null Imputation:")
print(df1['career_switch_willingness'].value_counts())

# 8.layoff_history
print(f"Before Imputation:")
print(df1['layoff_history'].value_counts())
print(f"\n{df1['layoff_history'].mode().tolist()}")
df1['layoff_history'].describe()['top']
df1['layoff_history']=df1['layoff_history'].fillna(df1['layoff_history'].mode()[0])
print(df1['layoff_history'].value_counts())

# 9.relocation_willingness 
print(f"Before Imputation:\n")
print(df1['relocation_willingness'].value_counts())
print(f"\n{df1['relocation_willingness'].mode().tolist()}")
df1['relocation_willingness'].describe()['top']
df1['relocation_willingness']=df1['relocation_willingness'].fillna(df1['relocation_willingness'].mode()[0])
print(f"\nAfter Null Imputation:")
print(df1['relocation_willingness'].value_counts())

print(df1.isna().sum())

# ----------------------------------------------------------------------------------------------------
## STEP 4: EDA
#univariate anlysis
#ouutlier detection
# -----------------------------------------------------------------------------------------------------
# Calculate IQR bounds

for i in df1.columns:
    if df1[i].dtype in ['float64','int64']:
        Q1 = df1[i].quantile(0.25)
        Q3 = df1[i].quantile(0.75)
        IQR = Q3 - Q1

        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        outliers = df1[(df1[i] < lower_bound) | (df1[i] > upper_bound)]

        # df1[i].plot.hist(bins=30, title=f'Distribution of {i}')
        # plt.xlabel(i)
        # plt.ylabel('Frequency')
        # plt.show()
        #  # Visualization: Boxplot with custom dimensions
        # # print(f"Visualization for outlier detection in column: {i}")
        # fig = px.box(df1, y=i, title=f"{i} Outlier Detection", width=900, height=600)

        # fig.add_hline(y=lower_bound, line_color="red", line_dash="dash", annotation_text=f"Lower Bound: {lower_bound:.2f}", annotation_position="top left")
        # fig.add_hline(y=upper_bound, line_color="green", line_dash="dash", annotation_text=f"Upper Bound: {upper_bound:.2f}", annotation_position="top right")
        # # fig.show()
        # print(f"____________________________________________________________________________________________________________________________________________________\n")

        if outliers.shape[0] > 0:          
            print(f"Column name: {i}")
            print(f"Before: {outliers.shape[0]} outliers found")
            df1.loc[:, i] = df1[i].clip(lower_bound, upper_bound)
            remaining = ((df1[i] < lower_bound) | (df1[i] > upper_bound)).sum()
            print(f"After:  {remaining} outliers remaining")
                          
        else:
            print(f"Column name: ***{i}***: No outliers")
        print("-" * 60)

# -----------------------------------------------------------------------------------------------------
# ## Step 4: Exploratory Data Analysis

# **Six required analyses:**
# 1. Interview score vs job acceptance
# 2. Skills match % vs placement
# 3. Company tier vs acceptance rate
# 4. Experience vs placement probability
# 5. Competition level vs job acceptance
# 6. Correlation heatmap among numeric features
# -----------------------------------------------------------------------------------------------------

# Creating spearate new column and converting dtype to int to identify % and other numerical calculation for the taget column. this is common for all the EDA analysis

df1['placement_rate'] = (df1['status'] == 'Placed').astype(int)

# Interview score vs job acceptance

avg_interview_score = (df1['technical_score']+df1['aptitude_score']+df1['communication_score']) / 3
score_levels = pd.cut(avg_interview_score,bins=5,labels=['Very Low', 'Low', 'Medium', 'High', 'Very High'])
placement_rate = df1.groupby(score_levels)['placement_rate'].mean() * 100
print("Placement Rate by Interview Score Group:")
print(placement_rate)

# placement_rate.plot(kind='bar', figsize=(8, 5))

# plt.title('Placement Rate by Interview Score Level', fontweight='bold')
# plt.xlabel('Interview Score Level')
# plt.ylabel('Placement Rate (%)')
# plt.xticks(rotation=0)
# plt.show()


plot_data = placement_rate.reset_index()
plot_data.rename(columns={'index':'score_levels'},inplace=True)
fig = px.bar(plot_data,x='score_levels',y='placement_rate',title='Placement Rate by Interview Score Level',text='placement_rate')
fig.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
fig.update_layout(width=700, height=700)
fig.show()

# 2. Skills match vs placement
skills_match_level = pd.cut(df1['skills_match_percentage'],bins=5,labels=['Very Low', 'Low', 'Medium', 'High', 'Very High'])
placement = df1.groupby(skills_match_level)['placement_rate'].mean() * 100
plot_data1 = placement.reset_index()
plot_data1.rename(columns={'skills_match_percentage':'skills_match_level'}, inplace=True)
print("Placement Rate by skills match level:")
print(plot_data1)


fig = px.bar(
    plot_data1,
    x='skills_match_level',
    y='placement_rate',
    title='Placement Rate by Skills Match Level',
    text='placement_rate',
)
fig.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
fig.update_layout(title_x=0.5, height=700, width=700)
fig.show()


# 3. Company tier vs acceptance rate
Acceptance_rate = df1.groupby('company_tier')['placement_rate'].mean()* 100
plot_data=Acceptance_rate.reset_index()
print("Acceptance Rate by Interview Score Group:")
print(plot_data)

fig = px.bar(
    plot_data,
    x='company_tier',
    y='placement_rate',
    title='Acceptance Rate by Skills Match Level',
    text='placement_rate',
)
fig.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
fig.update_layout(title_x=0.5, height=700, width=700)
fig.show()

# 4. Experience vs placement probability
Experience_level = df1.groupby(['years_of_experience','avg_interview_score', 'expected_ctc_lpa'])['placement_rate'].mean()*100
plot_data1=Experience_level.reset_index()
print("Placement probability based on Experience:")
print(plot_data1)

fig = px.bar(
    plot_data1,
    x='years_of_experience',
    y='placement_rate',
    title='Placement probability based on Experience',
    text='placement_rate',
)
fig.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
fig.update_layout(title_x=0.5, height=700, width=700)
fig.show()

# 5.Competition level vs job acceptance

Competition_level = df1.groupby('competition_level')['placement_rate'].mean()*100
plot_data2=Competition_level.reset_index()
print("Placement Rate based on competition:")
print(plot_data2)

fig = px.bar(
    plot_data2,
    x='competition_level',
    y='placement_rate',
    title="Placement Rate based on competition",
    text='placement_rate',
)
fig.update_traces(texttemplate='%{text:.2f}%', textposition='outside')
fig.update_layout(title_x=0.5, height=700, width=700)
fig.show()

#6. Correlation of Numerical Columns 
# Encoding
df_corr=df1.copy()
for i in df_corr:
  if df_corr[i].dtype in ["category", "object"]:
    le = LabelEncoder()
    df_corr[i] = le.fit_transform(df1[i])

corr=df_corr.corr()
fig = px.imshow(corr,text_auto='.2f',width=1200, height=1200)
fig.show()

# -----------------------------------------------------------------------------------------------------
## Step 5: Feature Engineering
●	Experience category (Fresher / Junior / Senior)
●	Academic performance bands
●	Skills match level (Low / Medium / High)
●	Interview performance category
●	Placement probability score 
# -----------------------------------------------------------------------------------------------------

# ============================================================
# 1.EXPERIENCE CATEGORY
# ============================================================
df1['experience_category'] = df1['years_of_experience'].apply(lambda y: 'Fresher' if y==0 else ('Junior' if y<=3 else 'Senior'))
# ============================================================
# 2.ACADEMIC PERFORMANCE BANDS
# ============================================================
df1['academic_band'] = df1['degree_percentage'].apply(lambda p: 'Low' if p<60 else ('Medium' if p<=75 else 'High'))
# ============================================================
# 3. SKILLS MATCH LEVEL (3-TIER)
# ============================================================
df1['skills_match_level'] = df1['skills_match_percentage'].apply(lambda p: 'Low' if p<60 else ('Medium' if p<=80 else 'High'))
# ============================================================
# 4. INTERVIEW PERFORMANCE CATEGORY
# ============================================================
df1['interview_performance'] = df1['avg_interview_score'].apply(
    lambda s: 'Poor' if s<50 else ('Average' if s<65 else ('Good' if s<80 else 'Excellent'))) #avg_interview_score column has been created during EDA analysis

# ============================================================
# 5. PLACEMENT PROBABILITY SCORE
# ============================================================
df1['placement_prob_score'] = (0.40*(df1['technical_score']/100) +
                                0.25*(df1['skills_match_percentage']/100) +
                                0.15*(df1['avg_interview_score']/100) +
                                0.10*df1['job_role_match'].map({'Matched':1,'Not Matched':0}).astype(float) +
                                0.10*df1['company_tier'].map({'Tier 1':1,'Tier 2':0.5,'Tier 3':0}).astype(float))

print(f"✓ Categorical + interaction features created")

df1.to_csv("Job_accept_Final_analysis.csv",index=False)

from sqlalchemy import create_engine
import pymysql

# --- Create database ---
conn = pymysql.connect(host='127.0.0.1', port=3306, user='root', password=<your_password>)
cursor = conn.cursor()
cursor.execute("DROP DATABASE IF EXISTS Project")
cursor.execute("CREATE DATABASE project")


engine = create_engine('mysql+pymysql://root:Mwin%402028@127.0.0.1:3306/project')
df2 = pd.read_csv("Job_accept_Final_analysis.csv")
df2.to_sql("Job", engine, if_exists='replace', index=False)
print(f"Uploaded {len(df2)} rows to databse project")
cursor.close()
conn.close()

 ============================================================
## Machine Learning Modeling
# 1."Logistic Regression"
# 2."Decision Tree"
# 3."Random Forest": Randonm forest classifier
# 4."XGBoost": XGBClassifier
# 5."KNN": KNeighborsClassifier
#6.Naive Bayes
============================================================

#Filtering coumns for x and y
drop_columns = ["placement_rate", "avg_interview_score", "placement_prob_score",
         "score_levels", "skills_match_level", "experience_category",
         "academic_band", "interview_performance"]

y = (df2["status"] == "Placed").astype(int)
X = df2.drop(columns=["status"] + drop_columns)

# Encode categorical columns using LabelEncoder copied from eda analysis done during correlation analysis (dict included to store the encoded values for future use)
encoders = {}
for i in X.columns:
    if X[i].dtype in ["object", "category"]:
        le = LabelEncoder()
        X[i] = le.fit_transform(X[i])
        encoders[i] = le

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y) #stratify =y Keeps same % of Placed/Not Placed in train & test

# Models
models = {
    "lr": LogisticRegression(max_iter=2000, random_state=42),
    "dt": DecisionTreeClassifier(random_state=42),
    "rf": RandomForestClassifier(random_state=42),
    "xgb": XGBClassifier(random_state=42),
    "knn": KNeighborsClassifier(n_neighbors=5),
    "nb": GaussianNB(),
}
# Train & evaluate
results = []
best_metric=-1
best_model=None
best_name = ""

for i, j in models.items():
    j.fit(X_train, y_train)

    y_pred = j.predict(X_test)
    y_proba = j.predict_proba(X_test)[:, 1]

    acc  = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec  = recall_score(y_test, y_pred)
    f1   = f1_score(y_test, y_pred)
    auc  = roc_auc_score(y_test, y_proba)

    results.append({
        "j": i, "Accuracy": acc, "Precision": prec,
        "Recall": rec, "F1": f1, "ROC-AUC": auc
    })

    print(f"\n--- {i} ---")
    print(f"Accuracy: {acc:.4f}  Precision: {prec:.4f}  Recall: {rec:.4f}  F1: {f1:.4f}  ROC-AUC: {auc:.4f}")
    print(classification_report(y_test, y_pred, target_names=["Not Placed", "Placed"]))

    if auc > best_metric:
        best_metric, best_model, best_name = auc, j, i

# Summary
summary = pd.DataFrame(results).sort_values("ROC-AUC", ascending=False).reset_index(drop=True)
print("\n=== Model Comparison ===")
print(summary.to_string(index=False))

# # Save
# pickle.dump(best_model, open("/workspace/best_model.pkl", "wb"))
# pickle.dump(encoders, open("/workspace/encoders.pkl", "wb"))
print(f"\nBest: {best_name} (ROC-AUC = {best_metric:.4f})")