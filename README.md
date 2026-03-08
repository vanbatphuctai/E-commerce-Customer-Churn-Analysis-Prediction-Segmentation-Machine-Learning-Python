# 📉 E-commerce Customer Churn Analysis, Prediction & Segmentation | Machine Learning, Python

**Author:** Van Bat Phuc Tai  
**Tools Used:** Machine Learning - Python

---

## 📑 Table of Contents

- [📌 Background & Overview](#-background--overview)
- [📂 Dataset Description & Data Structure](#-dataset-description--data-structure)
- [🧹 Data Cleaning & Preprocessing](#-data-cleaning--preprocessing)
- [🔍 Customer Churn Behavior Analysis (EDA)](#-customer-churn-behavior-analysis-eda)
- [🤖 Customer Churn Prediction](#-customer-churn-prediction)
- [📊 Churned Customer Segmentation Using Clustering](#-churned-customer-segmentation-using-clustering)
- [📈 Visualization & Behavioral Analysis](#-visualization--behavioral-analysis)
- [💡 Insights & Business Recommendations](#-insights--business-recommendations)

---

## 📌 Background & Overview

### 🏢 Business Context

This project is based on a **company operating in the e-commerce industry**, where customers purchase products through an online platform. The company offers a variety of product categories including: Fashion, Laptop & Accessories, Mobile Phones, Mobile Devices, Grocery, Other consumer products.

As the company grows, **customer retention becomes increasingly important**. Losing customers (customer churn) not only reduces revenue but also increases marketing costs, since **acquiring a new customer is typically much more expensive than retaining an existing one**.

With thousands of users interacting with the platform daily, identifying customers who are at risk of leaving becomes difficult through manual analysis alone. Therefore, the company aims to leverage **data analytics and machine learning** to better understand churn behavior and proactively retain customers.

To support **data-driven decision making**, the Data Analytics team developed a churn analysis framework that combines:

- Behavioral analysis (**Exploratory Data Analysis – EDA**)  
- **Machine learning models** to predict churn risk  
- **Customer segmentation** to understand different churn patterns  

This approach helps the business move from **reactive churn management** to **proactive customer retention strategies**.

---

### 🎯 Project Objectives

This project focuses on **three key analytical goals**:

#### 1️⃣ Customer Churn Behavior Analysis (EDA)

Analyze behavioral patterns of churned customers to identify key factors related to:

- Customer engagement  
- Satisfaction  
- Service experience  

#### 2️⃣ Customer Churn Prediction

Develop a **supervised machine learning model** to estimate the **probability that a customer will churn**, allowing the business to identify high-risk users and implement early retention actions.

#### 3️⃣ Churned Customer Segmentation Using Clustering

Use **KMeans clustering** to group churned users based on behavioral characteristics, enabling more **targeted win-back promotions**.

---

### ❓ Key Business Questions

This analysis aims to answer several important business questions:

- What behavioral and service-related factors contribute to customer churn?  
- How can machine learning help predict customers who are likely to churn?  
- Which features have the strongest impact on churn behavior?  
- How can churned customers be segmented into meaningful groups for targeted marketing strategies?

---

### 👤 Stakeholders

This project supports multiple stakeholders within the organization:

**Data Analysts & Business Analysts**

- Understand churn drivers  
- Generate actionable insights  

**Marketing & Customer Retention Teams**

- Design data-driven promotional campaigns  

**Business Leaders & Decision Makers**

- Reduce churn and improve customer lifetime value

---

## 📂 Dataset Description & Data Structure

This dataset contains customer behavior and transaction information collected from an e-commerce company's internal database. It is commonly used for **customer churn prediction, data analysis, and machine learning projects**.

### 📌 Data Source

- **Source:** Internal e-commerce company database  
- **Dataset File:** `churn_prediction.xlsx`  
- **Size:** 5,630 rows × 20 columns  

The dataset records **customer activity, demographic information, purchasing behavior, and engagement metrics** to analyze factors influencing customer churn.

---

### 📊 Data Structure

#### Tables Used

The dataset contains **only one table** with customer and transaction-related data.

#### Table: Customer Churn Data

<details>
<summary><strong>📋 Click to expand table schema</strong></summary>

<br>

| Column Name | Data Type | Description |
|-------------|-----------|-------------|
| CustomerID | INT | Unique identifier for each customer |
| Churn | INT | Churn flag (1 if customer churned, 0 if active) |
| Tenure | FLOAT | Duration of customer's relationship with the company (months) |
| PreferredLoginDevice | OBJECT | Device used for login (Mobile, Desktop, etc.) |
| CityTier | INT | City tier (1 = Tier 1, 2 = Tier 2, 3 = Tier 3) |
| WarehouseToHome | FLOAT | Distance between warehouse and customer's home (km) |
| PreferredPaymentMode | OBJECT | Customer’s preferred payment method |
| Gender | OBJECT | Gender of the customer |
| HourSpendOnApp | FLOAT | Hours spent on the app or website in the past month |
| NumberOfDeviceRegistered | INT | Number of devices registered under the customer's account |
| PreferredOrderCat | OBJECT | Preferred order category for the customer |
| SatisfactionScore | INT | Satisfaction rating given by the customer |
| MaritalStatus | OBJECT | Marital status of the customer |
| NumberOfAddress | INT | Number of addresses registered by the customer |
| Complain | INT | Indicator if the customer made a complaint (1 = Yes) |
| OrderAmountHikeFromLastYear | FLOAT | Percentage increase in order amount compared to last year |
| CouponUsed | FLOAT | Number of coupons used by the customer last month |
| OrderCount | FLOAT | Number of orders placed by the customer last month |
| DaySinceLastOrder | FLOAT | Days since the last order was placed |
| CashbackAmount | FLOAT | Average cashback received by the customer in the past month |

</details>

---

### 🎯 Target Variable

- **Churn**
  - `1` → Customer churned  
  - `0` → Active customer  

---

## 🧹 Data Cleaning & Preprocessing

### 📌 Import Necessary Libraries

[In 1]:
```python
# Data manipulation
import pandas as pd
import numpy as np

# Data visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Preprocessing
from sklearn.preprocessing import LabelEncoder, StandardScaler, RobustScaler

# Model selection
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV

# Evaluation metrics
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    recall_score,
    precision_score,
    f1_score,
    confusion_matrix,
    silhouette_score)

# Classification models
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

# Gradient boosting library
import xgboost as xgb

# Dimensionality reduction
from sklearn.decomposition import PCA

# Clustering
import scipy.cluster.hierarchy as sch
from sklearn.cluster import AgglomerativeClustering
```

### 📥 Data Loading

[In 2]:

```python
# Mount Google Drive to access dataset
from google.colab import drive
drive.mount('/content/drive')

# Load dataset
df = pd.read_excel('/content/drive/MyDrive/Machine Learning - Final Project/churn_prediction.xlsx')
```

### ⚙️ Data Cleaning & Preprocessing

Before diving into analysis, let's take a quick look at the first few rows of the dataset to examine its structure and key features

[In 3]:

```python
# Print the first five rows of the dataset
df.head(5)
```

[Out 3]:


[In 4]:

```python
# Check Data Summary
df.describe()
```

[Out 4]:


[In 5]:

```python
# Check the general information of df
df.info()
```

[Out 5]:


#### 💡 Data Understanding

📌 Before performing any analysis or modeling, I carried out several steps to preprocess the data:

#### 📝 Checked Dataset Structure

The dataset contains **5,630 rows** and **20 columns**, with a mix of **numeric** and **categorical** variables.

Missing values were identified in several columns, such as `Tenure`, `WarehouseToHome`, `HourSpendOnApp`, etc.

#### 📝 Checked for Missing Values

[In 6]:

```python
# Check for missing values
df.isnull().sum()
```

[Out 6]:

Missing values were detected in multiple columns. The columns with missing values are:

- `Tenure`: 264 missing values  
- `WarehouseToHome`: 251 missing values  
- `HourSpendOnApp`: 255 missing values  
- `OrderAmountHikeFromLastYear`: 265 missing values  
- `CouponUsed`: 256 missing values  
- `OrderCount`: 258 missing values  
- `DaySinceLastOrder`: 307 missing values

#### 📝 Checked for Duplicates

[In 7]:

```python
# Check for duplicate records
df.duplicated().sum()
```

[Out 7]:

Aftering checkeing for duplicate rows in the dataset and found that there were no duplicate entries.

#### 📝 Check Unique Values in Categorical Columns

[In 8]:

```python
# Check unique values in categorical columns
for col in df.select_dtypes(include='object'):
    print(col)
    print(df[col].unique())
    print()
```

[Out 8]:


#### 💡 Summary: Handling Data Issues

- **Missing Values:** Missing values in numerical columns were imputed using the median to minimize the impact of outliers.

- **Duplicate Records:** No duplicate records were found; therefore, no further action was necessary.

- **Consistent Categorical Values:** Categorical variables with synonymous meanings were standardized and merged into unified labels to ensure consistency.

#### 📝 Missing Value Handling

[In 9]:

```python
# Define the list of columns with missing values
cols_missing = ['Tenure', 'WarehouseToHome', 'HourSpendOnApp', 'OrderAmountHikeFromlastYear', 'CouponUsed', 'OrderCount', 'DaySinceLastOrder']

# Replace missing columns with median
for col in cols_missing:
    # Fill missing values in each column with the median of that column
    df[col].fillna(value= df[col].median(), inplace=True)
```

#### 📝 Standardizing Categorical Values

[In 10]:

```python
# Standardizing Equivalent Categorical Values

standardize_map = {
    'PreferredLoginDevice': {
        'Phone': 'Mobile Phone'},
    'PreferredPaymentMode': {
        'COD': 'Cash on Delivery',
        'CC': 'Credit Card'},

    'PreferredOrderCat': {
        'Mobile': 'Mobile Phone'}}

for col, mapping in standardize_map.items():
    df[col] = df[col].replace(mapping)
```
---

## 🔍 Customer Churn Behavior Analysis (EDA)

### Target Variable Analysis

Check for class imbalance in the target variable `Churn` by calculating the count and percentage of churned vs. active customers.

[In 11]:

```python
# Check class imbalance
imb_df = df['Churn'].value_counts().reset_index()
imb_df.columns = ['Churn','Count']

# Calculate percentage
imb_df['Percentage'] = imb_df['Count'] / imb_df['Count'].sum()

imb_df
```
[Out 11]:


#### 💡 Insight

The dataset is **moderately imbalanced**, with **83.16% non-churn** and **16.84% churn** customers.  
Although churn cases are fewer, the sample size is sufficient for predictive modeling.

Therefore, evaluation metrics such as **Balanced Accuracy**, **Precision**, **Recall**, and **F1-score** should be considered instead of relying solely on overall accuracy.

### Univariate Analysis

Analyze the distribution of categorical variables by plotting the count of each category.

[In 12]:

```python
# List of categorical columns
cat_cols = [
    'PreferredLoginDevice',
    'Gender',
    'MaritalStatus',
    'PreferredPaymentMode',
    'PreferredOrderCat'
]

# Create 1 row, 5 columns
fig, axes = plt.subplots(1, 5, figsize=(18,5))

# Loop through columns
for i, col in enumerate(cat_cols):

    sns.countplot(x=col, data=df, ax=axes[i])

    axes[i].set_title(f'Distribution of {col}')
    axes[i].tick_params(axis='x', rotation=45)

# Adjust layout
plt.tight_layout()

# Show
plt.show()
```
[Out 12]:


Analyze the distribution of numerical variables using boxplots to identify spread, skewness, and potential outliers.

[In 13]:

```python
# List of numerical columns
num_cols = [
    'Tenure',
    'WarehouseToHome',
    'HourSpendOnApp',
    'OrderAmountHikeFromlastYear',
    'CouponUsed',
    'OrderCount',
    'DaySinceLastOrder',
    'CashbackAmount',
    'SatisfactionScore',
    'NumberOfDeviceRegistered',
    'NumberOfAddress'
]

# Create subplots (2 rows x 6 columns)
fig, axes = plt.subplots(2, 6, figsize=(22,8))
axes = axes.flatten()

for i, col in enumerate(num_cols):

    sns.boxplot(x=df[col], ax=axes[i])
    axes[i].set_title(col)

# Remove empty subplot
for j in range(len(num_cols), len(axes)):
    fig.delaxes(axes[j])

plt.tight_layout()
plt.show()
```
[Out 13]:


### Final Data Inspection
Final data inspection was conducted to ensure the dataset is clean and ready for modeling.

[In 14]:

```python
# Print dataset info to verify missing values and column types
print(df.info())
```
[Out 14]:


#### 💡 Insight: Exploratory Data Analysis (EDA)

#### Categorical Features
- Most customers prefer to log in using **mobile phones**, indicating strong mobile platform usage.
- The dataset contains **more male customers than female customers**.
- **Married customers** represent the largest group among all marital statuses.
- The most commonly used payment methods are **Debit Card and Credit Card**.
- Customers mainly purchase products from **Laptop & Accessories** and **Mobile Phone** categories.

#### Numerical Features
- Most customers have **low tenure**, suggesting many relatively new users on the platform.
- The **distance between warehouse and customer homes** is mostly around **10–20 units**.
- Customers typically spend **around 2–3 hours on the app**.
- **Coupon usage and order counts are generally low**, indicating relatively infrequent purchasing behavior.
- Many customers show a **large number of days since their last order**, which may signal potential churn risk.
- **Cashback amounts** are mainly concentrated between **150 and 250**.

#### Key Business Insight
Customers with **low purchase frequency, low coupon usage, and longer time since their last order** are more likely to **churn**, highlighting the importance of retention strategies such as promotions and personalized offers.
---
### Train & Apply Churn Prediction Model
