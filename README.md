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
- [📈 Churned Customer Segmentation Visualization](#-churned-customer-segmentation-visualization)
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

<img width="920" alt="image" src="https://github.com/user-attachments/assets/e22bd399-dd2d-4e75-934a-19f05cdd9f98" />


[In 4]:

```python
# Check Data Summary
df.describe()
```

[Out 4]:

<img width="900" alt="image" src="https://github.com/user-attachments/assets/eb5bd0c9-53e2-4a75-9047-ca9978229a4a" />

[In 5]:

```python
# Check the general information of df
df.info()
```

[Out 5]:

<img width="900" alt="image" src="https://github.com/user-attachments/assets/8e6fcbfd-01eb-4121-8fae-0426842ce564" />


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

<img width="900" alt="image" src="https://github.com/user-attachments/assets/0068a65a-7ace-4f43-98df-28e3add0dd3d" />

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

<img width="200" alt="image" src="https://github.com/user-attachments/assets/8d96ced3-631a-4c4a-85f4-21f779583775" />

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

<img width="900" alt="image" src="https://github.com/user-attachments/assets/5d82dfee-e12e-46eb-9e28-af340e46a835" />

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

<img width="245" alt="image" src="https://github.com/user-attachments/assets/c0239a9c-15af-4f51-8893-b02613fd8cf4" />

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

<img width="1220" alt="image" src="https://github.com/user-attachments/assets/f5d3217a-56ca-4f1b-a07a-76b09ee4bc2c" />

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

<img width="1580" alt="image" src="https://github.com/user-attachments/assets/1131eca8-8663-4192-adf0-f24266e16e59" />


### Final Data Inspection
Final data inspection was conducted to ensure the dataset is clean and ready for modeling.

[In 14]:

```python
# Print dataset info to verify missing values and column types
print(df.info())
```

[Out 14]:

<img width="500" alt="image" src="https://github.com/user-attachments/assets/b23ec996-f281-4a6d-bb1e-5474b5363410" />


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

#### 📝 Categorical Feature Encoding

To prepare the dataset for model training, categorical variables were converted into numerical representations suitable for machine learning algorithms.

#### **One-Hot Encoding**

Categorical features with multiple categories were transformed using **One-Hot Encoding**.  
This technique creates separate binary columns for each category, allowing the model to interpret non-ordinal categorical data effectively.

**Encoded columns:**
- `PreferredLoginDevice`
- `PreferredPaymentMode`
- `PreferedOrderCat`
- `MaritalStatus`

#### **Label Encoding**

The `Gender` feature was encoded using **Label Encoding**, converting categorical values into binary numerical values (`0` and `1`).

#### **Removing Unnecessary Feature**

The `CustomerID` column was removed because it functions only as a **unique identifier** and does not contribute meaningful information for the **churn prediction model**.

[In 15]:

```python
# One-hot encoding for categorical variables
df_encoded = pd.get_dummies(
    df,
    columns=[
        'PreferredLoginDevice',
        'PreferredPaymentMode',
        'PreferredOrderCat',
        'MaritalStatus'])

# Label encoding for binary categorical variable
label_encoder = LabelEncoder()
df_encoded['Gender'] = label_encoder.fit_transform(df_encoded['Gender'])

# Drop CustomerID
df_encoded = df_encoded.drop(columns=['CustomerID'])

df_encoded
```

[Out 15]:

<img width="1520" alt="image" src="https://github.com/user-attachments/assets/24fb51b8-04a2-44e2-bf15-fe9df4150de4" />

#### 📊 **Feature and Target Separation**

To prepare the dataset for model training and evaluation, the data was split into:

- **Features (X):** All input variables used to train the model.
- **Target (y):** The `Churn` column, representing whether a customer churned or not.

[In 16]:

```python
# Split the data into features (X) and target (y)
x = df_encoded.drop('Churn', axis=1)
y = df_encoded['Churn']  # Target

# Split into training and testing sets (70/30 split)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)
```

#### ⚙️ Feature Scaling: Standardization

To ensure that all features contribute equally to the model, **feature scaling** was applied using **StandardScaler**.

The `StandardScaler` standardizes features by transforming them to have:

- **Mean = 0**
- **Standard Deviation = 1**

This step is important for many machine learning algorithms, especially those sensitive to feature magnitude (e.g., Logistic Regression, SVM, KNN).

The scaler was **fit on the training data** and then applied to both the **training** and **testing** sets to avoid data leakage.

[In 17]:

```python
# Standardize the features using StandardScaler
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)
```

---

### 📝 Apply Model - Random Forest Classifier

A **Random Forest Classifier** was trained using the scaled feature set.

Predictions were generated for both the **training** and **testing** datasets.

[In 18]:

```python
# Train a Random Forest model
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(x_train_scaled, y_train)

# Make predictions on training and test sets
y_pred_train = clf.predict(x_train_scaled)
y_pred_test = clf.predict(x_test_scaled)

# Evaluate model accuracy
train_acc = accuracy_score(y_train, y_pred_train)
test_acc = accuracy_score(y_test, y_pred_test)
train_balanced_acc = balanced_accuracy_score(y_train, y_pred_train)
test_balanced_acc = balanced_accuracy_score(y_test, y_pred_test)

# Print the results
print(f'Training Accuracy: {train_acc:.4f}')
print(f'Test Accuracy: {test_acc:.4f}')
print(f'Training Balanced Accuracy: {train_balanced_acc:.4f}')
print(f'Test Balanced Accuracy: {test_balanced_acc:.4f}')
```

[Out 18]:

<img width="300" alt="image" src="https://github.com/user-attachments/assets/81df9717-8b03-4da3-a429-3dfbcf115c53" />

#### 💡 Model Insight

The Random Forest model performs very well on churn prediction.  
Training accuracy reaches **100%**, indicating the model fits the training data perfectly, which may suggest **slight overfitting**.

On the test set, the model achieves **94.43% accuracy** and **86.25% balanced accuracy**, showing that it generalizes well and can effectively predict both churn and non-churn customers despite the class imbalance.

Overall, the model demonstrates **strong predictive performance** for identifying customers at risk of churn.

### 📝 Apply Random Forest To Find Important Features

[In 19]:

```python
# Get feature importances from the trained model
feats = {feature: importance for feature, importance in zip(x_train.columns, clf.feature_importances_)}

# Create a DataFrame
importances = pd.DataFrame.from_dict(feats, orient='index').rename(columns={0: 'Importances'})

# Sort values
importances = importances.sort_values(by='Importances', ascending=True).reset_index()

# Plot feature importances
plt.figure(figsize=(10,10))
plt.barh(importances.tail(20)['index'], importances.tail(20)['Importances'])
plt.title('Feature Importance')
plt.show()
```

[Out 19]:

<img width="900" alt="image" src="https://github.com/user-attachments/assets/a106055d-1522-4d13-8e8e-d2084bcb2288" />

#### 💡 Insight

**Feature importance** analysis shows that `Tenure` is the most influential factor affecting churn, followed by `CashbackAmount`, `WarehouseToHome`, `Complain`, and `DaySinceLastOrder`.

`WarehouseToHome` may influence delivery experience, which can impact customer satisfaction and retention.

Next, histograms are plotted to compare churn and non-churn behavior across the most important features, helping identify patterns related to customer churn.

#### Feature Distribution by Churn Status

The following histograms compare the distributions of `Tenure`, `CashbackAmount`, `WarehouseToHome`, and `DaySinceLastOrder` between churn and non-churn customers, helping reveal patterns associated with customer churn.

[In 20]:

```python
# Select top important features identified from feature importance analysis
top_features = [
    'Tenure',
    'CashbackAmount',
    'WarehouseToHome',
    'DaySinceLastOrder'
]

# Create a figure with 1 row and 4 columns for better comparison
fig, axes = plt.subplots(1, 4, figsize=(20,4))

# Loop through each top feature and plot histogram by churn status
for i, col in enumerate(top_features):
    
    sns.histplot(
        data=df,
        x=col,
        hue='Churn',      # Separate distributions for churn vs non-churn customers
        bins=30,          # Number of bins in histogram
        kde=True,         # Add density curve to observe distribution trend
        ax=axes[i]
    )
    
    # Set title for each subplot
    axes[i].set_title(col)

# Adjust layout to prevent overlapping
plt.tight_layout()

# Display the plots
plt.show()
```

[Out 20]:

<img width="1220" alt="image" src="https://github.com/user-attachments/assets/fa20b341-b5c8-4ea1-b9dd-bc5557617c77" />

#### Complaint vs Churn

The count plot compares churn and non-churn customers based on the `Complain` feature, helping illustrate how customer complaints relate to churn behavior.

[In 21]:

```python
# Create a figure for the plot
plt.figure(figsize=(5,4))

# Plot count distribution of the 'Complain' variable
# The hue parameter separates customers by churn status (Churn vs Non-Churn)
sns.countplot(data=df, x='Complain', hue='Churn')

# Set chart title
plt.title('Complain vs Churn')

# Display the plot
plt.show()
```

[Out 21]:

<img width="500" alt="image" src="https://github.com/user-attachments/assets/f4634d0b-db2a-4f56-86f2-21179b9839ef" />

### Customer Churn Behavior Insights and Retention Recommendations

#### 💡Findings:

| Metric | Churn (Blue) | Non-Churn (Yellow) | Insight | Recommendation |
|------|------|------|------|------|
| **Tenure (Customer Lifespan)** | Mostly concentrated in 0–10 months, very few customers stay long-term | More customers stay beyond 10–20 months | Customers with short tenure are more likely to churn, suggesting early-stage engagement is critical | Improve onboarding experience, loyalty programs, and early customer engagement strategies |
| **CashbackAmount (Rewards)** | Mostly around lower cashback levels, fewer customers receive high rewards | More concentrated around mid to higher cashback values | Customers receiving lower cashback rewards tend to churn more | Introduce better reward systems, tiered cashback, or personalized incentives |
| **WarehouseToHome (Delivery Distance)** | Wider spread with more customers at higher distance values | Mostly concentrated at shorter distances | Longer delivery distance may lead to poorer service experience and higher churn risk | Improve logistics efficiency, reduce delivery time, and optimize warehouse distribution |
| **Complain (Customer Complaints)** | Customers who complain show a higher proportion of churn | Majority of retained customers do not complain | Negative service experience strongly correlates with churn | Strengthen customer support, faster complaint resolution, and proactive service recovery |
| **DaySinceLastOrder (Customer Activity)** | Many churned customers show long inactivity periods | Active customers place orders more frequently | Low engagement and long inactivity are strong churn signals | Implement re-engagement campaigns, personalized offers, and reminder notifications |

## 🤖 Customer Churn Prediction

### 🎯 Feature Selection

Based on the Random Forest feature importance analysis, the top features selected were `Tenure`, `CashbackAmount`, `WarehouseToHome`, `Complain`, and `DaySinceLastOrder`.

These variables reflect customer engagement, service experience, and purchasing behavior, which are strongly associated with churn.

### 🤖 Create a Model for Predicting Churn

[In 22]:

```python
# Select top features affecting Churn
top_features = ['Tenure', 'CashbackAmount', 'WarehouseToHome', 'Complain', 'DaySinceLastOrder']
x_1 = df[top_features]
y_1 = df['Churn']

# Split: 70% train, 30% temp (val + test)
x_train1, x_val1, y_train1, y_val1 = train_test_split(x_1, y_1, test_size=0.3, random_state=42)

# Split temp into 15% val, 15% test
x_val1, x_test1, y_val1, y_test1 = train_test_split(x_val1, y_val1, test_size=0.5, random_state=42)

# Normalize data
from sklearn.preprocessing import StandardScaler
scaler_1 = StandardScaler()

# Fit on training set only
x_train1_scaled = scaler_1.fit_transform(x_train1)

# Transform validation and test set
x_val1_scaled = scaler_1.transform(x_val1)
x_test1_scaled = scaler_1.transform(x_test1)
```

### 🤖 Model Comparison & Selection

Several machine learning models will be used and evaluated to identify the most effective approach for predicting customer churn: `Logistic Regression`, `K-Nearest Neighbors (KNN)`, `Random Forest`, and `Gradient Boosting`.

Each model was trained on the **scaled training data** and evaluated using **Recall Score** on the validation set. Recall was prioritized because correctly identifying churn customers is critical for retention strategies.

[In 23]:

```python
# Initialize models
models = {
    "Logistic Regression": LogisticRegression(),
    "KNN": KNeighborsClassifier(),
    "Random Forest": RandomForestClassifier(random_state=42),
    "Gradient Boosting": GradientBoostingClassifier()
}

# Dictionary to store recall scores
recall_scores = {}

# Train and evaluate each model
for name, model in models.items():
    
    # Train model
    model.fit(x_train1_scaled, y_train1)
    
    # Predict on validation set
    y_pred = model.predict(x_val1_scaled)
    
    # Calculate recall
    recall = recall_score(y_val1, y_pred)
    
    # Save result
    recall_scores[name] = recall

# Convert results to dataframe
    model_comparison = pd.DataFrame(
    recall_scores.items(),
    columns=["Model", "Recall Score"]
)

# Sort by recall
model_comparison = model_comparison.sort_values(by="Recall Score", ascending=False)
model_comparison
```

[Out 23]:

<img width="280" alt="image" src="https://github.com/user-attachments/assets/e3ba4e64-1461-45cf-a378-fca52fa2da4e" />

After testing 4 models, Random Forest achieved the highest Recall score.

→ Based on the comparison results, **Random Forest** was selected as the final model for churn prediction, followed by **fine-tuning** to further improve its performance.

### 🤖 Apply Model & Fine tune

#### Apply XGBoost

An **XGBoost Classifier** is trained on the scaled training data and evaluated on the validation set using **Recall Score**.

[In 24]:

```python
# Initialize the XGBoost model
xgb_model = xgb.XGBClassifier()

# Train the model with the scaled training data
xgb_model.fit(x_train1_scaled, y_train1)

# Make predictions on the validation set
y_pred_val_xgb = xgb_model.predict(x_val1_scaled)

# Evaluate the model using the recall score
recall_XGB = recall_score(y_val1, y_pred_val_xgb)
```

#### Fine-tune XGBoost Model

The **XGBoost model** is optimized using **RandomizedSearchCV** to search for the best hyperparameter combination. Model performance is evaluated using **Recall Score** with cross-validation.

[In 25]:

```python
# Set param
param_xgb = {
    'n_estimators': [50, 100, 200, 300, 500],
    'max_depth': [3, 5, 7, 10],
    'learning_rate': [0.01, 0.05, 0.1, 0.2],
    'subsample': [0.6, 0.8, 1.0],
    'colsample_bytree': [0.6, 0.8, 1.0],
    'gamma': [0, 0.1, 0.2, 0.3],
    'min_child_weight': [1, 3, 5]}

# Perform a randomized search over the specified parameter grid
rf_finetune = RandomizedSearchCV(estimator=xgb_model, param_distributions=param_xgb, n_iter=20, cv=5, scoring='recall', random_state=42)

# Fit the randomized search model on the training data
rf_finetune.fit(x_train1_scaled, y_train1)

# Print the best hyperparameters found by the search
print("Best parameters found:", rf_finetune.best_params_)

# Print the best recall score obtained during cross-validation
print("Best recall score:", rf_finetune.best_score_)
```

[Out 25]:

<img width="950" alt="image" src="https://github.com/user-attachments/assets/ae08ffa0-f11a-47c4-acae-ef9ef012144e" />

#### 💡 Insight

The optimized **XGBoost** model achieved a Recall of 0.74, successfully identifying about **74% of churned customers**.  
→ **Decision:** The optimized **XGBoost model** will be selected as the final model for churn prediction.

## 📊 Churned Customer Segmentation Using Clustering

Cluster analysis is performed on churned customers to identify distinct behavioral groups and uncover potential churn patterns.

### 🔎 Filter Churned Customers

Customers with `Churn = 1` are filtered to focus the analysis on churned users only.

[In 26]:

```python
# Filter churned customers
churned_data = df[df['Churn'] == 1][top_features]
churned_data.head()
```

[Out 26]:

<img width="350" alt="image" src="https://github.com/user-attachments/assets/b0016c83-b261-49a5-b7d0-eabdb00146ba" />

### ⚙️ Scale Data

Selected features are standardized using `StandardScaler` so that all variables contribute equally to the clustering process.

[In 27]:

```python
# Standardize the churned customer data so that all features have the same scale
scaled_data = scaler.fit_transform(churned_data)
```

### 📉 Dimension Reduction: PCA
Principal Component Analysis (PCA) is applied to reduce dimensionality while preserving the main variance in the dataset.

[In 28]:

```python
# Initialize PCA to reduce the dataset to 4 principal components
pca = PCA(n_components=4)

# Fit PCA on the scaled data and transform it into principal components
pca_components = pca.fit_transform(scaled_data)

# Fit PCA model to the scaled data (learn the variance structure)
pca.fit(scaled_data)

# Transform the scaled data into a DataFrame of principal components
pca_df = pd.DataFrame(pca.transform(scaled_data), columns=(["col1","col2","col3","col4"]))

# Display the PCA-transformed dataset
pca_df
```

[Out 28]:

<img width="420" alt="image" src="https://github.com/user-attachments/assets/e88ae8b9-59b9-46e5-b42a-44ca8506e5f0" />

### 📊 Explained Variance of PCA Components

The `explained_variance_ratio_` shows how much variance (information) each principal component captures from the original dataset. This helps evaluate whether the selected number of components retains enough information.

[In 29]:

```python
pca.explained_variance_ratio_
```
[Out 29]:

<img width="370" alt="image" src="https://github.com/user-attachments/assets/cb5f74a7-5a82-440d-80b1-1cd33386d010" />

#### 💡 Insight

The first four PCA components explain approximately **89% of the total variance**, indicating that most of the information from the original features is retained.

### 🤖 KMeans Clustering
K-Means clustering is applied to group churned customers into segments based on their behavioral features.

[In 30]:

```python
# Import clustering and visualization libraries
from sklearn.cluster import KMeans
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import AgglomerativeClustering
from matplotlib.colors import ListedColormap

# List to store WCSS (Within-Cluster Sum of Squares)
ss = []

# Define the maximum number of clusters to test
max_cluster = 10

# Compute WCSS for different numbers of clusters (k)
for i in range(1, max_cluster+1):
    # Initialize KMeans with k clusters
    kmeans = KMeans(n_clusters=i, random_state=42)
    
    # Fit KMeans on the PCA-transformed dataset
    kmeans.fit(pca_df)
    
    # Store the inertia (WCSS) for the elbow method
    ss.append(kmeans.inertia_)
```

### 📊 Determine Optimal Clusters (Elbow Method)
Use the Elbow Method to identify the optimal number of clusters.

[In 31]:

```python
# Plot the Elbow method
plt.figure(figsize=(10,6))
plt.plot(range(1, max_cluster+1), ss, marker='o')
plt.title('Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()
```

[Out 31]:

<img width="720" alt="image" src="https://github.com/user-attachments/assets/96118bca-dab5-4df6-9687-7e4927533a2c" />

#### 💡 Elbow Method Insight

- The WCSS decreases sharply when the number of clusters increases from **1 to 4**, indicating improved clustering performance.
- After **k = 4**, the decrease in WCSS becomes more gradual, meaning additional clusters provide limited improvement.

**Conclusion:**  
The optimal number of clusters is approximately **4**, as this point represents the "elbow" where adding more clusters yields diminishing returns.

### 🤖 K-Means Clustering with k = 4

Based on the Elbow Method, **k = 4** was selected as the optimal number of clusters. The **K-Means algorithm** is applied to the PCA-transformed dataset to segment customers into four groups. Each data point is assigned a **cluster label (0–3)** representing its segment.

[In 32]:

```python
# Initialize the K-Means model with 4 clusters
kmeans = KMeans(n_clusters=4, random_state=42)

# Fit the model to the PCA-transformed dataset and predict cluster labels
clusters = kmeans.fit_predict(pca_df)

# Add the cluster labels to the PCA dataframe
pca_df['cluster'] = clusters

# Also append the cluster labels to the original churned dataset
churned_data['cluster'] = clusters
```

### 📈 Cluster Visualization (PCA)

The scatter plot visualizes customer segments using the **first two principal components (PC1 and PC2)**. Each color represents a **different customer cluster**. The visualization shows that customers can be grouped into **four distinct behavioral segments**, indicating meaningful patterns in the data.

[In 33]:

```python
# Create a figure for the scatter plot
plt.figure(figsize=(8,6))

# Plot PCA components with cluster labels
sns.scatterplot(
    data=pca_df,
    x='col1',        # First principal component (PC1)
    y='col2',        # Second principal component (PC2)
    hue='cluster',   # Color points by cluster label
    palette='Set2'   # Color palette for different clusters
)

# Add a title to the visualization
plt.title('Customer Segments (KMeans Clustering)')

# Display the plot
plt.show()
```

[Out 33]:

<img width="850" alt="image" src="https://github.com/user-attachments/assets/15a28d27-fb14-4d46-bca5-53e34e5a3798" />


#### 💡 Customer Segmentation Insight (K-Means Clustering)

- The visualization shows **four distinct customer clusters**, indicating that customers can be grouped based on similar behavioral patterns.
- **Cluster 0 (green)** spreads further on the right side of the PCA space, suggesting customers with **distinct characteristics compared to other groups**.
- **Cluster 1 (orange)** is concentrated near the center, representing a **large group of customers with average behavior**.
- **Cluster 2 (blue)** appears in the lower-left area, indicating a **segment with similar but slightly different patterns from the central group**.
- **Cluster 3 (pink)** forms a group in the upper region, suggesting **another unique customer segment with different behavior patterns**.

### 📊 Cluster Size Distribution

[In 34]:

```python
# Count the number of data points in each cluster
pca_df['cluster'].value_counts()
```

[Out 34]:
<img width="280" alt="image" src="https://github.com/user-attachments/assets/cde7fe23-4438-4612-9ea6-7dbec05447ce" />

#### 💡 Insight:
Clusters 2 and 3 contain the largest number of customers, indicating that most customers fall into these behavioral segments, while Cluster 0 represents a smaller and more distinct group.

---

### 📈 Customer Cluster Profiling

This table presents the **average characteristics of customers in each cluster**, helping identify behavioral differences between customer segments.

[In 35]:

```python
# Calculate the average values of each feature for every cluster
cluster_profile = churned_data.groupby('cluster').mean()

# Display the cluster profile
cluster_profile
```

[Out 35]:
<img width="750" alt="image" src="https://github.com/user-attachments/assets/73388a60-b3b1-4ffb-b5a0-9d27749bae60" />

#### 💡 Cluster Interpretation

Based on the average feature values of each cluster, we can interpret the customer segments as follows:

**Cluster 0 – Long-term but inactive customers**
- Highest **Tenure (13.68)** and **CashbackAmount (218.15)**
- Relatively high **DaysSinceLastOrder (6.54)**
→ These customers have been with the company for a long time and receive high cashback benefits, but they have not placed orders recently, indicating a potential risk of becoming inactive.

**Cluster 1 – Customers with frequent complaints**
- **Complain rate = 1.00 (highest among all clusters)**
- Low tenure and moderate cashback
→ This group represents customers who frequently complain, suggesting possible dissatisfaction with services or products.

**Cluster 2 – Low-engagement customers**
- Lowest **Tenure (2.10)** and **CashbackAmount (150.19)**
- **No complaints (0.00)** but very low activity
→ These customers appear to be new or low-engagement users with minimal interaction.

**Cluster 3 – Customers far from warehouse with moderate engagement**
- Highest **WarehouseToHome distance (28.85)**
- Moderate tenure and cashback
- Relatively high complaint rate
→ These customers live farther from the warehouse, which may influence delivery experience and satisfaction.

---

### 🎯 Cluster Centroids Visualization

The red points represent the **centroids of each cluster**, which are the central positions of customer groups in the PCA feature space.  
These centroids summarize the **typical characteristics of customers within each segment**.

[In 36]:

```python
# Extract the centroid coordinates of each cluster
centroids = kmeans.cluster_centers_

# Plot the centroids on the PCA scatter plot
plt.scatter(centroids[:,0], centroids[:,1], 
            s=200, c='red', label='Centroids')

# Add label for each centroid
for i in range(len(centroids)):
    plt.text(centroids[i,0], centroids[i,1], f'Cluster {i}')

# Add chart title
plt.title('Cluster Centroids in PCA Space')

# Show legend
plt.legend()

# Display plot
plt.show()
```

[Out 36]:
<img width="770" alt="image" src="https://github.com/user-attachments/assets/b4463fa0-65e4-480b-a65b-4aa94b6837e6" />


#### 💡 Key Insight

- Each **red point represents a cluster centroid**, which indicates the **average position of customers** within that segment in the PCA space.
- **Cluster 0** is positioned far to the right, suggesting this segment has **distinct characteristics compared to other clusters**.
- **Cluster 1** and **Cluster 2** appear relatively close to each other, indicating these two segments may have **similar behavioral patterns**.
- **Cluster 3** is located higher on the PCA space, representing **another unique customer group**.

The **clear separation between centroids** suggests that the K-Means algorithm successfully identified **distinct customer segments**, with each centroid representing the **typical profile of a cluster**.

## 📈 Churned Customer Segmentation Visualization

### Chart: Cluster Feature Comparison

This heatmap compares the **average values of key features across customer clusters**. Each row represents a cluster, and each column represents a feature.  
The color intensity indicates the magnitude of the values, helping highlight differences in customer behavior among clusters.

[In 37]:

```python
# Create a heatmap to visualize and compare feature values across clusters
plt.figure(figsize=(8,5))
sns.heatmap(cluster_profile, annot=True, cmap='coolwarm', fmt='.2f')

# Add title to describe the chart
plt.title('Cluster Feature Comparison')

# Display the heatmap
plt.show()
```

[Out 37]:
<img width="900" alt="image" src="https://github.com/user-attachments/assets/f4ba5cbc-35af-4989-bdd6-425f1735d7f0" />

#### 💡 Insight

The heatmap highlights clear behavioral differences across clusters.  
Cluster 0 shows the **highest tenure and cashback amount**, along with the **longest time since last order**.  
Cluster 1 has the **highest complaint rate**, indicating potential service dissatisfaction.  
Cluster 2 exhibits **low values across most features**, suggesting lower engagement.  
Cluster 3 has the **largest warehouse-to-home distance** and relatively **high cashback**.

### Chart: Distribution of Key Features by Cluster

These boxplots show how key features are distributed across different clusters.  
They help visualize variations, central tendencies, and potential outliers, providing deeper insight into the behavioral characteristics of each customer segment.

[In 38]:

```python
# Create boxplots to visualize feature distributions across clusters
features = ['Tenure','CashbackAmount','WarehouseToHome','DaySinceLastOrder']

# Create a 2x2 layout for the charts
fig, axes = plt.subplots(2, 2, figsize=(12,8))
axes = axes.flatten()

for i, col in enumerate(features):
    
    # Plot boxplot for each feature by cluster
    sns.boxplot(
        x='cluster',
        y=col,
        data=churned_data,
        ax=axes[i]
    )

    # Add title for each subplot
    axes[i].set_title(f'{col} Distribution by Cluster')

# Add overall title for the figure
fig.suptitle('Feature Distribution Across Clusters')

# Adjust layout for better spacing
plt.tight_layout()

# Display the charts
plt.show()
```

[Out 38]:
<img width="943" alt="image" src="https://github.com/user-attachments/assets/5d22c906-db9d-4feb-846a-09e9b9066e8f" />

#### 💡 Insight

The boxplots reveal clear distribution differences across clusters.  
Cluster 0 shows the **highest tenure and cashback values**, indicating long-term high-value customers.  
Cluster 1 and 2 exhibit **lower tenure and cashback**, suggesting newer or lower-value customers.  
Cluster 3 has the **largest warehouse-to-home distance**, while Cluster 0 also shows the **longest time since last order**.

### Chart: Cluster Size Visualization

This chart shows the number of customers in each cluster.  
It helps understand the **size of each customer segment** and whether the clusters are **balanced or dominated by a particular group**.

[In 39]:

```python

# Create a figure with specified size
plt.figure(figsize=(6,4))

# Generate a countplot to show the number of customers in each cluster
ax = sns.countplot(
    x='cluster',
    data=churned_data,
    palette='Set2'
)

# Loop through each bar in the chart
for p in ax.patches:
    
    # Add a text label showing the count value on top of each bar
    ax.annotate(
        f'{int(p.get_height())}',                 # Text: number of customers
        (p.get_x() + p.get_width() / 2,           # X position: center of the bar
         p.get_height()),                         # Y position: top of the bar
        ha='center',                              # Horizontally center the text
        va='bottom'                               # Place text slightly above the bar
    )

# Add title to describe the chart
plt.title('Customer Distribution by Cluster')

# Add axis labels
plt.xlabel('Cluster')
plt.ylabel('Number of Customers')

# Display the chart
plt.show()
```

[Out 39]:
<img width="900" alt="image" src="https://github.com/user-attachments/assets/69342107-e9d6-4415-89de-985d47130a0d" />

#### **Insight**

The chart shows an uneven distribution of customers across clusters.  
Cluster 2 and Cluster 1 represent the **largest segments**, while Cluster 3 forms a **moderate-sized group**.  
Cluster 0 is the **smallest segment**, indicating that long-tenure customers make up a smaller portion of the churned population.

---

### 📈 Final Clustering Insights

#### 💡 Key Insights from Customer Segmentation

Customer segmentation identified **four distinct churned customer groups**, each with different behavioral characteristics and potential churn drivers.

**Cluster 0 – Previously loyal but inactive customers**
- Highest tenure and cashback amount
- Longest time since last order
- Smallest group of customers

→ These customers were long-term and high-value but recently became inactive, indicating a need for **re-engagement strategies**.

**Cluster 1 – Dissatisfied active customers**
- Highest complaint rate
- Moderate cashback and tenure
- One of the largest customer groups

→ Service dissatisfaction appears to be the main factor influencing churn in this segment.

**Cluster 2 – Low engagement customers**
- Very low tenure and activity
- Lowest complaint rate
- Largest customer group

→ These customers show minimal engagement with the platform and may have churned due to **lack of strong attachment or usage**.

**Cluster 3 – Distance-sensitive customers**
- Largest warehouse-to-home distance
- Moderate cashback and tenure
- Mid-sized customer segment

→ Delivery distance may negatively impact customer experience, suggesting **logistics or delivery improvements** could help retention.

## 💡 Key Insights & Business Recommendations

| Insight | Evidence from Analysis | Business Recommendation |
|--------|-----------------------|-------------------------|
| **Low tenure customers churn more easily** | Customers with short tenure appear frequently in high-risk churn clusters | Improve onboarding experience and provide early engagement incentives for new users |
| **Customer complaints strongly relate to churn** | High churn rate observed among customers who submitted complaints | Improve complaint handling process and implement faster support resolution |
| **Inactive customers show high churn risk** | Customers with longer *DaysSinceLastOrder* tend to appear in churn clusters | Launch re-engagement campaigns such as reminders, promotions, or personalized offers |
| **Delivery distance affects customer experience** | Customers with larger *WarehouseToHome* distance show signs of dissatisfaction | Optimize logistics, improve delivery time estimates, or expand warehouse coverage |
| **Rewards help improve retention** | Customers receiving higher *CashbackAmount* tend to be more engaged | Strengthen loyalty programs and offer targeted cashback incentives |
| **Customer segments behave differently** | Clustering reveals four distinct customer groups | Use cluster-based marketing strategies and personalized retention campaigns |


