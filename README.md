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

<details>
<summary><strong>📋 Click to expand table schema</strong></summary>

<br>

### Table: Customer Churn Data

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
| PreferedOrderCat | OBJECT | Preferred order category for the customer |
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
