# Intelligent Software Defect Tracking System with Resolution Assistance

##  Project Overview

The **Intelligent Software Defect Tracking System with Resolution Assistance** is a web-based software defect management application developed using **Python and Streamlit**.

The system helps software development teams analyze, identify, compare, and manage software bugs. It also provides duplicate bug detection and resolution recommendations based on the reported defect.

The application provides an interactive dashboard for analyzing bug statistics such as status, priority, sprint, module, root cause, resolution time, and developer performance.

---

## 🎯 Project Objectives

* Analyze software defect records.
* Identify newly reported bugs and classify their basic characteristics.
* Detect duplicate or similar bugs using text similarity.
* Analyze bug severity and priority.
* Identify possible root causes.
* Provide recommended resolutions for reported defects.
* Analyze bug resolution time.
* Provide interactive visualizations and KPIs.
* Allow users to search and download bug records.

---

## 🚀 Main Features

### 1. 📊 Dashboard

The dashboard provides an overview of software defects using interactive charts and KPIs.

It includes:

* Total Bugs
* Closed Bugs
* Average Resolution Time
* Defect Density
* Bug Status Distribution
* Priority Distribution
* Sprint-wise Bug Distribution
* Module-wise Defect Distribution
* Monthly Bug Resolution Trend
* Developer Performance
* Root Cause Analysis
* Actionable Insights

Users can filter the dashboard using:

* Sprint
* Module
* Priority

---

### 2. 🐞 Bug Identification

The Bug Identification module allows users to enter a new bug report.

Users can provide:

* Bug Title
* Affected Module
* Bug Description
* Severity
* Current Status

The system analyzes the bug description and identifies:

* Bug Category
* Affected Module
* Severity
* Suggested Priority
* Current Status

The system uses keyword-based analysis to identify categories such as:

* Authentication Issue
* Transaction / Checkout Issue
* API / Server Issue
* Database Issue
* Frontend / UI Issue
* Application Logic Issue

Priority is suggested based on severity:

| Severity | Priority |
| -------- | -------- |
| Critical | P1       |
| High     | P2       |
| Medium   | P3       |
| Low      | P4       |

---

### 3. 🔄 Duplicate Bug Detection

The Duplicate Bug Detection module compares a newly reported bug with existing bug descriptions in the dataset.

The system uses:

* TF-IDF Vectorization
* Cosine Similarity

The entered bug description is compared against existing bug descriptions.

The system displays:

* Duplicate Detection Result
* Similarity Percentage
* Closest Bug ID
* Module
* Severity
* Priority
* Status
* Top 5 Similar Existing Bugs

Similarity classification:

| Similarity   | Result              |
| ------------ | ------------------- |
| 70% or above | Possible Duplicate  |
| 40% - 69%    | Similar Bug         |
| Below 40%    | No Strong Duplicate |

This helps reduce duplicate bug reports and saves developer time.

---

### 4. 🤖 Resolution Assistance

The Resolution Assistance module provides recommendations for a reported defect.

Users enter:

* Bug Title
* Affected Module
* Bug Description
* Severity

The system provides:

* Severity
* Recommended Priority
* Affected Module
* Root Cause
* Recommended Resolution

Examples of root cause categories include:

* Authentication / Login Validation
* Payment / Transaction Processing Issue
* Database / Query Issue
* API / Server Issue
* Frontend / UI Issue
* Dashboard / Data Visualization Issue
* Performance Issue
* Application Logic Issue

The system also provides troubleshooting recommendations based on the affected module and bug description.

---

### 5. 📄 Bug Records

The Bug Records module allows users to view and search the available bug records.

Users can search by:

* Bug ID
* Module
* Status
* Priority
* Root Cause
* Bug Description

The module also displays:

* Total Records
* Open / Active Bugs
* High Priority Bugs

Users can download the filtered bug records as a CSV file.

---

## 🛠️ Technologies Used

### Programming Language

* Python

### Framework

* Streamlit

### Data Processing

* Pandas

### Data Visualization

* Plotly

### Machine Learning / Text Analysis

* Scikit-learn
* TF-IDF Vectorization
* Cosine Similarity

### Dataset

* CSV

---

## 📂 Project Structure

```text
Intelligent-Software-Defect-Tracking-System-with-Resolution-Assistance/
│
├── BugReport.csv
├── dashboard_app.py
├── requirements.txt
├── README.md
└── LICENSE
```

---


## ▶️ Run the Application

Run the Streamlit application using:

```bash
streamlit run dashboard_app.py
```

The application will open in your browser.

Usually, Streamlit runs at:

```text
http://localhost:8501
```

---

## 📊 Dataset

The project uses `BugReport.csv` as the main dataset.

The dataset contains software defect information used for:

* Bug analysis
* Bug status analysis
* Priority analysis
* Sprint analysis
* Module analysis
* Root cause analysis
* Resolution time analysis
* Duplicate bug detection

Important fields include:

```text
Bug_ID
Sprint
Module
Priority
Severity
Status
Root_Cause
Assigned_To
Date_Closed
Resolution_Time_Hours
Bug_Description
```

---

## 🔍 Duplicate Detection Method

The duplicate detection process follows these steps:

```text
New Bug Description
        ↓
Text Preprocessing
        ↓
TF-IDF Vectorization
        ↓
Cosine Similarity
        ↓
Compare with Existing Bugs
        ↓
Calculate Similarity Score
        ↓
Identify Closest Bug
        ↓
Duplicate / Similar / No Strong Duplicate
```

### TF-IDF

TF-IDF converts text descriptions into numerical vectors based on the importance of words in the bug reports.

### Cosine Similarity

Cosine similarity measures how similar two bug descriptions are.

A higher similarity score indicates that the two bug descriptions are more similar.

---

## 🧠 Resolution Assistance Process

```text
User Reports Bug
        ↓
Bug Title + Description + Module + Severity
        ↓
Analyze Bug Information
        ↓
Recommend Priority
        ↓
Identify Possible Root Cause
        ↓
Generate Resolution Recommendation
```

---

## 📈 Dashboard Analysis

The dashboard helps development teams answer questions such as:

* Which module has the highest number of bugs?
* Which priority level has the most defects?
* Which sprint contains the most bugs?
* How many bugs are closed?
* What is the average resolution time?
* Which developers have higher average resolution time?
* What are the most common root causes?
* How many bugs are currently active?

---

## 💡 Benefits

* Reduces duplicate bug reporting.
* Improves bug tracking.
* Helps prioritize critical defects.
* Provides quick root cause suggestions.
* Provides resolution assistance.
* Helps identify problematic modules.
* Supports software quality analysis.
* Provides interactive visual analytics.
* Saves time during defect investigation.

---

## 🔮 Future Enhancements

The project can be extended with:

* Machine Learning based severity prediction.
* Machine Learning based priority prediction.
* Advanced NLP-based bug classification.
* Automatic root cause prediction.
* Automatic resolution recommendation using trained models.
* User authentication and role-based access.
* Database integration using MySQL or PostgreSQL.
* Real-time bug tracking.
* Email notifications.
* REST API integration.
* Cloud deployment.
* Advanced duplicate detection using transformer models.

---

## 📜 License

This project is licensed under the MIT License.
