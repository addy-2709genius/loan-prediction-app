# Loan Eligibility Predictor

A machine learning web application that predicts whether a loan application is likely to be approved based on applicant details.

---

## Overview

This project demonstrates an end-to-end machine learning workflow — from data preprocessing and feature engineering to model building and deployment using Streamlit.

The application allows users to input financial and personal details and instantly receive a prediction along with approval probability.

---

## Features

* Real-time loan approval prediction
* Probability-based output
* Clean and user-friendly interface
* End-to-end ML pipeline integration

---

## Input Parameters

* Applicant Income (₹ per month)
* Co-applicant Income (₹ per month)
* Loan Amount (**in ₹ thousands**)
* Loan Term (**in months**)
* Credit History (Good / Bad)
* Personal details (Gender, Education, etc.)

---

## Tech Stack

* Python
* Streamlit
* Scikit-learn
* NumPy
* Pandas

---

## Challenges Faced & Solutions

### 1. Data Cleaning Issues

* Missing values in LoanAmount, Credit_History, and Dependents
* Inconsistent formats (e.g., "3+" in Dependents)

**Solution:**

* Filled numerical values using median
* Filled categorical values using mode
* Converted "3+" into numeric value (3)

---

### 2. Feature Engineering

* Raw dataset did not fully represent financial strength

**Solution:**

* Created `Total_Income = ApplicantIncome + CoapplicantIncome`
* Created `EMI = LoanAmount / Loan_Amount_Term`
* Improved model performance

---

### 3. Feature Mismatch Error

**Error:**
X has 14 features, but StandardScaler is expecting 15 features

**Cause:**
Mismatch between training and prediction features

**Solution:**

* Ensured same feature count and order
* Added missing engineered features

---

### 4. Loan Amount Unit Confusion

* Model trained on LoanAmount in **thousands**
* UI initially used full rupee values

**Solution:**

* Standardized input to ₹ thousands
* Clearly mentioned in UI

---

### 5. Loan Term Format Issue

* Initially used days instead of months

**Solution:**

* Corrected to months as per dataset

---

### 6. Virtual Environment Issue

* `pip` command not recognized

**Solution:**

* Activated virtual environment
* Used `python3 -m pip freeze > requirements.txt`

---

### 7. Git & GitHub Setup

* Needed proper version control

**Solution:**

* Created `.gitignore`
* Initialized Git and pushed to GitHub

---

### 8. Deployment Preparation

* Ensuring app runs in cloud

**Solution:**

* Added `requirements.txt`
* Structured project properly

---

## Key Learnings

* Importance of feature consistency between training and inference
* Handling real-world messy datasets
* Feature engineering improves model performance
* Building and deploying ML applications end-to-end

---

## Run Locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

---

## Project Structure

```
loan_model/
│
├── app.py
├── loan_model.pkl
├── scaler.pkl
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Future Improvements

* Add explanation for why loan is rejected
* Improve model accuracy
* Enhance UI with analytics dashboard

---

## Disclaimer

This tool provides an indicative prediction based on historical data.
Actual loan approval depends on lender policies and additional checks.
