import streamlit as st
import numpy as np
import pickle
import os

# ---------------- PAGE CONFIG ---------------- #
st.set_page_config(
    page_title="Loan Eligibility",
    layout="centered"
)

# ---------------- CLEAN CSS ---------------- #
st.markdown("""
<style>
.stApp {
    background-color: #f5f7fa;
}

.container {
    background: white;
    padding: 2rem;
    border-radius: 12px;
    box-shadow: 0 2px 10px rgba(0,0,0,0.06);
}

.title {
    font-size: 1.8rem;
    font-weight: 600;
    color: #1f2d3d;
}

.subtitle {
    color: #6b7c93;
    font-size: 0.95rem;
    margin-bottom: 1.5rem;
}

.section {
    margin-top: 1.5rem;
    font-weight: 600;
    color: #34495e;
}

.metric-box {
    background: #f1f5f9;
    padding: 10px;
    border-radius: 8px;
    text-align: center;
}

button[kind="primary"] {
    background-color: #2c3e50;
    color: white;
    border-radius: 8px;
    height: 45px;
    font-weight: 600;
}

</style>
""", unsafe_allow_html=True)

# ---------------- HEADER ---------------- #
st.markdown('<div class="title">Loan Eligibility Assessment</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Provide your financial and personal details to evaluate loan approval likelihood.</div>', unsafe_allow_html=True)

# ---------------- LOAD MODEL ---------------- #
@st.cache_resource
def load():
    model = pickle.load(open("loan_model.pkl", "rb"))
    scaler = pickle.load(open("scaler.pkl", "rb"))
    return model, scaler

model, scaler = load()

# ---------------- INPUT FORM ---------------- #
st.markdown('<div class="section">Personal Information</div>', unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)
gender = col1.selectbox("Gender", ["Male", "Female"])
married = col2.selectbox("Marital Status", ["Yes", "No"])
dependents = col3.selectbox("Dependents", ["0", "1", "2", "3+"])

col4, col5 = st.columns(2)
education = col4.selectbox("Education", ["Graduate", "Not Graduate"])
self_employed = col5.selectbox("Self Employed", ["No", "Yes"])

# ---------------- FINANCIAL ---------------- #
st.markdown('<div class="section">Financial Information</div>', unsafe_allow_html=True)

col6, col7 = st.columns(2)
applicant_income = col6.number_input("Applicant Income (₹ per month)", value=5000, step=500)
coapplicant_income = col7.number_input("Co-applicant Income (₹ per month)", value=0, step=500)

# ---------------- LOAN ---------------- #
st.markdown('<div class="section">Loan Details</div>', unsafe_allow_html=True)

col8, col9 = st.columns(2)
loan_amount = col8.number_input(
    "Loan Amount (in ₹ thousands)",
    min_value=9,
    max_value=700,
    value=150,
    step=10
)

loan_term = col9.selectbox(
    "Loan Term (months)",
    [12, 36, 60, 120, 180, 240, 300, 360]
)

col10, col11 = st.columns(2)
credit_history = col10.selectbox("Credit History", ["Good", "Bad"])
property_area = col11.selectbox("Property Area", ["Urban", "Semiurban", "Rural"])

# ---------------- CALCULATIONS ---------------- #
total_income = applicant_income + coapplicant_income
emi = loan_amount / loan_term

colA, colB, colC = st.columns(3)
colA.metric("Total Income", f"₹{total_income:,.0f}")
colB.metric("EMI (₹ thousands/month)", f"{emi:.2f}")
colC.metric("EMI / Income (%)", f"{(emi*1000/total_income*100) if total_income else 0:.1f}")

# ---------------- PREDICTION ---------------- #
if st.button("Evaluate Loan Application"):

    # Encoding
    gender = 1 if gender == "Male" else 0
    married = 1 if married == "Yes" else 0
    education = 1 if education == "Graduate" else 0
    self_employed = 1 if self_employed == "Yes" else 0
    credit = 1 if credit_history == "Good" else 0

    dependents = 3 if dependents == "3+" else int(dependents)

    prop_rural = 1 if property_area == "Rural" else 0
    prop_semi = 1 if property_area == "Semiurban" else 0
    prop_urban = 1 if property_area == "Urban" else 0

    features = np.array([[
        gender, married, dependents, education, self_employed,
        applicant_income, coapplicant_income, loan_amount,
        loan_term, credit,
        prop_rural, prop_semi, prop_urban,
        total_income, emi
    ]])

    features = scaler.transform(features)

    pred = model.predict(features)[0]
    prob = model.predict_proba(features)[0]

    st.markdown("---")

    if pred == 1:
        st.success("Loan Approved")
    else:
        st.error("Loan Not Approved")

    st.write(f"Approval Probability: {prob[1]*100:.2f}%")

    # Explanation
    if total_income < 40000:
        st.warning("Low income may affect eligibility")

    if credit == 0:
        st.warning("Credit history has a strong impact on rejection")

# ---------------- FOOTER ---------------- #
st.markdown("---")
st.caption("This tool provides an indicative assessment based on historical data. Final decisions depend on lender policies.")