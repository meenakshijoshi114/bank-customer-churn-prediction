

import joblib
import pandas as pd
import streamlit as st
 
st.set_page_config(page_title="Bank Customer Churn Predictor", page_icon="🏦", layout="centered")
 
 
@st.cache_resource
def load_model():
    bundle = joblib.load("model.joblib")
    return bundle["model"], bundle["columns"]
 
 
model, columns = load_model()
 
st.title("🏦 Bank Customer Churn Predictor")
st.write(
    "Enter a customer's details to estimate how likely they are to leave the bank. "
    "Model: Gradient Boosting (cross-validated ROC-AUC ≈ 0.87)."
)
 
# ---------- Inputs ----------
col1, col2 = st.columns(2)
 
with col1:
    geography = st.selectbox("Country", ["France", "Germany", "Spain"])
    gender = st.selectbox("Gender", ["Female", "Male"])
    age = st.slider("Age", 18, 92, 40)
    credit_score = st.slider("Credit score", 350, 850, 650)
    tenure = st.slider("Tenure (years with bank)", 0, 10, 5)
 
with col2:
    balance = st.number_input("Account balance", min_value=0.0, value=50000.0, step=1000.0)
    salary = st.number_input("Estimated salary", min_value=0.0, value=100000.0, step=1000.0)
    num_products = st.selectbox("Number of products", [1, 2, 3, 4], index=1)
    has_card = st.radio("Has credit card?", ["Yes", "No"], horizontal=True)
    is_active = st.radio("Active member?", ["Yes", "No"], horizontal=True)
 
threshold = st.slider(
    "Decision threshold",
    0.10, 0.90, 0.30, 0.05,
    help="Lower = flag more customers (catch more churners, more false alarms). "
         "0.3 gave the best F1 in testing.",
)
 
# ---------- Build the row exactly like the training data ----------
row = {
    "CreditScore": credit_score,
    "Age": age,
    "Tenure": tenure,
    "Balance": balance,
    "NumOfProducts": num_products,
    "HasCrCard": 1 if has_card == "Yes" else 0,
    "IsActiveMember": 1 if is_active == "Yes" else 0,
    "EstimatedSalary": salary,
    "Geography_Germany": 1 if geography == "Germany" else 0,
    "Geography_Spain": 1 if geography == "Spain" else 0,
    "Gender_Male": 1 if gender == "Male" else 0,
}
X_new = pd.DataFrame([row])[columns]  # same column order as training
 
# ---------- Prediction ----------
if st.button("Predict churn risk", type="primary"):
    prob = float(model.predict_proba(X_new)[0, 1])
    st.divider()
 
    st.metric("Churn probability", f"{prob:.1%}")
    st.progress(min(prob, 1.0))
 
    if prob >= threshold:
        st.error(f"⚠️ High risk: flagged as likely to churn (probability {prob:.1%} ≥ threshold {threshold:.0%}).")
    else:
        st.success(f"✅ Low risk: likely to stay (probability {prob:.1%} < threshold {threshold:.0%}).")
 
    # Simple risk factors based on patterns found in the data exploration
    notes = []
    if 45 <= age <= 60:
        notes.append("Age 45-60 is the highest-churn age group in this dataset.")
    if num_products >= 3:
        notes.append("Customers with 3-4 products churn at very high rates.")
    elif num_products == 1:
        notes.append("Customers with only 1 product churn more than those with 2.")
    if is_active == "No":
        notes.append("Inactive members churn about twice as often as active ones.")
    if geography == "Germany":
        notes.append("Customers in Germany churn about twice as often as in France or Spain.")
    if balance > 0 and gender == "Female":
        notes.append("Female customers with a non-zero balance show higher churn in this data.")
 
    if notes:
        st.subheader("Risk factors present")
        for n in notes:
            st.write("• " + n)
 
    with st.expander("Show model input"):
        st.dataframe(X_new)
 
st.caption(
    "Predictions come from patterns in a historical dataset and show correlation, not cause. "
    "Use as decision support, not as a final decision."
)
 

