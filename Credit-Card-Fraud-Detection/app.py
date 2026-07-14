import streamlit as st
import pandas as pd
import joblib
import time

# ----------------------------
# Load Model
# ----------------------------
model = joblib.load("model.pkl")
category_encoder = joblib.load("category_encoder.pkl")
gender_encoder = joblib.load("gender_encoder.pkl")

# ----------------------------
# Page Config
# ----------------------------
st.set_page_config(
    page_title="AI Credit Card Fraud Detection",
    page_icon="💳",
    layout="wide"
)

# ----------------------------
# CSS
# ----------------------------
st.markdown("""
<style>

.stApp{
background:linear-gradient(135deg,#0F172A,#1E293B,#111827);
color:white;
}

.title{
text-align:center;
font-size:48px;
font-weight:bold;
color:#38BDF8;
}

.subtitle{
text-align:center;
font-size:18px;
color:#CBD5E1;
margin-bottom:30px;
}

.card{
background:rgba(255,255,255,.08);
padding:25px;
border-radius:20px;
backdrop-filter:blur(15px);
border:1px solid rgba(255,255,255,.15);
}

.result{
padding:20px;
border-radius:15px;
text-align:center;
font-size:28px;
font-weight:bold;
margin-top:20px;
}

.stButton>button{
width:100%;
height:55px;
background:#2563EB;
color:white;
font-size:20px;
border-radius:12px;
}

.stButton>button:hover{
background:#1D4ED8;
}

footer{
visibility:hidden;
}

</style>
""", unsafe_allow_html=True)

# ----------------------------
# Header
# ----------------------------
st.markdown("<div class='title'>💳 AI Credit Card Fraud Detection</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Detect fraudulent transactions using Machine Learning</div>", unsafe_allow_html=True)

left,right = st.columns([2,1])

# ----------------------------
# Left Card
# ----------------------------
with left:

    st.markdown("<div class='card'>",unsafe_allow_html=True)

    category = st.selectbox(
        "🛒 Transaction Category",
        category_encoder.classes_
    )

    amount = st.number_input(
        "💰 Transaction Amount ($)",
        min_value=0.0,
        value=100.0
    )

    gender = st.selectbox(
        "👤 Customer Gender",
        gender_encoder.classes_
    )

    predict = st.button("🔍 Detect Fraud")

    st.markdown("</div>",unsafe_allow_html=True)

# ----------------------------
# Right Card
# ----------------------------
with right:

    st.markdown("<div class='card'>",unsafe_allow_html=True)

    st.subheader("📊 Model Information")

    st.write("✅ Algorithm : Random Forest")

    st.write("📂 Dataset : Credit Card Transactions")

    st.write("🎯 Accuracy : 99.44%")

    st.write("⚡ Prediction : Real-Time")

    st.markdown("</div>",unsafe_allow_html=True)

# ----------------------------
# Prediction
# ----------------------------
if predict:

    with st.spinner("🤖 AI is analysing transaction..."):
        time.sleep(2)

    category_value = category_encoder.transform([category])[0]
    gender_value = gender_encoder.transform([gender])[0]

    sample = pd.DataFrame(
        [[category_value, amount, gender_value]],
        columns=["category","amt","gender"]
    )

    prediction = model.predict(sample)[0]

    if prediction == 1:

        st.markdown("""
<div class='result' style='background:#DC2626;color:white;'>
🚨 FRAUD DETECTED
</div>
""",unsafe_allow_html=True)

        st.error("High Risk Transaction!")

        st.progress(95)

    else:

        st.markdown("""
<div class='result' style='background:#16A34A;color:white;'>
✅ LEGITIMATE TRANSACTION
</div>
""",unsafe_allow_html=True)

        st.success("Transaction appears safe.")

        st.progress(20)

st.markdown("---")

st.markdown(
"<center>💻 Developed using Python • Scikit-Learn • Streamlit</center>",
unsafe_allow_html=True
)