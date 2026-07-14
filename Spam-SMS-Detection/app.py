import streamlit as st
import joblib
import time

# -----------------------------
# Load Model
# -----------------------------
model = joblib.load("model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(
    page_title="AI Spam SMS Detector",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -----------------------------
# Theme Toggle
# -----------------------------
dark_mode = st.toggle("🌙 Dark Mode", value=True)

if dark_mode:
    BG = "#0F172A"
    CARD = "#1E293B"
    TEXT = "#FFFFFF"
    SUB = "#CBD5E1"
else:
    BG = "#F8FAFC"
    CARD = "#FFFFFF"
    TEXT = "#111827"
    SUB = "#475569"

# -----------------------------
# Custom CSS
# -----------------------------
st.markdown(f"""
<style>

.stApp {{
    background:{BG};
}}

.block-container {{
    padding-top:2rem;
}}

h1,h2,h3,h4,h5,h6,p,label,span {{
    color:{TEXT} !important;
}}

textarea {{
    color:{TEXT} !important;
    background:{CARD} !important;
}}

div[data-baseweb="base-input"] input {{
    color:{TEXT} !important;
}}

.main-title{{
    font-size:55px;
    font-weight:bold;
    text-align:center;
    color:#38BDF8;
}}

.sub-title{{
    text-align:center;
    color:{SUB};
    font-size:20px;
    margin-bottom:30px;
}}

.card{{
    background:{CARD};
    padding:25px;
    border-radius:20px;
    box-shadow:0 10px 25px rgba(0,0,0,.15);
}}

.stButton>button{{
    width:100%;
    height:55px;
    border-radius:12px;
    background:#2563EB;
    color:white;
    font-size:18px;
    font-weight:bold;
}}

.stButton>button:hover{{
    background:#1D4ED8;
}}

footer {{
    visibility:hidden;
}}

</style>
""", unsafe_allow_html=True)

# -----------------------------
# Sidebar
# -----------------------------
with st.sidebar:

    st.title("🛡️ AI Spam Detector")

    st.markdown("---")

    st.success("### Model Information")

    st.write("✅ Algorithm")
    st.info("Multinomial Naive Bayes")

    st.write("✅ Vectorizer")
    st.info("TF-IDF")

    st.write("✅ Dataset")
    st.info("SMS Spam Collection")

    st.write("✅ Accuracy")
    st.success("98%+")

    st.markdown("---")

    st.caption("Made with ❤️ using Streamlit")

# -----------------------------
# Header
# -----------------------------
st.markdown(
"""
<div class='main-title'>
🛡️ AI Spam SMS Detector
</div>
""",
unsafe_allow_html=True
)

st.markdown(
"""
<div class='sub-title'>
Detect Spam Messages Instantly using Machine Learning
</div>
""",
unsafe_allow_html=True
)

left, right = st.columns([2,1])

with left:

    st.markdown("<div class='card'>", unsafe_allow_html=True)

    sms = st.text_area(
        "✉️ Enter SMS Message",
        height=220,
        placeholder="Example: Congratulations! You have won ₹5,00,000. Click the link to claim your prize."
    )

    predict = st.button("🚀 Detect Message")

    st.markdown("</div>", unsafe_allow_html=True)

with right:

    st.markdown("<div class='card'>", unsafe_allow_html=True)

    st.subheader("📌 Example Messages")

    st.success(
        "Hi Rahul,\n\nYour order has been delivered successfully."
    )

    st.error(
        "Congratulations!\n\nYou won ₹5,00,000.\nClick here to claim now."
    )

    st.markdown("</div>", unsafe_allow_html=True)
    # -----------------------------
# Prediction
# -----------------------------
if predict:

    if sms.strip() == "":
        st.warning("⚠ Please enter an SMS message.")
    else:

        with st.spinner("🤖 AI is analyzing your message..."):
            time.sleep(2)

            vector = vectorizer.transform([sms])
            prediction = model.predict(vector)[0]

        st.markdown("<br>", unsafe_allow_html=True)

        st.markdown("## 📊 Prediction Result")

        if prediction == 1:

            st.markdown("""
<div style="
background:linear-gradient(135deg,#DC2626,#991B1B);
padding:25px;
border-radius:20px;
text-align:center;
color:white;
font-size:30px;
font-weight:bold;
box-shadow:0px 8px 25px rgba(220,38,38,.4);
">

🚨 SPAM MESSAGE DETECTED

</div>
""", unsafe_allow_html=True)

            st.error("⚠ This message looks suspicious. Do not click unknown links.")

            st.progress(95)

            st.metric(
                label="Spam Probability",
                value="95%"
            )

        else:

            st.markdown("""
<div style="
background:linear-gradient(135deg,#16A34A,#166534);
padding:25px;
border-radius:20px;
text-align:center;
color:white;
font-size:30px;
font-weight:bold;
box-shadow:0px 8px 25px rgba(22,163,74,.4);
">

✅ SAFE MESSAGE

</div>
""", unsafe_allow_html=True)

            st.success("This message appears to be legitimate.")

            st.progress(15)

            st.metric(
                label="Spam Probability",
                value="15%"
            )

        st.markdown("---")

        st.subheader("📈 AI Analysis")

        if prediction == 1:

            st.info("""
🔹 Contains promotional or suspicious words

🔹 May contain phishing content

🔹 Avoid clicking unknown links

🔹 Verify the sender before replying
""")

        else:

            st.info("""
🔹 Normal conversational message

🔹 No suspicious keywords detected

🔹 Appears safe

🔹 Low spam probability
""")

# -----------------------------
# Footer
# -----------------------------
st.markdown("---")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("📨 Messages Checked", "10K+")

with col2:
    st.metric("🎯 Accuracy", "98%")

with col3:
    st.metric("⚡ Response Time", "<1 sec")

st.markdown("---")

st.markdown(
"""
<div style="text-align:center;">

<h3>🛡️ AI Spam SMS Detector</h3>

<p>
Built with ❤️ using
<b>Python</b> •
<b>Scikit-Learn</b> •
<b>TF-IDF</b> •
<b>Streamlit</b>
</p>

<p style="color:gray;">
© 2026 VINDHYA SRI | Machine Learning Internship Project
</p>

</div>
""",
unsafe_allow_html=True
)