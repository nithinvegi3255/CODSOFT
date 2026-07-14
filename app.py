import streamlit as st
import joblib
import time

# -----------------------------
# Load Model
# -----------------------------
model = joblib.load("model.pkl")
tfidf = joblib.load("tfidf.pkl")

# -----------------------------
# Page Config
# -----------------------------
st.set_page_config(
    page_title="AI Movie Genre Classifier",
    page_icon="🎬",
    layout="wide"
)

# -----------------------------
# Custom CSS
# -----------------------------
st.markdown("""
<style>

.stApp{
background:linear-gradient(135deg,#0f172a,#1e293b,#111827);
color:white;
}

.main-title{
text-align:center;
font-size:52px;
font-weight:bold;
color:#38bdf8;
}

.sub{
text-align:center;
font-size:20px;
color:#cbd5e1;
margin-bottom:30px;
}

.card{
background:rgba(255,255,255,0.08);
padding:25px;
border-radius:18px;
backdrop-filter:blur(12px);
border:1px solid rgba(255,255,255,.15);
}

.result{
background:linear-gradient(90deg,#2563eb,#06b6d4);
padding:18px;
border-radius:15px;
text-align:center;
font-size:30px;
font-weight:bold;
color:white;
margin-top:20px;
}

.genre{
font-size:26px;
font-weight:bold;
}

.stButton>button{
width:100%;
height:55px;
font-size:20px;
border-radius:12px;
background:#2563eb;
color:white;
border:none;
}

.stButton>button:hover{
background:#1d4ed8;
color:white;
}

footer{
visibility:hidden;
}

</style>
""", unsafe_allow_html=True)

# -----------------------------
# Header
# -----------------------------
st.markdown("<div class='main-title'>🎬 AI Movie Genre Classifier</div>", unsafe_allow_html=True)

st.markdown("<div class='sub'>Predict the genre of any movie using Machine Learning</div>", unsafe_allow_html=True)

col1,col2=st.columns([2,1])

with col1:

    st.markdown("<div class='card'>",unsafe_allow_html=True)

    plot=st.text_area(
        "📝 Enter Movie Plot",
        height=260,
        placeholder="Example: A detective investigates mysterious murders in a dark city..."
    )

    predict=st.button("🚀 Predict Genre")

    st.markdown("</div>",unsafe_allow_html=True)

with col2:

    st.markdown("<div class='card'>",unsafe_allow_html=True)

    st.subheader("📊 Model Details")

    st.write("✅ Algorithm : Logistic Regression")

    st.write("✅ Vectorizer : TF-IDF")

    st.write("✅ Dataset : Kaggle Movie Genre Dataset")

    st.write("🎯 Accuracy : 58.36%")

    st.markdown("</div>",unsafe_allow_html=True)

# -----------------------------
# Prediction
# -----------------------------
if predict:

    if plot.strip()=="":

        st.warning("⚠ Please enter a movie plot.")

    else:

        with st.spinner("🤖 AI is analysing the movie plot..."):

            time.sleep(2)

            vector=tfidf.transform([plot])

            genre=model.predict(vector)[0].strip()

        emoji={

            "drama":"🎭",

            "comedy":"😂",

            "action":"💥",

            "thriller":"🔪",

            "horror":"👻",

            "romance":"❤️",

            "crime":"🚔",

            "adventure":"🗺️",

            "animation":"🎨",

            "sci-fi":"🚀",

            "fantasy":"🧙",

            "family":"👨‍👩‍👧",

            "documentary":"🎥"

        }

        icon=emoji.get(genre.lower(),"🎬")

        st.markdown(f"""

<div class="result">

{icon}<br>

Predicted Genre<br><br>

<span class="genre">{genre.upper()}</span>

</div>

""",unsafe_allow_html=True)

        st.progress(95)

        st.success("Prediction completed successfully.")

st.markdown("---")

st.markdown(
"<center>Developed with ❤️ using Python • Scikit-Learn • Streamlit</center>",
unsafe_allow_html=True
)