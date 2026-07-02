import streamlit as st
import numpy as np
import pandas as pd
import pickle

st.markdown("""
<style>
.stApp {
    background-image: url("https://images.pexels.com/photos/6823567/pexels-photo-6823567.jpeg");
    background-size: cover;
    background-position: center;
    background-repeat: no-repeat;
    background-attachment: fixed;
}
</style>
""", unsafe_allow_html=True)

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Diabetes Prediction System",
    page_icon="🩺",
    layout="wide"
)

# ---------------- LOAD MODEL ----------------
model = pickle.load(open("model.pkl", "rb"))
sc = pickle.load(open("sc.pkl", "rb"))

# ---------------- TITLE ----------------
st.title("🩺 Diabetes Prediction System")
st.markdown("### Predict whether a patient is diabetic using Machine Learning.")

# ---------------- SIDEBAR ----------------
st.sidebar.title("About Project")

st.sidebar.info("""
This application predicts whether a patient is likely to have diabetes based on medical information.

**Machine Learning Model**
- Input: 8 Health Parameters
- Output: Diabetic / Non-Diabetic
""")

st.sidebar.success("Developed using Streamlit & Python")

# ---------------- INPUT ----------------
st.subheader("📝 Patient Information")

with st.container(border=True):
    col1, col2 = st.columns(2)

    with col1:
        Pregnancies = st.slider("🤰 Pregnancies", 0, 17, 1)
        BloodPressure = st.slider("🩸 Blood Pressure", 40, 140, 72)
        Insulin = st.slider("💉 Insulin", 15, 300, 80)
        DiabetesPedigreeFunction = st.number_input(
            "🧬 Diabetes Pedigree Function",
            min_value=0.05,
            max_value=3.0,
            value=0.47,
            step=0.001,
            format="%.3f"
        )

    with col2:
        Glucose = st.slider("🍬 Glucose", 50, 200, 120)
        SkinThickness = st.slider("📏 Skin Thickness", 7, 99, 20)
        BMI = st.slider("⚖️ BMI", 18.0, 50.0, 32.0, step=0.1)
        Age = st.slider("🎂 Age", 21, 81, 33)
st.markdown("---")

st.markdown("""
<style>
label {
    background-color: #56117d !important;
    color: white !important;
    padding: 6px 12px !important;
    border-radius: 10px !important;
    font-weight: bold !important;
}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<style>

/* Slider track (gray line) */
.stSlider > div > div > div > div {
    background: #B0BEC5 !important;
}

/* Filled part of the slider */
.stSlider [data-baseweb="slider"] div:nth-child(2) {
    background: #2196F3 !important;
}

/* Slider handle (circle) */
.stSlider [role="slider"] {
    background-color: #1565C0 !important;
    border: 3px solid white !important;
}

/* Slider value text */
.stSlider label {
    color: white !important;
    font-weight: bold;
}

</style>
""", unsafe_allow_html=True)

# ---------------- PREDICT ----------------
if st.button("🔍 Predict", use_container_width=True):

    columns = [
        'Pregnancies',
        'Glucose',
        'BloodPressure',
        'SkinThickness',
        'Insulin',
        'BMI',
        'DiabetesPedigreeFunction',
        'Age'
    ]

    myinput = [[
        Pregnancies,
        Glucose,
        BloodPressure,
        SkinThickness,
        Insulin,
        BMI,
        DiabetesPedigreeFunction,
        Age
    ]]

    myinput = pd.DataFrame(myinput, columns=columns)

    scaled_data = sc.transform(myinput)

    result = model.predict(scaled_data)

    # Probability (works only if your model supports it)
    if hasattr(model, "predict_proba"):
        probability = model.predict_proba(scaled_data)[0][1]
    else:
        probability = None

    st.subheader("Prediction Result")

    if result[0] == 0:
        st.success("✅ Patient is NOT Diabetic")

        if probability is not None:
            st.metric("Risk Probability", f"{probability*100:.2f}%")

        st.info("""
### Healthy Lifestyle Tips
✔ Maintain a balanced diet

✔ Exercise regularly

✔ Drink plenty of water

✔ Sleep 7–8 hours daily

✔ Get regular health checkups
""")

    else:
        st.error("⚠ Patient is Diabetic")

        if probability is not None:
            st.metric("Risk Probability", f"{probability*100:.2f}%")

        st.warning("""
### Recommended Actions

• Consult a doctor.

• Monitor blood sugar regularly.

• Follow a healthy diet.

• Exercise daily.

• Avoid sugary foods and drinks.
""")

# ---------------- FOOTER ----------------
st.markdown("---")
st.caption("Developed using ❤️ Python, Streamlit & Machine Learning")

if result[0] == 0:
    st.markdown("""
    <div style="
        background:#E8F5E9;
        color:#1B5E20;
        padding:20px;
        border-radius:12px;
        border:2px solid #4CAF50;
        text-align:center;
        font-size:24px;
        font-weight:bold;">
        ✅ Patient is NOT Diabetic
    </div>
    """, unsafe_allow_html=True)

else:
    st.markdown("""
    <div style="
        background:#FFEBEE;
        color:#B71C1C;
        padding:20px;
        border-radius:12px;
        border:2px solid #F44336;
        text-align:center;
        font-size:24px;
        font-weight:bold;">
        ⚠️ Patient is Diabetic
    </div>
    """, unsafe_allow_html=True)

