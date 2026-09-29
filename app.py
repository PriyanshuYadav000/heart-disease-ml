import streamlit as st
import pandas as pd
import joblib

st.set_page_config(
    page_title="Heart Risk Checker",
    page_icon="❤️",
    layout="centered"
)


@st.cache_resource
def load_model():
    return joblib.load("heart_model.pkl")


try:
    model = load_model()
except FileNotFoundError:
    st.error(
        "heart_model.pkl was not found. "
        "Make sure it is in the same folder as app.py."
    )
    st.stop()


st.markdown(
    """
    <style>
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }

    div.stButton > button {
        width: 100%;
        background-color: #e63946;
        color: white;
        font-size: 1.1rem;
        font-weight: 600;
        padding: 0.6rem 0;
        border-radius: 10px;
        border: none;
    }

    div.stButton > button:hover {
        background-color: #c1121f;
        color: white;
    }
    </style>
    """,
    unsafe_allow_html=True
)


st.title("❤️ Heart Disease Risk Checker")
st.caption("Built with a Naive Bayes model")

st.info(
    "Fill in your details below and click **Predict**. "
    "Hover over the ❓ icons for help."
)


with st.sidebar:
    st.header("About")

    st.write(
        "This tool uses a machine learning model to estimate "
        "the predicted heart disease class from the supplied "
        "clinical information."
    )

    st.divider()

    st.write("**Model:** Gaussian Naive Bayes")
    st.write("**Test Accuracy:** 90.76%")
    st.write("**Test F1 Score:** 91.54%")

    st.warning(
        "⚠️ Educational use only. This is not a medical diagnosis. "
        "Please consult a qualified healthcare professional for "
        "medical advice."
    )


with st.form("heart_form"):

    st.subheader("👤 Personal Details")

    c1, c2 = st.columns(2)

    with c1:
        age = st.slider(
            "Age",
            18,
            100,
            40,
            help="Patient age in years."
        )

    with c2:
        sex = st.radio(
            "Sex",
            ["M", "F"],
            format_func=lambda x: (
                "Male" if x == "M" else "Female"
            ),
            horizontal=True,
            help="Patient sex."
        )

    st.subheader("🩺 Clinical Measurements")

    c1, c2 = st.columns(2)

    with c1:
        resting_bp = st.number_input(
            "Resting Blood Pressure (mm Hg)",
            min_value=80,
            max_value=220,
            value=120,
            step=1,
            help="Resting blood pressure."
        )

        cholesterol = st.number_input(
            "Cholesterol (mg/dL)",
            min_value=0,
            max_value=700,
            value=200,
            step=1,
            help="Serum cholesterol level."
        )

        fasting_bs = st.radio(
            "Fasting Blood Sugar > 120 mg/dL?",
            [0, 1],
            format_func=lambda x: (
                "Yes" if x == 1 else "No"
            ),
            horizontal=True,
            help="Whether fasting blood sugar is above 120 mg/dL."
        )

    with c2:
        max_hr = st.slider(
            "Max Heart Rate",
            60,
            220,
            150,
            help="Maximum heart rate achieved."
        )

        oldpeak = st.slider(
            "Oldpeak (ST Depression)",
            -3.0,
            7.0,
            0.0,
            0.1,
            help="ST depression induced by exercise relative to rest."
        )

    st.subheader("📈 Heart Tests")

    c1, c2 = st.columns(2)

    with c1:
        chest_pain = st.selectbox(
            "Chest Pain Type",
            ["ATA", "NAP", "TA", "ASY"],
            format_func=lambda x: {
                "ATA": "ATA – Atypical Angina",
                "NAP": "NAP – Non-Anginal Pain",
                "TA": "TA – Typical Angina",
                "ASY": "ASY – Asymptomatic"
            }[x]
        )

        resting_ecg = st.selectbox(
            "Resting ECG",
            ["Normal", "ST", "LVH"],
            format_func=lambda x: {
                "Normal": "Normal",
                "ST": "ST-T Wave Abnormality",
                "LVH": "Left Ventricular Hypertrophy"
            }[x]
        )

    with c2:
        exercise_angina = st.radio(
            "Exercise-Induced Angina?",
            ["Y", "N"],
            format_func=lambda x: (
                "Yes" if x == "Y" else "No"
            ),
            horizontal=True,
            help="Whether exercise causes angina."
        )

        st_slope = st.selectbox(
            "ST Slope",
            ["Up", "Flat", "Down"],
            help="Slope of the peak exercise ST segment."
        )

    submitted = st.form_submit_button(
        "🔍 Predict"
    )


if submitted:

    patient = pd.DataFrame({
        "Age": [age],
        "Sex": [sex],
        "ChestPainType": [chest_pain],
        "RestingBP": [resting_bp],
        "Cholesterol": [cholesterol],
        "FastingBS": [fasting_bs],
        "RestingECG": [resting_ecg],
        "MaxHR": [max_hr],
        "ExerciseAngina": [exercise_angina],
        "Oldpeak": [oldpeak],
        "ST_Slope": [st_slope]
    })

    with st.spinner("Analyzing..."):

        prediction = model.predict(patient)[0]

        probabilities = model.predict_proba(patient)[0]

        no_disease_probability = float(probabilities[0])
        disease_probability = float(probabilities[1])

    st.divider()

    st.subheader("Result")

    if prediction == 1:
        st.error(
            "⚠️ **High Risk of Heart Disease**"
        )

        st.write(
            "The model predicts the heart disease class "
            "for the provided input values."
        )

    else:
        st.success(
            "✅ **Low Risk of Heart Disease**"
        )

        st.write(
            "The model predicts the non-heart-disease class "
            "for the provided input values."
        )

    st.metric(
        "Estimated model probability",
        f"{disease_probability * 100:.1f}%"
    )

    st.progress(
        min(
            max(
                disease_probability,
                0.0
            ),
            1.0
        )
    )

    c1, c2 = st.columns(2)

    with c1:
        st.metric(
            "No Heart Disease",
            f"{no_disease_probability * 100:.1f}%"
        )

    with c2:
        st.metric(
            "Heart Disease",
            f"{disease_probability * 100:.1f}%"
        )

    with st.expander("📋 View your entered details"):

        details = pd.DataFrame({
            "Feature": [
                "Age",
                "Sex",
                "Chest Pain",
                "Resting BP",
                "Cholesterol",
                "Fasting Blood Sugar",
                "Resting ECG",
                "Max Heart Rate",
                "Exercise Angina",
                "Oldpeak",
                "ST Slope"
            ],
            "Value": [
                age,
                "Male" if sex == "M" else "Female",
                chest_pain,
                resting_bp,
                cholesterol,
                "Yes" if fasting_bs == 1 else "No",
                resting_ecg,
                max_hr,
                "Yes" if exercise_angina == "Y" else "No",
                oldpeak,
                st_slope
            ]
        })

        st.dataframe(
            details,
            hide_index=True,
            use_container_width=True
        )

    st.caption(
        "This prediction is for educational purposes and is not "
        "a substitute for professional medical advice."
    )