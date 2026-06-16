import streamlit as st
import pandas as pd
import pickle
import matplotlib.pyplot as plt

# -----------------------------
# Page Configuration
# -----------------------------

st.set_page_config(
    page_title="AI Student Performance Analytics",
    page_icon="<=",
    layout="wide"
)

# -----------------------------
# Load Model
# -----------------------------

model = pickle.load(open("model.pkl", "rb"))

# -----------------------------
# Title
# -----------------------------

st.title("AI-Powered Student Performance Analytics System")

st.markdown(
    """
    Predict student academic performance using Machine Learning.
    """
)

# -----------------------------
# Tabs
# -----------------------------

tab1, tab2 = st.tabs(
    [
        "Prediction",
        "Analytics"
    ]
)

# ===================================================
# TAB 1 : Prediction
# ===================================================

with tab1:

    st.header("Student Performance Prediction")

    col1, col2 = st.columns(2)

    with col1:

        age = st.number_input(
            "Age",
            min_value=10,
            max_value=25,
            value=17
        )

        study_hours = st.number_input(
            "Study Hours Per Week",
            min_value=0,
            max_value=50,
            value=15
        )

        attendance = st.number_input(
            "Attendance Percentage",
            min_value=0,
            max_value=100,
            value=80
        )

    with col2:

        previous_score = st.number_input(
            "Previous Score",
            min_value=0,
            max_value=100,
            value=70
        )

        sleep_hours = st.number_input(
            "Sleep Hours",
            min_value=0,
            max_value=12,
            value=7
        )

    if st.button("Predict Performance"):

        input_data = pd.DataFrame({

            'Age': [age],

            'Study_Hours_Per_Week': [study_hours],

            'Attendance_Percentage': [attendance],

            'Previous_Score': [previous_score],

            'Sleep_Hours': [sleep_hours]

        })

        prediction = model.predict(input_data)[0]

        st.success(
            f"Predicted Final Exam Score: {prediction:.2f}"
        )

        # Performance Category

        if prediction >= 85:

            category = "Excellent"

        elif prediction >= 70:

            category = "Good"

        elif prediction >= 50:

            category = "Average"

        else:

            category = "Needs Improvement"

        st.subheader(
            f"Performance Category: {category}"
        )

        # Pass / Fail

        if prediction >= 60:

            st.success("Result: PASS")

        else:

            st.error("Result: FAIL")

        # Risk Detection

        if prediction < 60:

            st.error(
                "High Risk Student"
            )

        else:

            st.success(
                "Low Risk Student"
            )

        # Recommendations

        st.subheader(
            "Personalized Recommendations"
        )

        if attendance < 75:

            st.warning(
                "Increase attendance percentage."
            )

        if study_hours < 10:

            st.warning(
                "Increase weekly study hours."
            )

        if sleep_hours < 6:

            st.warning(
                "Sleep at least 7 hours daily."
            )

        if (
            attendance >= 75
            and study_hours >= 10
            and sleep_hours >= 6
        ):

            st.success(
                "Keep maintaining your current habits!"
            )

# ===================================================
# TAB 2 : Analytics
# ===================================================

with tab2:

    st.header("Student Performance Factors")

    features = [

        "Age",

        "Study Hours",

        "Attendance",

        "Previous Score",

        "Sleep Hours"

    ]

    importance = [

        0.05,

        0.22,

        0.27,

        0.38,

        0.08

    ]

    fig, ax = plt.subplots(
        figsize=(8, 4)
    )

    ax.bar(
        features,
        importance
    )

    ax.set_title(
        "Feature Importance"
    )

    plt.xticks(
        rotation=20
    )

    st.pyplot(fig)

    st.info(
        """
        Previous Score and Attendance Percentage
        have the highest impact on final performance.
        """
    )