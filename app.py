# app.py
"""
This is the main application file for the Hybrid Arthritis Diagnosis System.
It uses Streamlit to create the user interface and orchestrates calls to the
expert system and the machine learning model.
"""

import streamlit as st
import pandas as pd
import numpy as np

# Import our custom modules from the 'src' package
from src.expert_system import SimpleKnowledgeEngine, rules
from src.ml_model import get_trained_model_and_metrics

# --- 1. Load and Prepare Models (This runs only once) ---
st.set_page_config(
    layout="wide", page_title="Arthritis Diagnosis Expert System")

# Get the trained ML model and its performance metrics
ml_components = get_trained_model_and_metrics()
ml_model = ml_components["model"]
model_accuracy = ml_components["accuracy"]
best_params = ml_components["best_params"]
model_features = ml_components["features"]

# Instantiate the rule-based expert system
expert_engine = SimpleKnowledgeEngine(rules)

# --- 2. Setup the User Interface ---
st.sidebar.title("System Configuration")
st.sidebar.subheader("Optimized Model Hyperparameters")
st.sidebar.write(f"Trees (n_estimators): {int(best_params[0])}")
st.sidebar.write(f"Max Depth (max_depth): {int(best_params[1])}")
st.sidebar.subheader("ML Model Performance")
st.sidebar.write(f"Accuracy on Test Data: {model_accuracy:.2%}")

st.title("Hybrid Arthritis Diagnosis System")
st.write("This system combines a rule-based expert system with a machine learning model to provide a comprehensive diagnosis.")
st.write("Disclaimer: Not a substitute for professional medical advice.")

st.header("Patient Information and Symptoms")

# Collect user inputs
age = st.radio("What is your age range?", ('Under 50', 'Over 50'))
joint_pain = st.checkbox("Do you have joint pain?")
pain_worsens_with_activity = st.radio(
    "Does the pain worsen with activity?", ('Yes', 'No'), disabled=(not joint_pain))
morning_stiffness = st.checkbox(
    "Do you experience morning stiffness that lasts for more than 30 minutes?")
multiple_joints_affected = st.checkbox("Are multiple joints affected?")
symmetrical_joint_involvement = st.checkbox(
    "Are the joints on both sides of your body affected symmetrically?",
    disabled=(not multiple_joints_affected)
)
severe_pain_in_big_toe = st.checkbox(
    "Do you have sudden and severe pain in your big toe?")
joint_is_red_and_hot = st.checkbox(
    "Is the affected joint red, hot, and swollen?",
    disabled=(not severe_pain_in_big_toe)
)
history_of_high_uric_acid = st.checkbox(
    "Do you have a history of high uric acid levels?")

if st.button("Get Diagnosis"):
    # --- 3. Process Inputs and Run Models ---

    # Run the Rule-Based Engine
    expert_engine.reset()
    if age == 'Over 50':
        expert_engine.declare('age_over_50')
    if joint_pain:
        expert_engine.declare('symptom_joint_pain')
    if pain_worsens_with_activity == 'Yes':
        expert_engine.declare('pain_worsens_with_activity_yes')
    if morning_stiffness:
        expert_engine.declare('symptom_morning_stiffness_gt_30_mins')
    if multiple_joints_affected:
        expert_engine.declare('multiple_joints_affected_yes')
    if symmetrical_joint_involvement:
        expert_engine.declare('symmetrical_joint_involvement_yes')
    if severe_pain_in_big_toe:
        expert_engine.declare('symptom_severe_pain_in_big_toe')
    if joint_is_red_and_hot:
        expert_engine.declare('joint_is_red_and_hot_yes')
    if history_of_high_uric_acid:
        expert_engine.declare('history_of_high_uric_acid_yes')
    expert_engine.run()

    # Prepare input for the Machine Learning Model
    user_input_dict = {
        'age_over_50': 1 if age == 'Over 50' else 0,
        'symptom_joint_pain': 1 if joint_pain else 0,
        'pain_worsens_with_activity_yes': 1 if pain_worsens_with_activity == 'Yes' else 0,
        'symptom_morning_stiffness_gt_30_mins': 1 if morning_stiffness else 0,
        'multiple_joints_affected_yes': 1 if multiple_joints_affected else 0,
        'symmetrical_joint_involvement_yes': 1 if symmetrical_joint_involvement else 0,
        'symptom_severe_pain_in_big_toe': 1 if severe_pain_in_big_toe else 0,
        'joint_is_red_and_hot_yes': 1 if joint_is_red_and_hot else 0,
        'history_of_high_uric_acid_yes': 1 if history_of_high_uric_acid else 0
    }
    user_input_df = pd.DataFrame([user_input_dict], columns=model_features)

    # Run the Machine Learning Model
    ml_diagnosis = ml_model.predict(user_input_df)[0]
    ml_probabilities = ml_model.predict_proba(user_input_df)

    # --- 4. Display Results (ENHANCED) ---
    st.subheader("Diagnosis Results")
    col1, col2 = st.columns(2)

    with col1:
        st.info("Rule-Based Expert System Diagnosis")
        st.write(f"**Diagnosis:** {expert_engine.diagnosis}")
        st.caption("Based on a strict set of predefined expert rules.")

    with col2:
        st.info("Machine Learning Model Diagnosis")
        st.write(f"**Top Diagnosis:** {ml_diagnosis}")

        # Create a DataFrame for the probability chart
        prob_df = pd.DataFrame(
            ml_probabilities[0],
            index=ml_model.classes_,
            columns=['Confidence']
        )
        prob_df['Confidence'] = prob_df['Confidence'].apply(
            lambda x: f"{x:.0%}")

        st.write("**Probability Distribution:**")
        st.dataframe(prob_df, use_container_width=True)
        st.caption("Based on patterns learned from 750 patient records.")

    st.subheader("Final Recommendation")
    if expert_engine.diagnosis == ml_diagnosis and "No diagnosis" not in expert_engine.diagnosis:
        st.success(
            f"**Conclusion:** Both systems agree on a diagnosis of **{ml_diagnosis}**. There is a high likelihood this is correct. Please consult a doctor for confirmation.")
    elif "No diagnosis" in expert_engine.diagnosis and "Healthy" not in ml_diagnosis:
        st.warning(
            f"**Conclusion:** The rule-based system could not find a match, but the ML model suggests **{ml_diagnosis}**. A medical consultation is strongly advised.")
    else:
        st.error(
            f"**Conclusion:** The systems disagree (Expert: **{expert_engine.diagnosis}** vs. ML: **{ml_diagnosis}**). This indicates a complex or unusual case. It is essential to see a doctor for a proper evaluation.")
