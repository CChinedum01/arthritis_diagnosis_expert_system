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

# --- CORRECTED: Renamed the function for clarity. This is our callback. ---


def clear_selections():
    st.session_state.age = 'Under 50'
    st.session_state.joint_pain = False
    st.session_state.pain_worsens = 'No'
    st.session_state.morning_stiffness = False
    st.session_state.multiple_joints = False
    st.session_state.symmetrical = False
    st.session_state.severe_pain_toe = False
    st.session_state.joint_red_hot = False
    st.session_state.high_uric_acid = False


# Initialize the state only on the very first run
if 'age' not in st.session_state:
    clear_selections()

# --- 2. Setup the User Interface ---
st.sidebar.title("System Configuration")
st.sidebar.subheader("Optimized Model Hyperparameters")
st.sidebar.write(f"Trees (n_estimators): {best_params['n_estimators']}")
st.sidebar.write(f"Max Depth (max_depth): {best_params['max_depth']}")
st.sidebar.subheader("ML Model Performance")
st.sidebar.write(f"Accuracy on Test Data: {model_accuracy:.2%}")

st.title("Arthritis Diagnosis System")
st.write("This system combines a rule-based expert system with a machine learning model to provide a comprehensive diagnosis.")
st.write("Disclaimer: Not a substitute for professional medical advice.")

st.header("Patient Information and Symptoms")

# Connect each widget to session_state using the 'key' parameter
st.radio("What is your age range?", ('Under 50', 'Over 50'), key='age')
st.checkbox("Do you have joint pain?", key='joint_pain')
st.radio("Does the pain worsen with activity?", ('Yes', 'No'),
         key='pain_worsens', disabled=(not st.session_state.joint_pain))
st.checkbox("Do you experience morning stiffness that lasts for more than 30 minutes?",
            key='morning_stiffness')
st.checkbox("Are multiple joints affected?", key='multiple_joints')
st.checkbox(
    "Are the joints on both sides of your body affected symmetrically?",
    key='symmetrical',
    disabled=(not st.session_state.multiple_joints)
)
st.checkbox("Do you have sudden and severe pain in your big toe?",
            key='severe_pain_toe')
st.checkbox(
    "Is the affected joint red, hot, and swollen?",
    key='joint_red_hot',
    disabled=(not st.session_state.severe_pain_toe)
)
st.checkbox("Do you have a history of high uric acid levels?",
            key='high_uric_acid')

# Add columns for the buttons
# Adjusted column widths for better spacing
col1, col2, col3 = st.columns([1.5, 2, 5])

get_diagnosis_button = col1.button("Get Diagnosis")

# --- CORRECTED: Use the on_click callback for the clear button ---
col2.button("Clear Selections", on_click=clear_selections)

# --- REMOVED the old `if clear_button:` block ---

if get_diagnosis_button:
    # --- 3. Process Inputs and Run Models ---

    # Run the Rule-Based Engine
    expert_engine.reset()
    if st.session_state.age == 'Over 50':
        expert_engine.declare('age_over_50')
    if st.session_state.joint_pain:
        expert_engine.declare('symptom_joint_pain')
    if st.session_state.pain_worsens == 'Yes':
        expert_engine.declare('pain_worsens_with_activity_yes')
    if st.session_state.morning_stiffness:
        expert_engine.declare('symptom_morning_stiffness_gt_30_mins')
    if st.session_state.multiple_joints:
        expert_engine.declare('multiple_joints_affected_yes')
    if st.session_state.symmetrical:
        expert_engine.declare('symmetrical_joint_involvement_yes')
    if st.session_state.severe_pain_toe:
        expert_engine.declare('symptom_severe_pain_in_big_toe')
    if st.session_state.joint_red_hot:
        expert_engine.declare('joint_is_red_hot_yes')
    if st.session_state.high_uric_acid:
        expert_engine.declare('history_of_high_uric_acid_yes')
    expert_engine.run()

    # Prepare input for the Machine Learning Model
    user_input_dict = {
        'age_over_50': 1 if st.session_state.age == 'Over 50' else 0,
        'symptom_joint_pain': 1 if st.session_state.joint_pain else 0,
        'pain_worsens_with_activity_yes': 1 if st.session_state.pain_worsens == 'Yes' else 0,
        'symptom_morning_stiffness_gt_30_mins': 1 if st.session_state.morning_stiffness else 0,
        'multiple_joints_affected_yes': 1 if st.session_state.multiple_joints else 0,
        'symmetrical_joint_involvement_yes': 1 if st.session_state.symmetrical else 0,
        'symptom_severe_pain_in_big_toe': 1 if st.session_state.severe_pain_toe else 0,
        'joint_is_red_and_hot_yes': 1 if st.session_state.joint_red_hot else 0,
        'history_of_high_uric_acid_yes': 1 if st.session_state.high_uric_acid else 0
    }
    user_input_df = pd.DataFrame([user_input_dict], columns=model_features)

    # Run the Machine Learning Model
    ml_diagnosis = ml_model.predict(user_input_df)[0]
    ml_probabilities = ml_model.predict_proba(user_input_df)

    # --- 4. Display Results ---
    st.subheader("Diagnosis Results")
    col1, col2 = st.columns(2)

    with col1:
        st.info("Rule-Based Expert System Diagnosis")
        st.write(f"**Diagnosis:** {expert_engine.diagnosis}")
        st.caption("Based on a strict set of predefined expert rules.")

    with col2:
        st.info("Machine Learning Model Diagnosis")
        st.write(f"**Top Diagnosis:** {ml_diagnosis}")

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
            f"**Conclusion:** The systems disagree (Expert: **{expert_engine.diagnosis}** vs. ML: **{ml_diagnosis}**). This indicates a complex case. It is essential to see a doctor for a proper evaluation.")
