# arthritis_diagnosis.py

import streamlit as st

# --- A Simple, Self-Contained Rule Engine ---


class SimpleKnowledgeEngine:
    def __init__(self, rules):
        self.rules = rules
        self.facts = set()
        self.diagnosis = "No diagnosis could be made based on the provided symptoms."

    def reset(self):
        self.facts = set()
        self.diagnosis = "No diagnosis could be made based on the provided symptoms."

    def declare(self, fact):
        self.facts.add(fact)

    def run(self):
        new_facts_found = True
        while new_facts_found:
            new_facts_found = False
            for rule in self.rules:
                # Check if all conditions for the rule are in our set of facts
                if rule['conditions'].issubset(self.facts):
                    # If the rule's conclusion is not already a fact, add it
                    if rule['conclusion'] not in self.facts:
                        self.facts.add(rule['conclusion'])
                        # Store the diagnosis
                        self.diagnosis = rule['conclusion']
                        new_facts_found = True


# --- Define All Your Rules Here ---
# Each rule is a dictionary with a set of 'conditions' and a 'conclusion'.
rules = [
    {
        'conditions': {'symptom_joint_pain', 'pain_worsens_with_activity_yes', 'age_over_50'},
        'conclusion': 'Osteoarthritis'
    },
    {
        'conditions': {'symptom_morning_stiffness_gt_30_mins', 'multiple_joints_affected_yes', 'symmetrical_joint_involvement_yes'},
        'conclusion': 'Rheumatoid Arthritis'
    },
    {
        'conditions': {'symptom_severe_pain_in_big_toe', 'joint_is_red_and_hot_yes', 'history_of_high_uric_acid_yes'},
        'conclusion': 'Gout'
    },
    # --- ADD YOUR 100+ RULES HERE IN THE SAME FORMAT ---
]

# --- Streamlit User Interface ---


def main():
    st.title("Arthritis Diagnosis Expert System")
    st.write("Please answer the following questions to get a possible diagnosis.")
    st.write("Disclaimer: This is for educational purposes and not a substitute for professional medical advice.")

    # Instantiate our simple engine with the rules
    engine = SimpleKnowledgeEngine(rules)

    st.header("Patient Information and Symptoms")

    age = st.radio("What is your age range?", ('Under 50', 'Over 50'))
    joint_pain = st.checkbox("Do you have joint pain?")
    pain_worsens_with_activity = st.radio(
        "Does the pain worsen with activity?",
        ('Yes', 'No'),
        disabled=(not joint_pain)
    )
    morning_stiffness = st.checkbox(
        "Do you experience morning stiffness that lasts for more than 30 minutes?")
    multiple_joints_affected = st.checkbox("Are multiple joints affected?")
    symmetrical_joint_involvement = st.checkbox(
        "Are the joints on both sides of your body affected symmetrically (e.g., both wrists)?",
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
        engine.reset()

        # --- Declare facts based on user input ---
        if age == 'Over 50':
            engine.declare('age_over_50')
        if joint_pain:
            engine.declare('symptom_joint_pain')
        if pain_worsens_with_activity == 'Yes':
            engine.declare('pain_worsens_with_activity_yes')
        if morning_stiffness:
            engine.declare('symptom_morning_stiffness_gt_30_mins')
        if multiple_joints_affected:
            engine.declare('multiple_joints_affected_yes')
        if symmetrical_joint_involvement:
            engine.declare('symmetrical_joint_involvement_yes')
        if severe_pain_in_big_toe:
            engine.declare('symptom_severe_pain_in_big_toe')
        if joint_is_red_and_hot:
            engine.declare('joint_is_red_and_hot_yes')
        if history_of_high_uric_acid:
            engine.declare('history_of_high_uric_acid_yes')

        # --- Run the engine and display the result ---
        engine.run()

        st.subheader("Diagnosis Result")
        st.success(f"The probable diagnosis is: {engine.diagnosis}")


if __name__ == "__main__":
    main()
