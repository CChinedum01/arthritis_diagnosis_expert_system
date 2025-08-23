import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from geneticalgorithm import geneticalgorithm as ga


@st.cache_data
def generate_mock_data(num_records=750):
    """Generates a mock dataset for training."""
    np.random.seed(42)
    symptoms = [
        'age_over_50', 'symptom_joint_pain', 'pain_worsens_with_activity_yes',
        'symptom_morning_stiffness_gt_30_mins', 'multiple_joints_affected_yes',
        'symmetrical_joint_involvement_yes', 'symptom_severe_pain_in_big_toe',
        'joint_is_red_and_hot_yes', 'history_of_high_uric_acid_yes'
    ]
    diagnoses = ['Osteoarthritis', 'Rheumatoid Arthritis', 'Gout', 'Healthy']
    data = []
    archetypes = {
        'Osteoarthritis':         [0.9, 0.9, 0.9, 0.1, 0.2, 0.1, 0.05, 0.05, 0.1],
        'Rheumatoid Arthritis':   [0.4, 0.9, 0.2, 0.9, 0.9, 0.9, 0.05, 0.05, 0.1],
        'Gout':                   [0.4, 0.9, 0.3, 0.1, 0.1, 0.1, 0.9, 0.9, 0.9],
        'Healthy':                [0.2, 0.1, 0.1, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05]
    }
    for _ in range(num_records):
        diagnosis = np.random.choice(diagnoses)
        profile = archetypes[diagnosis]
        record = {symptom: 1 if np.random.rand(
        ) < profile[i] else 0 for i, symptom in enumerate(symptoms)}
        record['diagnosis'] = diagnosis
        data.append(record)
    return pd.DataFrame(data)


def get_trained_model_and_metrics():
    """
    This function orchestrates the data generation, GA optimization,
    and final model training. It's the public interface of this module.
    """
    df = generate_mock_data()
    X = df.drop('diagnosis', axis=1)
    y = df['diagnosis']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)

    def fitness_function(X_ga):
        """The function to be optimized by the GA."""
        n_estimators = int(X_ga[0])
        max_depth = int(X_ga[1])
        if n_estimators < 10 or max_depth < 2:
            return -0.0
        model = RandomForestClassifier(
            n_estimators=n_estimators, max_depth=max_depth, random_state=42)
        score = cross_val_score(model, X_train, y_train, cv=3)
        return -np.mean(score)

    # The corrected run_ga_optimization function in src/ml_model.py

    @st.cache_resource
    def run_ga_optimization():
        """Runs the GA and caches the result."""
        varbound = np.array([[10, 200], [2, 30]])

        algorithm_params = {
            'max_num_iteration': 50,
            'population_size': 20,
            'mutation_probability': 0.1,
            'elit_ratio': 0.1,
            'crossover_probability': 0.8,
            'parents_portion': 0.3,
            'crossover_type': 'uniform',
            'max_iteration_without_improv': None
        }

        ga_model = ga(function=fitness_function, dimension=2, variable_type='int',
                      variable_boundaries=varbound, algorithm_parameters=algorithm_params)

        ga_model.run()

        # The results are stored in the 'output_dict' attribute under the key 'variable'
        return ga_model.output_dict['variable']

    best_params = run_ga_optimization()

    # Train the final model with the best parameters
    ml_model = RandomForestClassifier(
        n_estimators=int(best_params[0]),
        max_depth=int(best_params[1]),
        random_state=42
    )
    ml_model.fit(X_train, y_train)

    # Evaluate and return results
    y_pred = ml_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    return {
        "model": ml_model,
        "accuracy": accuracy,
        "best_params": best_params,
        "features": list(X.columns)
    }
