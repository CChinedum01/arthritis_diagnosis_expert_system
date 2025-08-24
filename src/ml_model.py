import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import optuna

# Disable Optuna's default logging messages to keep the terminal clean
optuna.logging.set_verbosity(optuna.logging.WARNING)


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
    This function orchestrates data generation, Bayesian optimization using Optuna,
    and final model training. It is the public interface of this module.
    """
    df = generate_mock_data()
    X = df.drop('diagnosis', axis=1)
    y = df['diagnosis']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)

    def objective(trial):
        """
        The objective function for Optuna to optimize.
        A 'trial' is a single run with a specific set of hyperparameters.
        """
        # Optuna intelligently suggests the next hyperparameters to try.
        n_estimators = trial.suggest_int('n_estimators', 10, 200)
        max_depth = trial.suggest_int('max_depth', 2, 30)

        model = RandomForestClassifier(
            n_estimators=n_estimators, max_depth=max_depth, random_state=42)

        # We return the mean cross-validation score, which Optuna will try to maximize.
        score = cross_val_score(model, X_train, y_train, cv=3)
        return np.mean(score)

    @st.cache_resource
    def run_optuna_optimization():
        """
        Runs the Bayesian optimization and caches the best parameters.
        This is significantly faster than the Genetic Algorithm.
        """
        # Create a "study" to track the optimization process.
        # We specify we want to 'maximize' the objective function's return value.
        study = optuna.create_study(direction='maximize')

        # Start the optimization. n_trials is the number of different hyperparameter
        # combinations Optuna will test. 50 is a good starting point.
        study.optimize(objective, n_trials=50)

        # After the study is complete, the best parameters are stored here.
        return study.best_params

    # Run the optimization to get the best hyperparameters.
    # The result will be a dictionary, e.g., {'n_estimators': 147, 'max_depth': 24}
    best_params = run_optuna_optimization()

    # Train the final model using the best parameters found by Optuna.
    ml_model = RandomForestClassifier(
        n_estimators=best_params['n_estimators'],
        max_depth=best_params['max_depth'],
        random_state=42
    )
    ml_model.fit(X_train, y_train)

    # Evaluate the final model's performance on the unseen test data.
    y_pred = ml_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    # Return everything the main app needs.
    return {
        "model": ml_model,
        "accuracy": accuracy,
        "best_params": best_params,
        "features": list(X.columns)
    }
