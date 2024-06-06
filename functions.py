import psycopg2
import pandas as pd
import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix

# Import different classifiers
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
import shap

def redshift_connection():
    try:
        # Establish a connection to the Redshift database
        conn = psycopg2.connect(
            host=os.getenv('REDSHIFT_HOST'),
            port=os.getenv('REDSHIFT_PORT'),
            dbname=os.getenv('REDSHIFT_DB'),
            user=os.getenv('REDSHIFT_USER'),
            password=os.getenv('REDSHIFT_PASSWORD')
        )

        # Create a cursor to execute SQL queries
        print("Connection to Redshift database successful")
        cursor = conn.cursor()

        # SQL query for sequence analysis
        # SQL query for sequence analysis with player_dob included without duplicates
        sql_script = """
        SELECT 
            fixture_mid,
            home_team,
            away_team,
            home_score,
            away_score,
            team_name,
            SUM(kick_errors) as total_kick_errors,
            sum(rucks_won) as total_rucks_won,
            sum(rucks_lost) as total_rucks_lost,
            SUM(linebreaks) AS total_linebreaks,
            SUM(tries) AS total_tries,
            SUM(supported_break) AS total_supported_breaks,
            SUM(defenders_beaten) AS total_defenders_beaten,
            SUM(jackals_success) AS total_jackals_success,
            SUM(intercepts) AS total_intercepts,
            SUM(tackles_made) AS total_tackles_made,
            SUM(tackles_missed) AS total_tackles_missed,
            SUM(offloads) AS total_offloads,
            SUM(carries) AS total_carries,
            SUM(carry_metres) AS total_carry_metres,
            SUM(carries_dominant) AS total_carries_dominant,
            SUM(turnovers_conceded) AS total_turnovers_conceded,
            SUM(penalties_conceded) AS total_penalties_conceded,
            SUM(carry_metres_post_contact) AS total_carry_metres_post_contact,
            sum(ruck_arrivals_attack_first2) as total_ruck_arrivals_attack_first2,
            sum(ruck_arrivals_defence_first2) as total_ruck_arrivals_defence_first2,
            CASE 
                WHEN team_name = home_team AND home_score > away_score THEN 1
                WHEN team_name = away_team AND away_score > home_score THEN 1
                ELSE 0
            END AS won_lost
        FROM 
            bal.tab_match_player_stats_v2
        WHERE 
            season = 2024
            AND competition = 'Super Rugby Pacific'
        GROUP BY 
            fixture_mid,
            home_team,
            away_team,
            home_score,
            away_score,
            team_name
        ORDER BY 
            fixture_mid,
            team_name;
        """

        cursor.execute(sql_script)
        print("Database connection is active.")

        results = cursor.fetchall()

        return results

    except psycopg2.Error as e:
        # Log and re-raise the database connection error
        print(f"Database connection error: {e}")
        raise

    finally:
        # Ensure the database connection is closed
        if conn:
            conn.close()
            print("Redshift connection closed")

def create_redshift_df(results):
    column_names_for_results_df = ['fixture_mid',
            'home_team',
            'away_team',
            'home_score',
            'away_score',
            'team_name',
            'kick_errors',
            'rucks_won',
            'rucks_lost',
            'total_linebreaks',
            'total_tries',
            'total_supported_breaks',
            'total_defenders_beaten',
            'total_jackals_success',
            'total_intercepts',
            'total_tackles_made',
            'total_tackles_missed',
            'total_offloads',
            'total_carries',
            'total_carry_metres',
            'total_carries_dominant',
            'total_turnovers_conceded',
            'total_penalties_conceded', 
            'total_carry_metres_post_contact',
            'ruck_arrivals_attack_first2',
            'ruck_arrivals_defence_first2',
            'won_lost']
    df_redshift_data = pd.DataFrame(results, columns=column_names_for_results_df)
    return df_redshift_data

def train_evaluate_model(model, X_train, y_train, X_test, y_test, feature_names=None):
    """
    Trains a model, predicts on test data, evaluates the model, and returns feature importance if applicable.
    
    :param model: The machine learning model to be trained
    :param X_train: Training data features
    :param y_train: Training data labels
    :param X_test: Testing data features
    :param y_test: Testing data labels
    :param feature_names: List of feature names for calculating feature importance
    :return: model, accuracy, confusion_matrix, feature_importance (if applicable)
    """
    # Initialize feature_importance to None to ensure it is always defined
    feature_importance = None

    # Train the model
    model.fit(X_train, y_train)

    # Predict on the test set
    y_pred = model.predict(X_test)

    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)

    # Check for feature importances (tree-based models)
    if hasattr(model, "feature_importances_") and feature_names is not None:
        feature_importance = pd.DataFrame({
            'feature': feature_names, 
            'importance': model.feature_importances_
        })
        print('Feature importance found')

    # Check for coefficients (linear models)
    elif hasattr(model, "coef_") and feature_names is not None and model.coef_.size == len(feature_names):
        feature_importance = pd.DataFrame({
            'feature': feature_names, 
            'coef': model.coef_[0]
        })
        print('Coef found')

    # Return the function's outputs
    return model, accuracy, conf_matrix, feature_importance

def calculate_mean_shap_values(model, X_train, X_test, feature_names):
    """
    Calculates the mean SHAP values for each feature and prints them.
    
    :param model: The trained machine learning model
    :param X_train: Training data features
    :param X_test: Testing data features
    :param feature_names: List of feature names
    :return: Dictionary of mean SHAP values for each feature
    """
    # Use TreeExplainer for tree-based models
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)

    # If shap_values is a list (multi-output), calculate the mean absolute SHAP values across all classes
    if isinstance(shap_values, list):
        # Average the absolute SHAP values across samples and classes
        mean_shap_values = np.mean([np.abs(values).mean(axis=0) for values in shap_values], axis=0)
    else:
        mean_shap_values = np.mean(np.abs(shap_values), axis=0)

    # Create a dictionary of mean SHAP values per feature
    mean_shap_per_feature = dict(zip(feature_names, mean_shap_values))
    
    return mean_shap_per_feature






