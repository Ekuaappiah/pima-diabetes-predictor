#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import joblib
from scipy.stats import mstats
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from pathlib import Path

# Configuration
FILE_PATH = 'data/diabetes.csv'
MODELS_DIR = Path('models')
MODEL_PATH = MODELS_DIR / 'pima_diabetes_predictor.pkl'


class DiabetesPredictionModel:
    def __init__(self, data_path, smote_strategy="auto", random_state=42):
        self.data_path = data_path
        self.smote_strategy = smote_strategy
        self.random_state = random_state
        self.model = None
        self.df = None
        self.X_train_resample = None
        self.X_test_resample = None
        self.y_train_resample = None
        self.y_test_resample = None

    def load_data(self):
        try:
            self.df = pd.read_csv(self.data_path)
        except FileNotFoundError:
            print(f"File at {self.data_path} not found.")

    def preprocess_data(self):
        # Winsorize specified columns to handle outliers
        features_to_winsorize = ['BloodPressure', 'Insulin', 'DiabetesPedigreeFunction']
        for feature in features_to_winsorize:
            self.df[feature] = mstats.winsorize(self.df[feature], limits=[0.05, 0.05])

        # Feature Engineering
        self.df['HighGlucose'] = (self.df['Glucose'] > 140).astype(int)
        self.df['Age_BMI'] = self.df['Age'] * self.df['BMI']

        # Define features (X) and target (y)
        self.X = self.df.drop(columns=['Outcome'])
        self.y = self.df['Outcome']

    def resample_data(self):
        smote = SMOTE(sampling_strategy=self.smote_strategy, random_state=self.random_state)
        self.X_resampled, self.y_resampled = smote.fit_resample(self.X, self.y)

        # Train-test split on resampled data
        self.X_train_resample, self.X_test_resample, self.y_train_resample, self.y_test_resample = train_test_split(
            self.X_resampled, self.y_resampled, test_size=0.2, random_state=self.random_state
        )

        # Train-test split on original data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=0.2, random_state=self.random_state
        )

    def train_model(self, params=None):
        if params is None:
            params = {
                'max_depth': 10,
                'learning_rate': 0.06,
                'n_estimators': 397,
                'gamma': 0.8,
                'min_child_weight': 1,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'reg_alpha': 0.1,
                'reg_lambda': 1.0,
                'scale_pos_weight': 1,
                'use_label_encoder': False,
                'eval_metric': 'logloss',
                'random_state': self.random_state
            }

        self.model = XGBClassifier(**params)
        self.model.fit(self.X_train_resample, self.y_train_resample)

    def evaluate_model(self):
        print("\n=== Evaluation on Original Test Set ===")
        y_pred_original = self.model.predict(self.X_test)
        print("Accuracy:", accuracy_score(self.y_test, y_pred_original))
        print("Classification Report:")
        print(classification_report(self.y_test, y_pred_original))
        print("Confusion Matrix:")
        print(confusion_matrix(self.y_test, y_pred_original))

    def save_model(self, model_path=MODEL_PATH):
        if self.model:
            # Create the models directory if it doesn't exist
            MODELS_DIR.mkdir(parents=True, exist_ok=True)
            joblib.dump(self.model, model_path)
            print(f"Model saved at {model_path}")
        else:
            print("Model has not been trained yet.")


# Usage
if __name__ == "__main__":
    model = DiabetesPredictionModel(data_path=FILE_PATH)

    model.load_data()
    model.preprocess_data()
    model.resample_data()
    model.train_model()
    # model.evaluate_model()
    model.save_model(MODEL_PATH)
