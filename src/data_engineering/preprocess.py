import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from .load_sql import load_data_from_sql

NUMERIC_FEATURES = ['age', 'bmi', 'liver_function_score', 'alpha_fetoprotein_level']
CATEGORICAL_FEATURES = ['gender', 'alcohol_consumption', 'smoking_status', 'physical_activity_level']
TARGET = 'liver_cancer'

def build_preprocessor():
    numeric_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(drop='first', handle_unknown='ignore')
    return ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, NUMERIC_FEATURES),
            ('cat', categorical_transformer, CATEGORICAL_FEATURES)
        ]
    )

def preprocess_data():
    import os
    df = load_data_from_sql()
    binary_features = ['hepatitis_b', 'hepatitis_c', 'cirrhosis_history', 'family_history_cancer', 'diabetes']
    all_features = NUMERIC_FEATURES + CATEGORICAL_FEATURES + binary_features
    X = df[all_features]
    y = df[TARGET]
    preprocessor = build_preprocessor()
    X_processed = preprocessor.fit_transform(X)
    
    # Crear directorio models si no existe
    os.makedirs("models", exist_ok=True)
    joblib.dump(preprocessor, "models/preprocessor.pkl")
    return X_processed, y.values