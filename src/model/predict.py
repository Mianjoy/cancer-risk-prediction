import pandas as pd
import joblib
import torch
from .model import create_model
from ..data_engineering.preprocess import NUMERIC_FEATURES, CATEGORICAL_FEATURES

def predict_risk(input_dict):
    # Cargar preprocessor
    preprocessor = joblib.load("models/preprocessor.pkl")
    
    # Preparar datos de entrada
    binary_features = ['hepatitis_b', 'hepatitis_c', 'cirrhosis_history', 'family_history_cancer', 'diabetes']
    all_features = NUMERIC_FEATURES + CATEGORICAL_FEATURES + binary_features
    X = pd.DataFrame([input_dict], columns=all_features)
    X_processed = preprocessor.transform(X)
    
    # Convertir a tensor de PyTorch
    X_tensor = torch.FloatTensor(X_processed)
    
    # Cargar modelo
    model = create_model(X_processed.shape[1])
    model.load_state_dict(torch.load("models/cancer_risk_model.pth"))
    model.eval()
    
    # Hacer predicción
    with torch.no_grad():
        prob_cancer = model(X_tensor).item()
    
    return float(prob_cancer)