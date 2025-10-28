from .model import create_model
from ..data_engineering.preprocess import preprocess_data
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import os

def train_and_save():
    X, y = preprocess_data()
    
    # Convertir a tensores de PyTorch
    X_tensor = torch.FloatTensor(X)
    y_tensor = torch.FloatTensor(y).unsqueeze(1)
    
    # Dividir en entrenamiento y validación
    X_train, X_val, y_train, y_val = train_test_split(X_tensor, y_tensor, test_size=0.2, random_state=42)
    
    # Crear datasets y dataloaders
    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    
    # Crear modelo
    model = create_model(X.shape[1])
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Entrenar modelo
    model.train()
    for epoch in range(50):
        total_loss = 0
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        # Validación
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                val_loss += loss.item()
        model.train()
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/50, Train Loss: {total_loss/len(train_loader):.4f}, Val Loss: {val_loss/len(val_loader):.4f}")
    
    # Evaluación final
    model.eval()
    with torch.no_grad():
        y_pred_prob = model(X_tensor).numpy().flatten()
        y_pred = (y_pred_prob > 0.5).astype(int)
        acc = accuracy_score(y, y_pred)
        auc = roc_auc_score(y, y_pred_prob)
        print(f"Accuracy: {acc:.4f}, AUC: {auc:.4f}")
    
    # Guardar modelo
    os.makedirs("models", exist_ok=True)
    torch.save(model.state_dict(), "models/cancer_risk_model.pth")
    print("Modelo guardado en models/cancer_risk_model.pth")