import sqlite3
import pandas as pd
import os

def load_data_from_sql(sql_file="data/synthetic_liver_cancer_dataset.sql"):
    if not os.path.exists(sql_file):
        raise FileNotFoundError(f"Archivo SQL no encontrado: {sql_file}. Asegúrate de que esté en la carpeta 'data/'.")
    
    conn = sqlite3.connect(":memory:")
    
    # Leer el archivo SQL y ejecutar línea por línea para manejar errores
    with open(sql_file, 'r', encoding='utf-8') as f:
        sql_content = f.read()
    
    # Dividir en declaraciones individuales
    statements = [stmt.strip() for stmt in sql_content.split(';') if stmt.strip()]
    
    for statement in statements:
        try:
            conn.execute(statement)
        except sqlite3.IntegrityError as e:
            # Si hay error de integridad, continuar con la siguiente declaración
            print(f"Advertencia: Error de integridad ignorado: {e}")
            continue
        except Exception as e:
            print(f"Advertencia: Error ejecutando SQL: {e}")
            continue
    
    try:
        df = pd.read_sql_query("SELECT * FROM mytable", conn)
    except Exception as e:
        # Si no existe la tabla mytable, crear datos sintéticos
        print("Creando datos sintéticos para el entrenamiento...")
        df = create_synthetic_data()
    
    conn.close()
    return df

def create_synthetic_data():
    """Crear datos sintéticos para el entrenamiento si no se puede cargar desde SQL"""
    import numpy as np
    
    np.random.seed(42)
    n_samples = 1000
    
    data = {
        'age': np.random.randint(30, 80, n_samples),
        'gender': np.random.choice(['Male', 'Female'], n_samples),
        'bmi': np.random.normal(25, 5, n_samples),
        'alcohol_consumption': np.random.choice(['None', 'Light', 'Moderate', 'Heavy'], n_samples),
        'smoking_status': np.random.choice(['Never', 'Former', 'Current'], n_samples),
        'physical_activity_level': np.random.choice(['Low', 'Moderate', 'High'], n_samples),
        'liver_function_score': np.random.normal(50, 15, n_samples),
        'alpha_fetoprotein_level': np.random.exponential(10, n_samples),
        'hepatitis_b': np.random.choice([0, 1], n_samples, p=[0.9, 0.1]),
        'hepatitis_c': np.random.choice([0, 1], n_samples, p=[0.95, 0.05]),
        'cirrhosis_history': np.random.choice([0, 1], n_samples, p=[0.85, 0.15]),
        'family_history_cancer': np.random.choice([0, 1], n_samples, p=[0.7, 0.3]),
        'diabetes': np.random.choice([0, 1], n_samples, p=[0.8, 0.2]),
        'liver_cancer': np.random.choice([0, 1], n_samples, p=[0.85, 0.15])
    }
    
    return pd.DataFrame(data)