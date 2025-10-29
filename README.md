# 🩺 Aplicación de Predicción de Riesgo de Cáncer de Hígado

Una aplicación completa de Deep Learning que predice el riesgo de cáncer de hígado en pacientes utilizando una Red Neuronal Multicapa (MLP) implementada en PyTorch.

## 🎯 Características

- **Modelo de Deep Learning:** Red neuronal multicapa con PyTorch
- **API REST:** Backend con FastAPI
- **Interfaz Web:** Frontend responsive con HTML/CSS/JavaScript
- **Preprocesamiento:** Normalización y codificación de datos categóricos
- **Métricas de Evaluación:** Accuracy: 78.18%, AUC: 87.18%

## 🛠️ Tecnologías Utilizadas

### Backend
- **Python 3.14**
- **PyTorch 2.9.0** - Deep Learning
- **FastAPI 0.120.1** - API REST
- **Pandas 2.3.3** - Manipulación de datos
- **Scikit-learn 1.7.2** - Preprocesamiento
- **NumPy 2.3.4** - Computación numérica

### Frontend
- **HTML5/CSS3/JavaScript**
- **Fetch API** - Comunicación con backend

## 📁 Estructura del Proyecto

```
cancer-risk-prediction/
├── src/
│   ├── backend/
│   │   └── main.py              # API FastAPI
│   ├── data_engineering/
│   │   ├── load_sql.py          # Carga de datos
│   │   └── preprocess.py        # Preprocesamiento
│   ├── model/
│   │   ├── model.py             # Arquitectura PyTorch
│   │   ├── train.py             # Entrenamiento
│   │   └── predict.py           # Predicción
│   └── frontend/
│       ├── index.html           # Interfaz web
│       ├── style.css            # Estilos
│       └── script.js            # Lógica frontend
├── data/
│   └── synthetic_liver_cancer_dataset.sql
├── models/
│   ├── cancer_risk_model.pth    # Modelo entrenado
│   └── preprocessor.pkl         # Preprocesador
└── requirements.txt
```

## 🚀 Instalación y Uso

### 1. Instalar Dependencias

```bash
python3 -m pip install -r requirements.txt
```

### 2. Entrenar el Modelo

```bash
python3 -m src.model.train
```

### 3. Ejecutar el Backend

```bash
python3 -m uvicorn src.backend.main:app --reload --port 8000
```

### 4. Abrir la Aplicación Web

Abre `src/frontend/index.html` en tu navegador web.

## 📊 Modelo de Deep Learning

### Arquitectura
- **Capa de entrada:** 13 características
- **Capa oculta 1:** 64 neuronas + ReLU + Dropout(0.3)
- **Capa oculta 2:** 32 neuronas + ReLU + Dropout(0.3)
- **Capa de salida:** 1 neurona + Sigmoid

### Características de Entrada
- **Numéricas:** edad, BMI, puntaje función hepática, nivel AFP
- **Categóricas:** género, consumo alcohol, estado fumador, actividad física
- **Binarias:** hepatitis B/C, cirrosis, antecedentes familiares, diabetes

### Valores Categóricos Esperados
- **Género:** Male, Female
- **Consumo de alcohol:** None, Light, Moderate, Heavy
- **Estado de fumador:** Never, Former, Current
- **Actividad física:** Low, Moderate, High

## 🔬 Lógica Clínica

| Riesgo (%) | Mensaje de Acción |
|------------|-------------------|
| 0-50%      | Recomendación de seguimiento/chequeos |
| 51%+       | Alerta: Cita clínica inmediata |

## 📈 Métricas del Modelo

- **Accuracy:** 78.18%
- **AUC:** 87.18%
- **Épocas de entrenamiento:** 50
- **Pérdida final:** 0.4534 (entrenamiento), 0.4567 (validación)

## 🌐 API Endpoints

### POST /predict
Predice el riesgo de cáncer de hígado.

**Entrada:**
```json
{
  "age": 32,
  "gender": "Female",
  "bmi": 21.5,
  "alcohol_consumption": "None",
  "smoking_status": "Never",
  "hepatitis_b": 0,
  "hepatitis_c": 0,
  "liver_function_score": 4.2,
  "alpha_fetoprotein_level": 3.80,
  "cirrhosis_history": 0,
  "family_history_cancer": 0,
  "physical_activity_level": "High",
  "diabetes": 0
}
```

**Salida:**
```json
{
  "risk_percentage": 13.13,
  "clinical_message": "Recomendación de seguimiento/chequeos."
}
```

## 📚 Documentación API

Visita `http://localhost:8000/docs` para la documentación interactiva de la API.

## 🔧 Desarrollo

### Entrenar Nuevo Modelo
```bash
python3 -m src.model.train
```

### Probar Predicción Directa
```bash
python3 -c "from src.model.predict import predict_risk; print(predict_risk({...}))"
```

## 📝 Notas Importantes

- El modelo fue entrenado con datos sintéticos
- Los valores categóricos deben coincidir exactamente con los valores de entrenamiento
- El preprocesador se guarda automáticamente durante el entrenamiento
- El modelo usa PyTorch en lugar de TensorFlow para compatibilidad con Python 3.14

## 🤝 Contribuciones

Este proyecto fue desarrollado como parte de un taller de Deep Learning para predicción de riesgo de cáncer.

## 📄 Licencia

Proyecto académico - Uso educativo.

## 🙌 Créditos y Asistencia

Este documento fue elaborado con la ayuda de la IA Qwen y el IDE Cursor.