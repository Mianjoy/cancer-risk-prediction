"""API de predicción de riesgo de cáncer de hígado.

Provee un endpoint POST `/predict` que recibe características clínicas y de estilo de vida
de un paciente y devuelve la probabilidad estimada (en %) y un mensaje clínico
orientativo basado en el umbral.
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from src.model.predict import predict_risk

# Inicializa la aplicación FastAPI con un título descriptivo
app = FastAPI(title="Liver Cancer Risk Prediction API")

# Configuración de CORS: habilita acceso desde cualquier origen para facilitar pruebas front-end
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class PatientInput(BaseModel):
    """Esquema de entrada esperado por el endpoint de predicción.

    Campos:
    - age: edad en años.
    - gender: 'Male' | 'Female'.
    - bmi: índice de masa corporal.
    - alcohol_consumption: nivel de consumo de alcohol.
    - smoking_status: estado de tabaquismo.
    - hepatitis_b / hepatitis_c: 0 no, 1 sí.
    - liver_function_score: marcador agregado de función hepática.
    - alpha_fetoprotein_level: nivel de AFP.
    - cirrhosis_history, family_history_cancer, diabetes: 0/1 binarios.
    - physical_activity_level: nivel de actividad física.
    """

    age: int
    gender: str
    bmi: float
    alcohol_consumption: str
    smoking_status: str
    hepatitis_b: int
    hepatitis_c: int
    liver_function_score: float
    alpha_fetoprotein_level: float
    cirrhosis_history: int
    family_history_cancer: int
    physical_activity_level: str
    diabetes: int


@app.post("/predict")
def predict(data: PatientInput):
    """Calcula el riesgo de cáncer de hígado para un paciente dado.

    Retorna un porcentaje redondeado y un mensaje clínico simple según umbral 50%.
    """
    try:
        # Ejecuta la predicción usando el pipeline del modelo
        prob = predict_risk(data.dict())
        risk_pct = prob * 100

        # Mensaje clínico heurístico: >50% sugiere atención inmediata
        message = (
            "Recomendación de seguimiento/chequeos." if risk_pct <= 50 else "Alerta: Cita clínica inmediata."
        )
        return {"risk_percentage": round(risk_pct, 2), "clinical_message": message}
    except Exception as e:
        # Encapsula errores internos como HTTP 500 con detalle textual
        raise HTTPException(status_code=500, detail=str(e))