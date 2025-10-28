from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from src.model.predict import predict_risk

app = FastAPI(title="Liver Cancer Risk Prediction API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class PatientInput(BaseModel):
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
    try:
        prob = predict_risk(data.dict())
        risk_pct = prob * 100
        message = "Recomendación de seguimiento/chequeos." if risk_pct <= 50 else "Alerta: Cita clínica inmediata."
        return {"risk_percentage": round(risk_pct, 2), "clinical_message": message}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))