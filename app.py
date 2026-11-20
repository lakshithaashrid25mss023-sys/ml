from pathlib import Path

from fastapi import FastAPI
import joblib
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "Models" / "model.pkl"

app = FastAPI()
model = joblib.load(MODEL_PATH)


@app.get("/")
def home():
    return {"message": "Student Performance Prediction API running"}

@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(study_hours: float, attendance: float, previous_marks:float):
    input_data = pd.DataFrame(
        [{"study_hours": study_hours, "attendance": attendance, "previous_marks": previous_marks}]
    )
    prediction = model.predict(input_data)
    return {"prediction_result": prediction[0]}

