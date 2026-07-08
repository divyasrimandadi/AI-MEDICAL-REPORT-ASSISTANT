from fastapi import APIRouter, UploadFile, File
from PIL import Image
import io
import os

from backend.services.predictor import predict_image
from backend.services.report_generator import generate_medical_report
from backend.services.database_service import save_report

router = APIRouter()


@router.post("/predict")
async def predict(file: UploadFile = File(...)):

    image_bytes = await file.read()

    image = Image.open(io.BytesIO(image_bytes))

    prediction_result = predict_image(image)

    prediction = prediction_result["prediction"]

    confidence = prediction_result["confidence"]

    report = generate_medical_report(
        prediction,
        confidence
    )

    save_report(
        filename=file.filename,
        prediction=prediction,
        confidence=confidence,
        report=report
    )

    return {
        "filename": file.filename,
        "prediction": prediction,
        "confidence": confidence,
        "report": report
    }