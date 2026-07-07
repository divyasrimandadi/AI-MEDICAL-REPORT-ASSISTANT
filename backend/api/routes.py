from fastapi import APIRouter, UploadFile, File, HTTPException
from PIL import Image
import io

from backend.services.predictor import predict_image
from backend.services.report_generator import generate_medical_report
from backend.services.database_service import save_prediction

router = APIRouter()


@router.post("/predict")
async def predict(
    file: UploadFile = File(...)
):

    try:

        if not file.content_type.startswith("image/"):

            raise HTTPException(
                status_code=400,
                detail="Invalid image file"
            )

        contents = await file.read()

        image = Image.open(
            io.BytesIO(contents)
        ).convert("RGB")

        prediction, confidence = predict_image(
            image
        )

        report = generate_medical_report(
            prediction,
            confidence
        )

        save_prediction(
            file.filename,
            prediction,
            confidence,
            report
        )

        return {
            "filename": file.filename,
            "prediction": prediction,
            "confidence": confidence,
            "report": report
        }

    except Exception as e:

        return {
            "error": str(e)
        }