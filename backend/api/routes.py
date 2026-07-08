from fastapi import APIRouter, UploadFile, File, HTTPException
from PIL import Image
import io

from backend.services.predictor import predict_image
from backend.services.report_generator import generate_medical_report
from backend.services.database_service import save_prediction

router = APIRouter()


@router.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        print("1. Request received")

        if not file.content_type.startswith("image/"):
            raise HTTPException(status_code=400, detail="Invalid image file")

        contents = await file.read()
        print("2. File read")

        image = Image.open(io.BytesIO(contents)).convert("RGB")
        print("3. Image opened")

        prediction, confidence = predict_image(image)
        print("4. Prediction:", prediction, confidence)

        report = generate_medical_report(prediction, confidence)
        print("5. Report generated")

        save_prediction(
            file.filename,
            prediction,
            confidence,
            report
        )
        print("6. Saved to database")

        return {
            "filename": file.filename,
            "prediction": prediction,
            "confidence": confidence,
            "report": report
        }

    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"error": str(e)}