import os
from dotenv import load_dotenv
from google import generativeai as genai

load_dotenv()

API_KEY = os.getenv("GEMINI_API_KEY")

model = None

try:

    if API_KEY:

        genai.configure(
            api_key=API_KEY
        )

        model = genai.GenerativeModel(
            "gemini-1.5-flash"
        )

        print("Gemini Connected")

except Exception as e:

    print("Gemini Error:", e)


def generate_medical_report(
    prediction,
    confidence
):

    fallback_report = f"""
# 🩺 AI Medical Report

## Prediction
- Disease Detected: {prediction}
- Confidence Score: {confidence}%

## Findings
The uploaded chest X-ray image was analyzed using a deep learning medical imaging model.

Radiological patterns suggest possible signs associated with {prediction.lower()}.

## Impression
AI analysis indicates a probability of {confidence}% for {prediction.lower()}.

## Severity Assessment
- Mild to Moderate suspicion based on imaging patterns.

## Recommendation
- Clinical correlation recommended.
- Follow-up chest imaging may be considered.
- Consultation with a radiologist or pulmonologist is advised.

## Disclaimer
This report is AI-generated and intended only for research and assistance purposes. It is not a substitute for professional medical diagnosis.
"""

    try:

        if model is None:

            return fallback_report

        prompt = f"""
You are an expert AI radiologist assistant.

Generate a professional chest X-ray analysis report.

Patient Chest X-ray AI Prediction:
{prediction}

Confidence Score:
{confidence}%

Generate the report in EXACTLY this structure:

# 🩺 AI Medical Report

## Prediction
- Disease Detected
- Confidence Score

## Findings
Detailed radiological findings.

## Impression
Clinical interpretation.

## Severity Assessment
Mention whether the case appears mild, moderate, or severe.

## Recommendation
Provide next medical recommendations.

## Disclaimer
Mention this is AI-generated and not a final diagnosis.

Requirements:
- Professional medical language
- Concise
- Realistic radiology style
- Clear formatting
"""

        response = model.generate_content(
            prompt
        )

        if response.text:

            return response.text

        return fallback_report

    except Exception as e:

        print(
            "Gemini Generation Error:",
            e
        )

        return fallback_report