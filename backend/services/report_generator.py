import os
from dotenv import load_dotenv
from google import genai

load_dotenv()

API_KEY = os.getenv("GEMINI_API_KEY")

print("API KEY:", API_KEY[:15] + "..." if API_KEY else "NOT FOUND")

client = None

try:
    if API_KEY:
        client = genai.Client(api_key=API_KEY)
        print("✅ Gemini Client Connected")
    else:
        print("❌ Gemini API Key Not Found")
except Exception as e:
    print("Gemini Client Error:", e)


def generate_medical_report(prediction, confidence):

    fallback_report = f"""
# 🩺 AI Medical Report

## Prediction
- Disease Detected: {prediction}
- Confidence Score: {confidence}%

## Findings
The uploaded chest X-ray image was analyzed using an AI-powered medical imaging system.

## Impression
Radiological patterns suggest possible signs associated with {prediction.lower()}.

## Severity Assessment
Moderate suspicion based on AI findings.

## Recommendation
- Clinical correlation recommended
- Follow-up imaging advised
- Consult a radiologist

## Disclaimer
This report is AI-generated and should not replace professional medical diagnosis.
"""

    if client is None:
        return fallback_report

    prompt = f"""
You are an expert radiologist.

Prediction:
{prediction}

Confidence:
{confidence}%

Generate a professional chest X-ray report with these sections:

# 🩺 AI Medical Report

## Prediction

## Findings

## Impression

## Severity Assessment

## Recommendation

## Disclaimer

Use professional medical language.
"""

    try:

        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=prompt,
        )

        if response.text:
            return response.text

        return fallback_report

    except Exception as e:
        print("Gemini Error:", e)
        return fallback_report