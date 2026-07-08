<<<<<<< HEAD
import os
from dotenv import load_dotenv
from google import genai
=======
import google.generativeai as genai
>>>>>>> divya

from backend.config import GEMINI_API_KEY

genai.configure(api_key=GEMINI_API_KEY)

<<<<<<< HEAD
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
=======
model = genai.GenerativeModel("gemini-2.5-flash")
>>>>>>> divya


def generate_medical_report(prediction, confidence):

    prompt = f"""
You are an experienced medical AI assistant.

A chest X-ray image has already been analyzed.

<<<<<<< HEAD
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

=======
>>>>>>> divya
Prediction:
{prediction}

Confidence:
<<<<<<< HEAD
{confidence}%

Generate a professional chest X-ray report with these sections:
=======
{confidence:.2f}%

Generate a professional medical report.
>>>>>>> divya

Use exactly the following format.

<<<<<<< HEAD
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
=======
# AI Medical Report

## Predicted Disease
Mention only the predicted disease.

## Confidence
Mention the confidence percentage.

## Findings
Explain what the disease means in simple medical language.

## Possible Symptoms
Mention common symptoms.

## Recommendations
Mention next medical steps.

## Severity
Low / Moderate / High.

## Disclaimer
Mention that this is AI-generated and not a substitute for a doctor.
"""

    response = model.generate_content(prompt)

    return response.text
>>>>>>> divya
