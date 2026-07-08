import google.generativeai as genai

from backend.config import GEMINI_API_KEY

genai.configure(api_key=GEMINI_API_KEY)

model = genai.GenerativeModel("gemini-2.5-flash")


def generate_medical_report(prediction, confidence):

    prompt = f"""
You are an experienced medical AI assistant.

A chest X-ray image has already been analyzed.

Prediction:
{prediction}

Confidence:
{confidence:.2f}%

Generate a professional medical report.

Use exactly the following format.

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