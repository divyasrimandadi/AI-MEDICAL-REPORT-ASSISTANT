from google import genai

from backend.config import GEMINI_API_KEY

client = genai.Client(api_key=GEMINI_API_KEY)


def generate_medical_report(prediction, confidence):

    fallback_report = f"""
# AI Medical Report

## Prediction
{prediction}

## Confidence
{confidence:.2f}%

## Findings
The uploaded chest X-ray image was analyzed using an AI-powered medical imaging system.

## Impression
Radiological patterns suggest possible signs associated with {prediction.lower()}.

## Severity Assessment
Moderate suspicion based on AI findings.

## Recommendation
- Clinical correlation is recommended.
- Follow-up imaging may be required if symptoms persist.
- Please consult a qualified physician or radiologist.

## Disclaimer
This report is AI-generated and should not replace professional medical diagnosis.
"""

    prompt = f"""
You are an expert radiologist.

Prediction:
{prediction}

Confidence:
{confidence:.2f}%

Generate a professional chest X-ray report.

Use exactly the following format.

# AI Medical Report

## Prediction

## Findings

## Impression

## Severity Assessment

## Recommendation

## Disclaimer

Use professional medical language suitable for a clinical report.
"""

    try:

        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
        )

        if response.text:
            return response.text

        return fallback_report

    except Exception as e:

        print("Gemini Error:", e)

        return fallback_report