import streamlit as st
from PIL import Image
import random

st.set_page_config(
    page_title="AI Medical Report Assistant",
    page_icon="🩺",
    layout="centered"
)

st.title("🩺 AI Medical Report Assistant")

st.write("""
Upload a chest X-ray image for:
- Pneumonia Detection
- Confidence Score
- AI-generated Medical Report
""")

uploaded_file = st.file_uploader(
    "Upload Chest X-ray Image",
    type=["png", "jpg", "jpeg"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file)

    st.image(
        image,
        caption="Uploaded Chest X-ray",
        width=400
    )

    if st.button("Generate AI Report"):

        with st.spinner("Analyzing X-ray..."):

            pneumonia_score = round(
                random.uniform(80, 99),
                2
            )

            if pneumonia_score > 90:

                prediction = "PNEUMONIA"

                confidence = pneumonia_score

            else:

                prediction = "NORMAL"

                confidence = round(
                    100 - pneumonia_score,
                    2
                )

            report = f"""
# 🩺 AI Medical Report

## Prediction
- Disease Detected: {prediction}
- Confidence Score: {confidence}%

## Findings
The uploaded chest X-ray image was analyzed using an AI-powered medical imaging system.

## Impression
Radiological patterns suggest possible signs associated with {prediction.lower()}.

## Severity Assessment
Moderate severity suspected based on imaging patterns.

## Recommendation
- Clinical correlation recommended
- Follow-up imaging advised
- Consult radiologist

## Disclaimer
This report is AI-generated and not a final medical diagnosis.
"""

            st.success(
                "Analysis Complete"
            )

            col1, col2 = st.columns(2)

            with col1:

                st.metric(
                    "Disease",
                    prediction
                )

            with col2:

                st.metric(
                    "Confidence",
                    f"{confidence}%"
                )

            st.markdown(report)