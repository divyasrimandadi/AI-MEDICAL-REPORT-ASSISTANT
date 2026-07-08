import streamlit as st
import requests
from PIL import Image

API_URL = "http://127.0.0.1:8000/predict"

st.set_page_config(
    page_title="AI Medical Report Assistant",
    page_icon="🩺",
    layout="wide"
)

st.title("🩺 AI Medical Report Assistant")

st.write(
    "Upload a Chest X-ray image to predict the disease and generate an AI medical report."
)

uploaded_file = st.file_uploader(
    "Choose a Chest X-ray Image",
    type=["png", "jpg", "jpeg"]
)

if uploaded_file:

    image = Image.open(uploaded_file)

    st.image(
        image,
        caption="Uploaded Chest X-ray",
        width="stretch"     # replaces use_container_width=True
    )

    if st.button("Analyze Image"):

        with st.spinner("Analyzing image..."):

            files = {
                "file": (
                    uploaded_file.name,
                    uploaded_file.getvalue(),
                    uploaded_file.type
                )
            }

            try:
                response = requests.post(
                    API_URL,
                    files=files,
                    timeout=60
                )

                response.raise_for_status()

                result = response.json()

                st.success("Analysis Completed Successfully")

                col1, col2 = st.columns(2)

                with col1:
                    st.metric(
                        "Prediction",
                        result["prediction"]
                    )

                with col2:
                    st.metric(
                        "Confidence",
                        f"{result['confidence']:.2f}%"
                    )

                st.subheader("AI Medical Report")

                st.markdown(result["report"])

            except requests.exceptions.ConnectionError:
                st.error("❌ Cannot connect to backend. Make sure FastAPI is running on port 8000.")

            except requests.exceptions.Timeout:
                st.error("❌ Backend timed out.")

            except Exception as e:
                st.error(f"❌ {e}")