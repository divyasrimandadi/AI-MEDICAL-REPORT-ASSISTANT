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
        width="stretch"
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
                    timeout=120
                )

                st.write("Status Code:", response.status_code)

                if response.status_code != 200:
                    st.error("Backend returned an error.")
                    st.code(response.text)
                    st.stop()

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

            except requests.exceptions.ConnectionError as e:
                st.error("❌ Connection Error")
                st.exception(e)

            except requests.exceptions.Timeout as e:
                st.error("❌ Request Timed Out")
                st.exception(e)

            except requests.exceptions.RequestException as e:
                st.error("❌ Request Failed")
                st.exception(e)

            except Exception as e:
                st.error("❌ Unexpected Error")
                st.exception(e)