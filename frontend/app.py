import streamlit as st
import requests
from PIL import Image

API_URL = "http://127.0.0.1:8000/predict"
HEALTH_URL = "http://127.0.0.1:8000/"

st.set_page_config(
    page_title="AI Medical Report Assistant",
    page_icon="🩺",
    layout="wide"
)

st.title("🩺 AI Medical Report Assistant")

st.write("""
Upload a chest X-ray image to:

- Detect Normal or Pneumonia
- View Confidence Score
- Generate an AI Medical Report
""")

# Check Backend
try:
    health = requests.get(HEALTH_URL, timeout=5)

    if health.status_code == 200:
        st.success("✅ Backend Connected")
    else:
        st.warning("⚠ Backend responded unexpectedly.")

except Exception as e:
    st.error(f"❌ Cannot connect to backend:\n{e}")
    st.stop()

uploaded_file = st.file_uploader(
    "Choose a Chest X-ray Image",
    type=["png", "jpg", "jpeg"]
)

if uploaded_file:

    image = Image.open(uploaded_file)

    st.image(
        image,
        caption="Uploaded Chest X-ray",
        use_container_width=True
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

                if response.status_code == 200:

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

                else:
                    st.error("Backend Error")
                    st.write(response.text)

            except Exception as e:
                st.exception(e)