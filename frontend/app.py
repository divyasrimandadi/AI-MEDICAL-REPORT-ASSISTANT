import streamlit as st
from PIL import Image
import requests

API_URL = "http://127.0.0.1:8001/predict"
HEALTH_URL = "http://127.0.0.1:8001/health"

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

# Check Backend
try:
    health = requests.get(HEALTH_URL, timeout=5)

    if health.status_code == 200:
        st.success("✅ Backend Connected")
    else:
        st.error("❌ Backend Not Responding")
        st.stop()

except Exception as e:
    st.error(f"Cannot connect to backend:\n{e}")
    st.stop()

uploaded_file = st.file_uploader(
    "Upload Chest X-ray Image",
    type=["png", "jpg", "jpeg"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file)

    st.image(
        image,
        caption="Uploaded Chest X-ray",
        use_container_width=True
    )

    if st.button("Generate AI Report"):

        st.write("✅ Button Clicked")

        try:

            files = {
                "file": (
                    uploaded_file.name,
                    uploaded_file.getvalue(),
                    uploaded_file.type
                )
            }

            st.write("📤 Sending image to backend...")

            response = requests.post(
                API_URL,
                files=files,
                timeout=60
            )

            st.write("Status Code:", response.status_code)

            result = response.json()

            st.json(result)

        except Exception as e:

            st.exception(e)