import streamlit as st
import requests
from PIL import Image
<<<<<<< HEAD
import requests

API_URL = "http://127.0.0.1:8001/predict"
HEALTH_URL = "http://127.0.0.1:8001/health"
=======

API_URL = "http://127.0.0.1:8000/predict"
>>>>>>> divya

st.set_page_config(
    page_title="AI Medical Report Assistant",
    page_icon="🩺",
    layout="wide"
)

st.title("🩺 AI Medical Report Assistant")

<<<<<<< HEAD
st.write("""
Upload a chest X-ray image for:

- Pneumonia Detection
- Confidence Score
- AI-generated Medical Report
""")
=======
st.write(
    "Upload a Chest X-ray image to predict the disease and generate an AI medical report."
)
>>>>>>> divya

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

<<<<<<< HEAD
        st.write("✅ Button Clicked")

        try:

=======
        with st.spinner("Analyzing image..."):

>>>>>>> divya
            files = {
                "file": (
                    uploaded_file.name,
                    uploaded_file.getvalue(),
                    uploaded_file.type
                )
            }

<<<<<<< HEAD
            st.write("📤 Sending image to backend...")

            response = requests.post(
                API_URL,
                files=files,
                timeout=60
            )

            st.write("Status Code:", response.status_code)

            result = response.json()

            st.json(result)
=======
            response = requests.post(
                API_URL,
                files=files
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
                        f'{result["confidence"]:.2f}%'
                    )

                st.subheader("AI Medical Report")

                st.markdown(result["report"])
>>>>>>> divya

        except Exception as e:

<<<<<<< HEAD
            st.exception(e)
=======
                st.error("Backend Error")

                st.write(response.text)
>>>>>>> divya
