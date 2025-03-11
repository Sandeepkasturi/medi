import streamlit as st
import google.generativeai as genai
from PIL import Image
import io
import cv2
import numpy as np

# Set API Key (replace with your actual API key)
genai.configure(api_key="AIzaSyA0MDtCJ7v9Dhu83cK1dgcjGxmiH6CxrWg")

# Streamlit UI configuration
st.set_page_config(layout="wide")
# Apply custom CSS for background image
st.markdown(
    """
    <style>
        /* Full-screen background */
        .stApp {
            background: url("https://static.vecteezy.com/system/resources/previews/006/712/985/non_2x/abstract-health-medical-science-healthcare-icon-digital-technology-science-concept-modern-innovation-treatment-medicine-on-hi-tech-future-blue-background-for-wallpaper-template-web-design-vector.jpg") no-repeat center center fixed;
            background-size: cover;
        }
        /* Adjust text color for better visibility */
        h1, h5, p, label {
            color: white !important;
        }
    </style>
    """,
    unsafe_allow_html=True
)




st.markdown(
    """
    <h1 style="text-align: center;color:#F5F5F5">MEDI Vision AI</h1>
    <h5 style="text-align: center;color:#FFFDF0">Upload an X-ray or other medical image for AI-powered analysis and assistance</h5>
    """,
    unsafe_allow_html=True
)

# Layout: Split the page into two columns
col1, col2 = st.columns(2)

# Function to capture image from webcam
def capture_webcam():
    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    if ret:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        cap.release()
        return Image.fromarray(frame)
    cap.release()
    return None

with col1:
    # Option to capture image from webcam
    # Custom CSS for button styling
    st.markdown(
        """
        <style>
            div.stButton > button:first-child {
                background-color: #4CAF50; /* Green background */
                color: white; /* White text */
                font-size: 16px;
                border-radius: 8px;
                padding: 10px 20px;
            }
            div.stButton > button:hover {
                background-color: #45a049; /* Darker green on hover */
            }
        </style>
        """,
        unsafe_allow_html=True
    )

    # Styled button for capturing image
    if st.button("📷 Capture Image from Webcam"):
        captured_image = capture_webcam()
    else:
        captured_image = None

    # Option to upload an image file
    uploaded_file = st.file_uploader("Or upload a medical image (X-ray, MRI, etc.)", type=["jpg", "jpeg", "png"])

    # Input for a medical query (could be additional context)
    prompt = st.text_area("Enter your medical query:", "Analyze this medical image and provide insights.")

    # Select the AI model
    gemini_models = [
        "gemini-1.5-flash-latest",
        "gemini-2.0-flash-lite",
        "gemini-2.0-flash",
        "gemini-2.0-pro-exp-02-05"
    ]
    model_name = st.selectbox("Select AI Model:", gemini_models)

    # Process the image: use captured image if available; otherwise, the uploaded image
    image = captured_image if captured_image else (Image.open(uploaded_file) if uploaded_file else None)
    analyze_button = st.button("Analyze Medical Image")

if image and analyze_button:
    # Resize the image to fit within 512x512 pixels using a high-quality resampling method
    image.thumbnail([200, 200], Image.Resampling.LANCZOS)
    st.image(image, caption="Processed Medical Image", use_container_width=True)

    # Convert the image to bytes for processing
    img_byte_array = io.BytesIO()
    image.save(img_byte_array, format='PNG')
    img_byte_array = img_byte_array.getvalue()

    # Use the generative AI model to generate a diagnosis
    model = genai.GenerativeModel(model_name)
    response = model.generate_content([prompt, {"mime_type": "image/png", "data": img_byte_array}])
    diagnosis = response.text.strip()

    # Now generate treatment suggestions including specific medicines
    treatment_prompt = (
        f"Based on the following diagnosis: {diagnosis}, provide a recommended treatment plan. "
        "Include specific medicine suggestions such as ointments, tablets, or syrups. "
        "Also mention additional care instructions if needed."
    )
    treatment_response = model.generate_content([treatment_prompt])
    treatment_suggestion = treatment_response.text.strip()

    # Display AI Medical Analysis & Recommended Treatment side by side
    analysis_col, treatment_col = st.columns(2)

    with analysis_col:
        st.markdown(
            f"""
            <div style="background-color:#121212; padding:15px; border-radius:10px; text-align:center;">
                <h2 style="color:white;">AI Medical Analysis</h2>
                <p style="color:pink;">{diagnosis}</p>
            </div>
            """,
            unsafe_allow_html=True
        )

    with treatment_col:
        st.markdown(
            f"""
            <div style="background-color:#121212; padding:15px; border-radius:10px; text-align:center;">
                <h2 style="color:white;">Recommended Treatment</h2>
                <p style="color:pink;">{treatment_suggestion}</p>
            </div>
            """,
            unsafe_allow_html=True
        )
