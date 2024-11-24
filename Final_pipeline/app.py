import streamlit as st
import requests
import os
from PIL import Image
from config import ROP_MAP, STAGE_WISE_MAP

st.title("RetiScan")

# Input fields
PMA = st.number_input("Enter PMA", min_value=35, max_value=120, step=1)
uploaded_image = st.file_uploader("Upload Retinal Image", type=["png", "jpg", "jpeg"])

# Button to submit the data
if st.button("Submit"):
    if uploaded_image and PMA:
        # Send POST request to Flask API
        response = requests.post("http://127.0.0.1:5000/process", files={"image": uploaded_image}, data={"PMA": PMA})

        # Check if request was successful
        if response.status_code == 200:
            result = response.json()
            st.write("### Prediction Result:")
            
            result_text = ""
            if "result" in result:
                if result['result'] == ROP_MAP[0]:
                    if int(PMA) < 40:
                        result_text = f"Diagnosis: {result['result']} -> Temporal Avascular Retina"
                    else:
                        result_text = f"Diagnosis: {result['result']} -> Fully Vascularized Retina"
                    st.success(result_text)
                else:
                    result_text = f"**Diagnosis:** {ROP_MAP[1]} -> {result['result']}\n"
                    st.warning(result_text)
            
            # Display images from folder in a single row with image names as captions after diagnosis
            st.subheader("Enhanced Retinal Images")
            folder_path = "pipeline-output"  # Update to your actual folder path
            image_files = [f for f in os.listdir(folder_path) if f.endswith(('png', 'jpg', 'jpeg'))]

            # Create columns dynamically based on the number of images
            columns = st.columns(len(image_files))
            for col, image_file in zip(columns, image_files):
                img = Image.open(os.path.join(folder_path, image_file))
                # Display image with filename as caption
                col.image(img, caption=image_file, use_column_width=True)

        else:
            st.error(f"Error: {response.status_code} - {response.text}")
    else:
        st.warning("Please upload an image and enter PMA.")
