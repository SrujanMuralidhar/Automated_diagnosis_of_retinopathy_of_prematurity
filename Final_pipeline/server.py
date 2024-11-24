# server.py
import os
import cv2
import torch
from flask import Flask, request, jsonify
from preprocessing import apply_adaptive_sigmoid, apply_gabor
from generate_masks import process_single_image
from classifier import classify
from config import *
import warnings
warnings.filterwarnings('ignore')

# Initialize Flask app
app = Flask(__name__)

# Output folder setup
output_folder = r"pipeline-output"
if not os.path.exists(output_folder):
    os.makedirs(output_folder)


# Define function for the pipeline
def execute_pipeline(image_path, binary_classifier_path, stage_wise_classifier_path,PMA):
    original_image = cv2.imread(image_path)
    original_image = cv2.resize(original_image, size, interpolation=cv2.INTER_AREA)
    
    # Save original image to output folder
    original_image_path = os.path.join(output_folder, "1_original_image.png")
    # cv2.imwrite(original_image_path, original_image)

    # Apply preprocessing steps
    apply_gabor(original_image, output_folder)
    apply_adaptive_sigmoid(original_image, output_folder)
    
    # Generate masks and superpose them on the original image
    blood_vessel_path = os.path.join(output_folder, "1_input_image.png")
    ridge_path = os.path.join(output_folder, "2_gabor_image.png")
    output_path = os.path.join(output_folder, "3_superposed_image.png")
    process_single_image(blood_vessel_path, ridge_path, output_path)
    
    # Perform classification using the binary classifier
    binary_predicted_class = classify(binary_classifier_path, 2, output_path)
    if binary_predicted_class == 0:
        # print("result: ", ROP_MAP[binary_predicted_class])
        # if(int(PMA) < 40):
        #     return {"result": ROP_MAP[binary_predicted_class] + " -> "+ "Temporal Avascular Retina"} # TAR
        
        # return {"result": ROP_MAP[binary_predicted_class]  + " -> "+ "Fully Vascularized Retina"}  # No ROP detected
        return {"result": ROP_MAP[binary_predicted_class] }  # No ROP detected
    
    else:
        # If ROP is detected, use stage-wise classifier
        stage_wise_predicted_class = classify(stage_wise_classifier_path, 3, output_path)
        # print("result:", STAGE_WISE_MAP[stage_wise_predicted_class])
        return {"result": STAGE_WISE_MAP[stage_wise_predicted_class]}  # ROP stage classification result

# Define route to handle image processing and classification
@app.route('/process', methods=['POST'])
def process():
    if 'image' not in request.files or 'PMA' not in request.form:
        return jsonify({"error": "No image or PMA field provided"}), 400

    # Retrieve image and PMA from request
    image = request.files['image']
    PMA = request.form['PMA']
    
    # Save the input image temporarily for processing
    input_image_path = os.path.join(output_folder, "1_input_image.png")
    image.save(input_image_path)
    
    # Define paths to the models
    binary_classifier_model_path = r'Models\ResNet50_Rop_NoRop.pth'
    stage_wise_classifier_model_path = r'Models\ResNet50_3Stages.pth'
    
    # Execute the pipeline
    result = execute_pipeline(input_image_path, binary_classifier_model_path, stage_wise_classifier_model_path,PMA)
    
    # Add PMA to the result (if required)
    result["PMA"] = PMA
    
    return jsonify(result)

if __name__ == "__main__":
    app.run(debug=True)
