import os
import numpy as np
import cv2
import warnings
from preprocessing import apply_adaptive_sigmoid,apply_gabor
from generate_masks import process_single_image
from classifier import classify
from config import *
warnings.filterwarnings('ignore')



output_folder = r"pipeline-output"
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

def execute_pipeline(image_path,binary_classifier_path,stage_wise_classifier_path):
    original_image = cv2.imread(image_path)
    original_image = cv2.resize(original_image, size, interpolation=cv2.INTER_AREA)

    # original_image_path = os.path.join(output_folder, "1_original_image.png")
    # cv2.imwrite(original_image_path,original_image)

    apply_gabor(original_image,output_folder)
    apply_adaptive_sigmoid(original_image,output_folder)

    blood_vessel_path = r'pipeline-output\original_image.png'
    ridge_path = r'pipeline-output\gabor_image.png'
    output_path = r'pipeline-output\superposed_image.png'
    process_single_image(blood_vessel_path,ridge_path,output_path)

    binary_predicted_class = classify(binary_classifier_path,2,r'pipeline-output\superposed_image.png')
    if binary_predicted_class == 0:
        print(ROP_MAP[binary_predicted_class])
        pass # to be completed
    else:
        stage_wise_predicted_class = classify(stage_wise_classifier_path,3,r'pipeline-output\superposed_image.png')
        print(STAGE_WISE_MAP[stage_wise_predicted_class])
        



if __name__ == "__main__":
    path = r"C:\Users\xerom\Documents\CAPSTONE\Classification\Data\Stage 2\38.png"
    binary_classifier_model_path = r'Models\ResNet50_Rop_NoRop.pth'
    stage_wise_classifier_model_path = r'Models\ResNet50_3Stages.pth'

    execute_pipeline(path,binary_classifier_model_path,stage_wise_classifier_model_path)

    




