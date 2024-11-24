import os
import numpy as np
import cv2
from skimage import img_as_float
from skimage.filters import gabor
import warnings
warnings.filterwarnings('ignore')



def enhance_edges(image):
    image = cv2.normalize(image, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    image = np.uint8(image)
    image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    # Convert the image to float for Sato filter
    image_float = img_as_float(image)
    # Define the structuring element (kernel) for morphological operations
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (50, 50))
    # Apply the Top-Hat transformation
    top_hat = cv2.morphologyEx(image, cv2.MORPH_TOPHAT, kernel) 
    # Apply the Bottom-Hat transformation
    bottom_hat = cv2.morphologyEx(image, cv2.MORPH_BLACKHAT, kernel)
    # Enhance the image by adding Top-Hat and subtracting Bottom-Hat
    enhanced_image = cv2.add(image, top_hat)   # Add Top-Hat to enhance bright regions
    enhanced_image = cv2.subtract(enhanced_image, bottom_hat)  # Subtract Bottom-Hat to enhance dark regions
    
    return enhanced_image

def gab(image):
    img = enhance_edges(image)
    filt_real, filt_imaginary = gabor(img,1/4,30,sigma_x=1,sigma_y=1)
    # img = Image.fromarray(filt_real)
    return filt_real

def refine_edges(image):
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    center = (int(image.shape[1] / 2), int(image.shape[0] / 2))
    radius = image.shape[1]//2 - 1
    cv2.circle(mask, center, radius, 255, -1)
    result = cv2.bitwise_and(mask,image)
    return result


def sigmoid_correction(image, k=10, x0=0.5):
    # Normalize the image
    normalized_img = image / 255.0
    # Apply the sigmoid function
    sigmoid_img = 1 / (1 + np.exp(-k * (normalized_img - x0)))
    # Scale back to original range
    corrected_img = (sigmoid_img * 255).astype(np.uint8)
    return corrected_img


def adaptive_sigmoid(image):
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    center = (int(image.shape[1] / 2), int(image.shape[0] / 2))
    radius = image.shape[1]//2
    cv2.circle(mask, center, radius, 255, -1)
    hist = cv2.calcHist([image], [1], mask, [256], [0, 256])
    # Calculate cumulative distribution function (CDF)
    cdf = hist.cumsum()
    # Normalize CDF
    cdf_normalized = cdf * hist.max() / cdf.max()
    # Find the intensity level where CDF reaches 50% of the total pixel count
    total_pixels = cdf[-1]
    x_0 = np.searchsorted(cdf, total_pixels * 0.5)/255
    k = 15
    sig = sigmoid_correction(image,k,x_0)
    return sig



def apply_gabor(original_image,output_folder):
    gab_image = gab(original_image)
    gab_image = refine_edges(gab_image)
    gab_image_path = os.path.join(output_folder, "2_gabor_image.png")
    cv2.imwrite(gab_image_path,gab_image)
    print(f"Gabor image saved at {gab_image_path}")



def apply_adaptive_sigmoid(original_image,output_folder):
    adaptive_sigmoid_image = adaptive_sigmoid(original_image)
    adaptive_sigmoid_image_path = os.path.join(output_folder, "3_adaptive_sigmoid_image.png")
    cv2.imwrite(adaptive_sigmoid_image_path,adaptive_sigmoid_image)
    print(f"Adaptive Sigmoid image saved at {adaptive_sigmoid_image_path}")



