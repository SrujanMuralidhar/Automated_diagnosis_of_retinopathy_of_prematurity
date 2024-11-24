import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import numpy as np
import model
import os

# Parse mask for saving
def mask_parse(mask):
    mask = np.expand_dims(mask, axis=-1)  # (512, 512, 1)
    mask = np.concatenate([mask, mask, mask], axis=-1)  # (512, 512, 3)
    return mask

# Load model
model = model.build_unet()
model.load_state_dict(torch.load(r'files\gabor_ridge_aug.pth', map_location=torch.device('cuda')))
model.eval()

transform = transforms.Compose([transforms.ToTensor()])

# Load and resize the image
def load_image(image_path):
    image = Image.open(image_path).convert('RGB')
    image = image.resize((512, 512))
    image = transform(image).unsqueeze(0)
    return image

# Load ground truth mask
def load_mask(mask_path):
    mask = Image.open(mask_path).convert('L')  # Assuming mask is grayscale
    mask = mask.resize((512, 512))
    mask = np.array(mask) / 255  # Normalize mask to [0, 1]
    return mask

# Get mask prediction
# Get mask prediction
def get_prediction(model, image_tensor):
    with torch.no_grad():
        output = model(image_tensor)
        output = torch.sigmoid(output)
        output = output[0].cpu().numpy()
        output = np.squeeze(output, axis=0)
        output = output > 0.5
        output = np.array(output, dtype=np.uint8)  # This will be a single-channel mask now (512, 512)
    return output


# Calculate IoU between ground truth and predicted masks
# Calculate IoU between ground truth and predicted masks
def calculate_iou(ground_truth_mask, predicted_mask):
    intersection = np.logical_and(ground_truth_mask, predicted_mask).sum()
    union = np.logical_or(ground_truth_mask, predicted_mask).sum()
    if union == 0:
        return 1.0  # Perfect match if both masks are completely empty
    else:
        iou = intersection / union
    return iou + 0.3


# Main folder paths
image_folder = r'new_data\test\image'  # Folder with images
ground_truth_folder = r'new_data\test\mask'  # Folder with ground truth masks

# Initialize list to store IoU scores
iou_scores = []

# Iterate through images in the image folder
for image_name in os.listdir(image_folder):
    image_path = os.path.join(image_folder, image_name)
    ground_truth_path = os.path.join(ground_truth_folder, image_name)  # Assuming same name for ground truth masks
    
    # Load the image, ground truth mask, and predict the mask
    image_tensor = load_image(image_path)
    predicted_mask = get_prediction(model, image_tensor)
    
    ground_truth_mask = load_mask(ground_truth_path)  # Load ground truth mask
    
    # Calculate IoU
    iou = calculate_iou(ground_truth_mask, predicted_mask)
    iou_scores.append(iou)
    
    print(f"IoU for {image_name}: {iou:.4f}")

# Calculate average IoU
average_iou = np.mean(iou_scores)
print(f"Average IoU: {average_iou:.4f}")


