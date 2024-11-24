import numpy as np
from PIL import Image
import torch
from torchvision import transforms
import unet_model
import warnings
from preprocessing import *
from config import device
warnings.filterwarnings('ignore')


# Parse mask for saving
def mask_parse(mask):
    mask = np.expand_dims(mask, axis=-1)  # (512, 512, 1)
    mask = np.concatenate([mask, mask, mask], axis=-1)  # (512, 512, 3)
    return mask

# Load and preprocess image
def load_image(image_path):
    if image_path.endswith(('.png', '.jpg', '.jpeg', '.tif')):
        image = Image.open(image_path).convert('RGB')
        image = image.resize((512, 512))
        image = transform(image).unsqueeze(0).to(device)  # Move tensor to device
        return image

# Get mask prediction from the model
def get_prediction(model, image_tensor):
    with torch.no_grad():
        output = model(image_tensor)
        output = torch.sigmoid(output)
        output = output[0].cpu().numpy()  # Move to CPU for further processing
        output = np.squeeze(output, axis=0)
        output = output > 0.5
        output = np.array(output, dtype=np.uint8)
        return output

# Superpose masks on the original image with sigmoid correction
def superpose_masks(original_image, bv_mask, ridge_mask):
    original_image_np = np.array(original_image)
    corrected_image_np = adaptive_sigmoid(original_image_np)
    
    # Initialize the combined mask with 3 channels (for RGB)
    combined_mask = np.zeros((bv_mask.shape[0], bv_mask.shape[1], 3), dtype=np.uint8)

    # Color coding for the masks
    combined_mask[bv_mask == 1] = [179, 2, 2]
    combined_mask[ridge_mask == 1] = [25, 10, 242]
    
    # Superpose the masks on the sigmoid-corrected original image
    superposed_image = corrected_image_np.copy()
    mask_indices = combined_mask > 0
    superposed_image[mask_indices] = combined_mask[mask_indices]
    
    return Image.fromarray(superposed_image)

# Process a single image
def process_single_image(bv_image_path, ridge_image_path, output_path):
    bv_image_tensor = load_image(bv_image_path)
    ridge_image_tensor = load_image(ridge_image_path)
    original_image = Image.open(bv_image_path).convert('RGB').resize((512, 512))

    # Generate masks
    bv_mask = get_prediction(bv_model, bv_image_tensor)
    ridge_mask = get_prediction(ridge_model, ridge_image_tensor)

    # Superpose the masks onto the original image
    superposed_image = superpose_masks(original_image, bv_mask, ridge_mask)

    # Save the output image
    superposed_image.save(output_path)
    print(f"Superposed image saved at {output_path}")

# Load and move models to device
bv_model = unet_model.build_unet().to(device)
bv_model.load_state_dict(torch.load(r'Models\bv_no_enhancement.pth', map_location=device))
bv_model.eval()

ridge_model = unet_model.build_unet().to(device)
ridge_model.load_state_dict(torch.load(r'Models\gabor_ridge_aug.pth', map_location=device))
ridge_model.eval()

# Image transformation
transform = transforms.Compose([transforms.ToTensor()])

if __name__ == "__main__":
    bv_image_path = r''  # Add image path
    ridge_image_path = r''  # Add ridge mask path
    output_path = r''  # Specify output path

    process_single_image(bv_image_path, ridge_image_path, output_path)
