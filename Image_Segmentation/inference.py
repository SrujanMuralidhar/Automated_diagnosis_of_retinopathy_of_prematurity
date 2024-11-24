import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import numpy as np
import model


def mask_parse(mask):
    mask = np.expand_dims(mask, axis=-1)    ## (512, 512, 1)
    mask = np.concatenate([mask, mask, mask], axis=-1)  ## (512, 512, 3)
    return mask

model = model.build_unet()
model.load_state_dict(torch.load(r'files\bv_no_enhancement.pth', map_location=torch.device('cuda')))
model.eval()

transform = transforms.Compose([
    transforms.ToTensor()
])

def load_image(image_path):
    image = Image.open(image_path).convert('RGB') 
    image = image.resize((512,512))
    image = transform(image).unsqueeze(0)  
    return image

def get_prediction(model, image_tensor):
    with torch.no_grad():
        output = model(image_tensor)
        output = torch.sigmoid(output)
        output = output[0].cpu().numpy()
        output = np.squeeze(output,axis=0)
        output = output > 0.5
        output = np.array(output, dtype=np.uint8)
        output = mask_parse(output)
       
    return output

# Load and transform the new image
image_path = r"Data_BV\train_images\3 (10).png"
image_tensor = load_image(image_path)

# Get the predicted mask
predicted_mask = get_prediction(model, image_tensor)

# Save the predicted mask
predicted_mask_image = Image.fromarray(predicted_mask * 255)
predicted_mask_image.save(f'predicted_mask.png')

print("Predicted mask saved as 'predicted_mask.png'")
