from PIL import Image
import torch
from torchvision import transforms
from PIL import Image
import torchvision.transforms as transforms
import torchvision.models as models
import warnings
from config import device
warnings.filterwarnings('ignore')

# If num_classes == 2
'''
0 -> No RoP
1 -> RoP
'''

# If num_classes == 3
'''
0 -> Stage 1
1 -> Stage 2
2 -> Stage 3
'''


# Define transformations for the input image
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize to the model's expected input size
    transforms.ToTensor(),
    transforms.Normalize([0.4970856163948774, 0.3660721661299467, 0.012605847830753192], 
                         [0.3098917880987543, 0.251007258041955, 0.08280670520288899])
])

# Load and preprocess the image
def load_image(image_path):
    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0)  # Add batch dimension
    return image.to(device)

# Prediction function
def predict(image_path, model):
    image = load_image(image_path)
    with torch.no_grad():  # Disable gradient computation for inference
        output = model(image)
        _, predicted = torch.max(output, 1)  # Get the class index with highest score
    return predicted.item()

# Classify image with the model
def classify(model_path, num_classes, image_path):
    # Load ResNet50 model
    model = models.resnet50()
    model.fc = torch.nn.Linear(model.fc.in_features, num_classes)

    # Load model weights and move model to device
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()  # Set model to evaluation mode

    # Perform prediction
    predicted_class = predict(image_path, model)
    return predicted_class
