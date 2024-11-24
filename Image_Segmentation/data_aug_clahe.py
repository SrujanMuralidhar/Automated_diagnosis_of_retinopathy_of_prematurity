import os
import numpy as np
import cv2
from tqdm import tqdm
from albumentations import HorizontalFlip, VerticalFlip, Rotate
from skimage import img_as_float
from skimage.filters import gabor
from PIL import Image

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


""" Create a directory """
def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def load_data(path):
    train_x_dir = os.path.join(path, "train_images")
    train_y_dir = os.path.join(path, "train_masks")
    test_x_dir = os.path.join(path, "test_image")
    test_y_dir = os.path.join(path, "test_mask")

    train_x = sorted([os.path.join(train_x_dir, file) for file in os.listdir(train_x_dir)])
    train_y = sorted([os.path.join(train_y_dir, file) for file in os.listdir(train_y_dir)])

    test_x = sorted([os.path.join(test_x_dir, file) for file in os.listdir(test_x_dir)])
    test_y = sorted([os.path.join(test_y_dir, file) for file in os.listdir(test_y_dir)])

    return (train_x, train_y), (test_x, test_y)

def augment_data(images, masks, save_path, augment=True):
    size = (512, 512)

    for idx, (x, y) in tqdm(enumerate(zip(images, masks)), total=len(images)):
        """ Extracting the name """
        name = os.path.splitext(os.path.basename(x))[0]

        """ Reading image and mask """
        x = cv2.imread(x)
        y = cv2.imread(y, cv2.IMREAD_GRAYSCALE)

        if x is None or y is None:
            print(f"Error reading {x} or {y}. Skipping.")
            continue
        
        x = cv2.resize(x, size, interpolation=cv2.INTER_AREA)
        y = cv2.resize(y, size, interpolation=cv2.INTER_NEAREST)

        # CLAHE Green Channel
        # gc = x[:,:,1]
        # clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        # gc = clahe.apply(gc)
        # x[:,:,1] = gc
        # # x = cv2.equalizeHist(x)

        # Adaptive Sigmoid
        x = adaptive_sigmoid(x)

        # # Gabor Filter
        # x = gab(x)
        # x = refine_edges(x)


        x = np.array(x, dtype=np.uint8)
        
        if augment:
            aug = HorizontalFlip(p=1.0)
            augmented = aug(image=x, mask=y)
            x1 = augmented["image"]
            y1 = augmented["mask"]

            aug = VerticalFlip(p=1.0)
            augmented = aug(image=x, mask=y)
            x2 = augmented["image"]
            y2 = augmented["mask"]

            aug = Rotate(limit=45, p=1.0)
            augmented = aug(image=x, mask=y)
            x3 = augmented["image"]
            y3 = augmented["mask"]

            X = [x, x1, x2, x3]
            Y = [y, y1, y2, y3]

        else:
            X = [x]
            Y = [y]

        index = 0
        for i, m in zip(X, Y):
            image_path = os.path.join(save_path, "image", f"{name}_{index}.png")
            mask_path = os.path.join(save_path, "mask", f"{name}_{index}.png")

            cv2.imwrite(image_path, i)
            cv2.imwrite(mask_path, m)

            index += 1

if __name__ == "__main__":
    """ Seeding """
    np.random.seed(42)

    """ Load the data """
    data_path = "Data_BV"
    (train_x, train_y), (test_x, test_y) = load_data(data_path)

    print(f"Train: {len(train_x)} - {len(train_y)}")
    print(f"Test: {len(test_x)} - {len(test_y)}")

    """ Create directories to save the augmented data """
    create_dir("new_data/train/image/")
    create_dir("new_data/train/mask/")
    create_dir("new_data/test/image/")
    create_dir("new_data/test/mask/")

    """ Data augmentation """
    augment_data(train_x, train_y, "new_data/train/", augment=True)
    augment_data(test_x, test_y, "new_data/test/", augment=False)
