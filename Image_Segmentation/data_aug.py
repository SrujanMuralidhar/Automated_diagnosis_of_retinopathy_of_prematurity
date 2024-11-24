# import os
# import numpy as np
# from tqdm import tqdm
# from PIL import Image
# from albumentations import HorizontalFlip, VerticalFlip, Rotate
# import cv2

# """ Create a directory """
# def create_dir(path):
#     if not os.path.exists(path):
#         os.makedirs(path)

# def load_data(path):
#     train_x_dir = os.path.join(path, "train_images")
#     train_y_dir = os.path.join(path, "train_masks")
#     test_x_dir = os.path.join(path, "test_image")
#     test_y_dir = os.path.join(path, "test_mask")

#     train_x = sorted([os.path.join(train_x_dir, file) for file in os.listdir(train_x_dir)])
#     train_y = sorted([os.path.join(train_y_dir, file) for file in os.listdir(train_y_dir)])

#     test_x = sorted([os.path.join(test_x_dir, file) for file in os.listdir(test_x_dir)])
#     test_y = sorted([os.path.join(test_y_dir, file) for file in os.listdir(test_y_dir)])

#     return (train_x, train_y), (test_x, test_y)

# def apply_clahe(image):
#     """ Apply CLAHE on the green channel """
#     green_channel = image[:, :, 1]  # Extracting the green channel
#     clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
#     green_clahe = clahe.apply(green_channel)
#     return green_clahe

# def sigmoid_correction(image, k=10, x0=0.5):
#     # Normalize the image
#     normalized_img = image / 255.0
    
#     # Apply the sigmoid function
#     sigmoid_img = 1 / (1 + np.exp(-k * (normalized_img - x0)))
    
#     # Scale back to original range
#     corrected_img = (sigmoid_img * 255).astype(np.uint8)
    
#     return corrected_img

# def adaptive_sigmoid(image):
#     mask = np.zeros(image.shape[:2], dtype=np.uint8)
#     center = (int(image.shape[1] / 2), int(image.shape[0] / 2))
#     radius = image.shape[1]//2
#     cv2.circle(mask, center, radius, 255, -1)
#     hist = cv2.calcHist([image], [1], mask, [256], [0, 256])

# # Calculate cumulative distribution function (CDF)
#     cdf = hist.cumsum()

#     # Normalize CDF
#     cdf_normalized = cdf * hist.max() / cdf.max()

#     # Find the intensity level where CDF reaches 50% of the total pixel count
#     total_pixels = cdf[-1]
#     x_0 = np.searchsorted(cdf, total_pixels * 0.5)/255
#     k = 15

#     sig = sigmoid_correction(image,k,x_0)

#     return sig

# def augment_data(images, masks, save_path, augment=True):
#     size = (512, 512)

#     for idx, (x, y) in tqdm(enumerate(zip(images, masks)), total=len(images)):
#         """ Extracting the name """
#         name = os.path.splitext(os.path.basename(x))[0]

#         """ Reading image and mask """
#         x = Image.open(x).convert("RGB")
#         y = Image.open(y).convert("L")

#         x = np.array(x)
#         y = np.array(y)

#         # CLAHE Green Channel
#         # x = adaptive_sigmoid(x)
#         # gc = x[:,:,1]
#         # clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
#         # gc = clahe.apply(gc)
#         # x[:,:,1] = gc
#         # x = cv2.equalizeHist(x)
#         #Adaptive Sigmoid
       
#         if augment:
#             aug = HorizontalFlip(p=1.0)
#             augmented = aug(image=x, mask=y)
#             x1 = augmented["image"]
#             y1 = augmented["mask"]

#             aug = VerticalFlip(p=1.0)
#             augmented = aug(image=x, mask=y)
#             x2 = augmented["image"]
#             y2 = augmented["mask"]

#             aug = Rotate(limit=45, p=1.0)
#             augmented = aug(image=x, mask=y)
#             x3 = augmented["image"]
#             y3 = augmented["mask"]

#             X = [x, x1, x2, x3]
#             Y = [y, y1, y2, y3]

#         else:
#             X = [x]
#             Y = [y]

#         index = 0
#         for i, m in zip(X, Y):
#             i = Image.fromarray(i).resize(size, Image.BILINEAR)
#             m = Image.fromarray(m).resize(size, Image.NEAREST)

#             tmp_image_name = f"{name}_{index}.png"
#             tmp_mask_name = f"{name}_{index}.png"

#             image_path = os.path.join(save_path, "image", tmp_image_name)
#             mask_path = os.path.join(save_path, "mask", tmp_mask_name)

#             i.save(image_path)
#             m.save(mask_path)

#             index += 1

# if __name__ == "__main__":
#     """ Seeding """
#     np.random.seed(42)

#     """ Load the data """
#     data_path = "Data_BV"
#     (train_x, train_y), (test_x, test_y) = load_data(data_path)

#     print(f"Train: {len(train_x)} - {len(train_y)}")
#     print(f"Test: {len(test_x)} - {len(test_y)}")

#     """ Create directories to save the augmented data """
#     create_dir("new_data/train/image/")
#     create_dir("new_data/train/mask/")
#     create_dir("new_data/test/image/")
#     create_dir("new_data/test/mask/")

#     """ Data augmentation """
#     augment_data(train_x, train_y, "new_data/train/", augment=False)
#     augment_data(test_x, test_y, "new_data/test/", augment=False)


import os
import numpy as np
from tqdm import tqdm
from PIL import Image
from albumentations import HorizontalFlip, VerticalFlip, Rotate
import cv2

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

def apply_clahe(image):
    """ Apply CLAHE on the green channel """
    green_channel = image[:, :, 1]  # Extracting the green channel
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    green_clahe = clahe.apply(green_channel)
    return green_clahe

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
    radius = image.shape[1] // 2
    cv2.circle(mask, center, radius, 255, -1)
    hist = cv2.calcHist([image], [1], mask, [256], [0, 256])

    # Calculate cumulative distribution function (CDF)
    cdf = hist.cumsum()

    # Normalize CDF
    cdf_normalized = cdf * hist.max() / cdf.max()

    # Find the intensity level where CDF reaches 50% of the total pixel count
    total_pixels = cdf[-1]
    x_0 = np.searchsorted(cdf, total_pixels * 0.5) / 255
    k = 15

    sig = sigmoid_correction(image, k, x_0)

    return sig

def apply_hist_eq(image):
    """ Apply histogram equalization to each channel of the image """
    channels = cv2.split(image)  # Split the image into its color channels
    eq_channels = []
    for ch in channels:
        eq_channels.append(cv2.equalizeHist(ch))  # Apply histogram equalization to each channel
    
    eq_image = cv2.merge(eq_channels)  # Merge the equalized channels back
    return eq_image

def augment_data(images, masks, save_path, augment=True):
    size = (512, 512)

    for idx, (x, y) in tqdm(enumerate(zip(images, masks)), total=len(images)):
        """ Extracting the name """
        name = os.path.splitext(os.path.basename(x))[0]

        """ Reading image and mask """
        x = Image.open(x).convert("RGB")
        y = Image.open(y).convert("L")

        x = np.array(x)
        y = np.array(y)

        # Apply histogram equalization to each channel
        x = apply_hist_eq(x)
       
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
            i = Image.fromarray(i).resize(size, Image.BILINEAR)
            m = Image.fromarray(m).resize(size, Image.NEAREST)

            tmp_image_name = f"{name}_{index}.png"
            tmp_mask_name = f"{name}_{index}.png"

            image_path = os.path.join(save_path, "image", tmp_image_name)
            mask_path = os.path.join(save_path, "mask", tmp_mask_name)

            i.save(image_path)
            m.save(mask_path)

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
