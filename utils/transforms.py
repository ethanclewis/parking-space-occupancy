import torch
import torchvision.transforms.v2 as T
import torchvision.transforms.functional as TF
from utils import roi_analysis as ra
import random, numpy as np

# PREPROCESSING
def preprocess(image, res=None):
    """
    Takes an input image, performs optional resizing, converts it to a float tensor, 
    normalizes it, and returns the preprocessed image tensor ready for consumption by a neural network model. 
    This preprocessing ensures that the input data meets the requirements of the model and helps improve the 
    model's performance during training or inference.
    """
    # resize image to model input size
    if res is not None:
        image = TF.resize(image, res)

    # convert image to float
    image = image.to(torch.float32) / 255

    # normalize image to default/ standard torchvision values
    image = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(image)

    return image

# ROTATION HELPER FUNCTION
def random_image_rotation(image, points, max_angle=30.0):
    """
    Function exists to supplement augment()
    Randomly rotates an image and its annotations by [-max_angle, max_angle].
    Runs *much* faster on GPU than CPU, so try to avoid using on CPU.
    """
    device = image.device
    
    # check that the points are within [0, 1] (Achieved previously in preprocess())
    assert points.min() >= 0, points.min()
    assert points.max() <= 1, points.max()

    # generate random rotation angle in range [-max_angle, max_angle]
    angle_deg = (2*torch.rand(1).item() - 1) * max_angle
    
    # rotate the image and note the change in resolutions
    _, H1, W1 = image.shape
    image = TF.rotate(image, angle_deg, expand=True)
    _, H2, W2 = image.shape
    
    # create rotation matrix
    angle_rad = torch.tensor((angle_deg / 180.0) * 3.141592)
    R = torch.tensor([[torch.cos(angle_rad), -torch.sin(angle_rad)],
                      [torch.sin(angle_rad),  torch.cos(angle_rad)]], dtype=torch.float, device=device)
    
    # move points to an absolute cooridnate system with [0, 0] as the center of the image
    points = points.clone()
    points -= 0.5
    points[..., 0] *= (W1 - 1)
    points[..., 1] *= (H1 - 1)
    
    # rotate the points
    points = points @ R
    
    # move points back to the relative coordinate system
    points[..., 0] /= (W2 - 1)
    points[..., 1] /= (H2 - 1)
    points += 0.5
    
    # check that the points remain within [0, 1]
    assert points.min() >= 0, points.min()
    assert points.max() <= 1, points.max()
    
    return image, points

# AUGMENTATION FUNCTIONS

# Baseline
def augment(image, rois):
    """
    Applies rotation, color jitter, and a flip.
    Runs *much* faster on GPU than CPU, so try to avoid using on CPU.
    """
    # random horizontal flip
    if torch.rand(1).item() > 0.5:
        image = TF.hflip(image)
        rois = rois.clone()
        rois[:, :, 0] = 1 - rois[:, :, 0] # UPDATES BOUNDING BOX COORDINATES

    # color jitter
    image = T.ColorJitter(brightness=0.3, contrast=0.4, saturation=0.8, hue=0.1)(image)

    # random rotation
    image, rois = random_image_rotation(image, rois)
    
    return image, rois



# RANDOM ERASING

# Random Erasing
def random_erase(image, rois):
    """
    Random Erase with default parameters
    """
    # random horizontal flip
    if torch.rand(1).item() > 0.5:
        image = TF.hflip(image)
        rois = rois.clone()
        rois[:, :, 0] = 1 - rois[:, :, 0]

    # color jitter
    image = T.ColorJitter(brightness=0.3, contrast=0.4, saturation=0.8, hue=0.1)(image) 

    # ERASE
    image = T.RandomErasing(p=1.0)(image)

    # random rotation
    image, rois = random_image_rotation(image, rois)

    return image, rois

# Proportional Erasing
def proportional_erase(image, rois, scale=(0.01, 0.02), ratio=(0.8529840538857099, 1.9837538785910385)):
    """
    Random Erase with p=1.0

    Ratio parameter range determined by average dimensions of ROIs in image (roi_analysis script and notebook)

    Scale parameter range set to 1-2% of image
    """

    # random horizontal flip
    if torch.rand(1).item() > 0.5:
        image = TF.hflip(image)
        rois = rois.clone()
        rois[:, :, 0] = 1 - rois[:, :, 0]

    # color jitter
    image = T.ColorJitter(brightness=0.3, contrast=0.4, saturation=0.8, hue=0.1)(image) 

    # ERASE
    image = T.RandomErasing(p=1.0, scale=scale, ratio=ratio)(image)

    # random rotation
    image, rois = random_image_rotation(image, rois)

    return image, rois



# SPECKLE NOISE

# Speckle Noise (Before color jitter)
def speckle_noise_before(image, rois, var=1.0):
    """
    Apply Speckle Noise (var=1.0)
    """
    # random horizontal flip
    if torch.rand(1).item() > 0.5:
        image = TF.hflip(image)
        rois = rois.clone()
        rois[:, :, 0] = 1 - rois[:, :, 0] # UPDATES BOUNDING BOX COORDINATES

    # SPECKLE
    noise = torch.randn(image.size(), device=image.device) * var
    image = image + image * noise

    # color jitter
    image = T.ColorJitter(brightness=0.3, contrast=0.4, saturation=0.8, hue=0.1)(image)

    # random rotation
    image, rois = random_image_rotation(image, rois)

    return image, rois

# Speckle Noise (After color jitter)
def speckle_noise_after(image, rois, var=1.0):
    """
    Apply Speckle Noise (var=1.0)
    """
    # random horizontal flip
    if torch.rand(1).item() > 0.5:
        image = TF.hflip(image)
        rois = rois.clone()
        rois[:, :, 0] = 1 - rois[:, :, 0] # UPDATES BOUNDING BOX COORDINATES

    # color jitter
    image = T.ColorJitter(brightness=0.3, contrast=0.4, saturation=0.8, hue=0.1)(image)

    # SPECKLE
    noise = torch.randn(image.size(), device=image.device) * var
    image = image + image * noise

    # random rotation
    image, rois = random_image_rotation(image, rois)

    return image, rois



# SPECKLE ERASE

# Random Speckle Erase (Before color jitter)
def random_speckle_erase_before(image, rois, var=1.0):
    """
    Apply Speckle Noise to a randomly selected and sized region of the input image.
    Injects noise BEFORE color jitter.
    """
    # random horizontal flip
    if torch.rand(1).item() > 0.5:
        image = TF.hflip(image)
        rois = rois.clone()
        rois[:, :, 0] = 1 - rois[:, :, 0]

    # Select a random region for applying speckle noise
    height, width = image.shape[1], image.shape[2]
    x1 = random.randint(0, width - 1)
    y1 = random.randint(0, height - 1)
    x2 = random.randint(x1, width)
    y2 = random.randint(y1, height)

    # Create a mask for the selected region
    mask = torch.zeros_like(image)
    mask[:, y1:y2, x1:x2] = 1

    # Generate speckle noise
    noise = torch.randn(image[:, y1:y2, x1:x2].shape, device=image.device) * var

    # Apply speckle noise to the selected region
    image[:, y1:y2, x1:x2] = image[:, y1:y2, x1:x2] + image[:, y1:y2, x1:x2] * noise

    # color jitter
    image = T.ColorJitter(brightness=0.3, contrast=0.4, saturation=0.8, hue=0.1)(image)

    # random rotation
    image, rois = random_image_rotation(image, rois)

    return image, rois

# Random Speckle Erase (After color jitter)
def random_speckle_erase_after(image, rois, var=1.0):
    """
    Apply Speckle Noise to a randomly selected and sized region of the input image.
    Injects noise AFTER color jitter.
    """
    # random horizontal flip
    if torch.rand(1).item() > 0.5:
        image = TF.hflip(image)
        rois = rois.clone()
        rois[:, :, 0] = 1 - rois[:, :, 0]

    # Select a random region for applying speckle noise
    height, width = image.shape[1], image.shape[2]
    x1 = random.randint(0, width - 1)
    y1 = random.randint(0, height - 1)
    x2 = random.randint(x1, width)
    y2 = random.randint(y1, height)

    # Create a mask for the selected region
    mask = torch.zeros_like(image)
    mask[:, y1:y2, x1:x2] = 1

    # Generate speckle noise
    noise = torch.randn(image[:, y1:y2, x1:x2].shape, device=image.device) * var

    # color jitter
    image = T.ColorJitter(brightness=0.3, contrast=0.4, saturation=0.8, hue=0.1)(image)

    # Apply speckle noise to the selected region
    image[:, y1:y2, x1:x2] = image[:, y1:y2, x1:x2] + image[:, y1:y2, x1:x2] * noise

    # random rotation
    image, rois = random_image_rotation(image, rois)

    return image, rois

# Proportional Speckle Erase (Before color jitter)
def proportional_speckle_erase_before(image, rois, var=0.9, scale=(0.01, 0.02), ratio=(0.8529840538857099, 1.9837538785910385)):
    """
    Apply Speckle Noise to a randomly selected, but proportionally sized (to RoIs), region of the input image.
    Injects noise BEFORE color jitter.
    """
    # random horizontal flip
    if torch.rand(1).item() > 0.5:
        image = TF.hflip(image)
        rois = rois.clone()
        rois[:, :, 0] = 1 - rois[:, :, 0]

    # Calculate the size of the speckle noise region based on the scale parameter
    height, width = image.shape[1], image.shape[2]
    image_area = height * width
    noise_area = random.uniform(*scale) * image_area

    # Select aspect ratio
    # Range determined by average ROI dimensions across data set (roi_analysis.ipynb)
    noise_aspect_ratio = random.uniform(*ratio)

    # Calculate the dimensions of the noise region
    noise_height = int(np.sqrt(noise_area / noise_aspect_ratio))
    noise_width = int(noise_aspect_ratio * noise_height)

    # Ensure the noise region is not larger than the image
    noise_height = min(noise_height, height)
    noise_width = min(noise_width, width)

    # Choose a random position for the noise region
    x1 = random.randint(0, width - noise_width)
    y1 = random.randint(0, height - noise_height)
    x2 = x1 + noise_width
    y2 = y1 + noise_height

    # Generate speckle noise
    noise = torch.randn((image.size(0), noise_height, noise_width), device=image.device) * var

    # Apply speckle noise to the selected region
    image[:, y1:y2, x1:x2] = image[:, y1:y2, x1:x2] + image[:, y1:y2, x1:x2] * noise

    # color jitter
    image = T.ColorJitter(brightness=0.3, contrast=0.4, saturation=0.8, hue=0.1)(image)

    # random rotation
    image, rois = random_image_rotation(image, rois)

    return image, rois

# Proportional Speckle Erase (After color jitter)
def proportional_speckle_erase_after(image, rois, var=0.9, scale=(0.01, 0.02), ratio=(0.8529840538857099, 1.9837538785910385)):
    """
    Apply Speckle Noise to a randomly selected, but proportionally sized (to RoIs), region of the input image.
    Injects noise AFTER color jitter.
    """
    # random horizontal flip
    if torch.rand(1).item() > 0.5:
        image = TF.hflip(image)
        rois = rois.clone()
        rois[:, :, 0] = 1 - rois[:, :, 0]

    # Calculate the size of the speckle noise region based on the scale parameter
    height, width = image.shape[1], image.shape[2]
    image_area = height * width
    noise_area = random.uniform(*scale) * image_area

    # Select aspect ratio
    # Range determined by average ROI dimensions across data set (roi_analysis.ipynb)
    noise_aspect_ratio = random.uniform(*ratio)

    # Calculate the dimensions of the noise region
    noise_height = int(np.sqrt(noise_area / noise_aspect_ratio))
    noise_width = int(noise_aspect_ratio * noise_height)

    # Ensure the noise region is not larger than the image
    noise_height = min(noise_height, height)
    noise_width = min(noise_width, width)

    # Choose a random position for the noise region
    x1 = random.randint(0, width - noise_width)
    y1 = random.randint(0, height - noise_height)
    x2 = x1 + noise_width
    y2 = y1 + noise_height

    # Generate speckle noise
    noise = torch.randn((image.size(0), noise_height, noise_width), device=image.device) * var

    # color jitter
    image = T.ColorJitter(brightness=0.3, contrast=0.4, saturation=0.8, hue=0.1)(image)

    # Apply speckle noise to the selected region
    image[:, y1:y2, x1:x2] = image[:, y1:y2, x1:x2] + image[:, y1:y2, x1:x2] * noise

    # random rotation
    image, rois = random_image_rotation(image, rois)

    return image, rois
