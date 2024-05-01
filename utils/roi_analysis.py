"""
Calculates the average dimensions (and corresponding standard errors) of an RoI in a given input image/ dataset.
Assists scale and ratio params of proportional_speckle_erase()
Average RoI proportions (and corresponding standard errors) reported in roi_analysis.ipynb
"""

import torch

def calculate_avg_std_roi_dimensions(rois: torch.Tensor):
    """
    Calculate the average (width, height) of the ROIs in a single image, 
    as well as standard deviations.
    """
    # Compute width and height of each ROI
    w = torch.amax(rois[:, :, 0], 1) - torch.amin(rois[:, :, 0], 1)
    h = torch.amax(rois[:, :, 1], 1) - torch.amin(rois[:, :, 1], 1)
    
    # Compute average width, height
    avg_width = torch.mean(w)
    avg_height = torch.mean(h)
    
    # Compute standard deviation of width, height
    std_width = torch.std(w)
    std_height = torch.std(h)
    
    return avg_width.item(), avg_height.item(), std_width.item(), std_height.item()

def calculate_avg_std_dimensions_for_dataset(dataset):
    """
    Calculate the average (width, height) of the ROIs in an entire dataset, 
    as well as standard deviations.
    """
    avg_widths = []
    avg_heights = []
    std_widths = []
    std_heights = []
    
    for image_batch, rois_batch, labels_batch in dataset:
        for image, rois, labels in zip(image_batch, rois_batch, labels_batch):
            avg_width, avg_height, std_width, std_height = calculate_avg_std_roi_dimensions(rois)
            avg_widths.append(avg_width)
            avg_heights.append(avg_height)
            std_widths.append(std_width)
            std_heights.append(std_height)
    
    # Calculate overall averages
    overall_avg_width = sum(avg_widths) / len(avg_widths)
    overall_avg_height = sum(avg_heights) / len(avg_heights)
    
    # Calculate overall standard deviations
    overall_std_width = sum(std_widths) / len(std_widths)
    overall_std_height = sum(std_heights) / len(std_heights)
    
    return overall_avg_width, overall_avg_height, overall_std_width, overall_std_height



def calculate_avg_roi_area_proportion(image, rois):
    """
    NOT CURRENTLY IMPLEMENTED
    Calculate the average proportion of a single ROI area relative to 
    the area of the entire input image.
    """
    # Get image width and height from the first ROI
    image_width = image.shape[2]
    image_height = image.shape[1]

    # Compute width and height of each ROI
    w = torch.amax(rois[:, :, 0], 1) - torch.amin(rois[:, :, 0], 1)
    h = torch.amax(rois[:, :, 1], 1) - torch.amin(rois[:, :, 1], 1)
    
    # Compute area of each ROI
    areas = w * h
    
    # Compute proportion of ROI area to image area
    proportions = areas / (image_width * image_height)
    
    # Compute average and standard deviation of the proportions
    avg_proportion = torch.mean(proportions)
    std_proportion = torch.std(proportions)
    
    return avg_proportion.item(), std_proportion.item()