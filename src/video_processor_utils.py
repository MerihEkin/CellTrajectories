import cv2
import pims
import numpy as np 


@pims.pipeline
def gray(image):
    """Decorator to convert RGB images to gray scale when loading with PIMS"""
    return np.dot(image[...,:3], [0.2989, 0.5870, 0.1140]).astype(np.uint8)


def filterOutsideCircle(img, circle, k):
    """
    Filter outside of the sampling chamber
    
    k : in videos where the cell escapes the sampling chamber from inlet and
    outlet channels you can set k to 1.1 or 1.2 to draw a larger circle 
    """

    mask = np.zeros_like(img)

    # Circle the frame and set outside of the circle to 0 
    cv2.circle(mask, (circle['x'], circle['y']), int(circle['radius'] * k), (255, 255, 255), thickness=-1)
    mask_gray = mask.astype(bool)
    result = img.copy()
    result[mask_gray == False] = 0

    return result


def spatialFiltering(img):
    """
    Apply spatial filtering if there is too much noise
    
    Note: the kernel sizes and thresholds are experimental values that are open to tuning
    """

    blurred_frame = cv2.medianBlur(img, ksize=15)   # blur frame
    sharpened_frame = cv2.addWeighted(img, 1.5, blurred_frame, -0.5, 0)
    
    _, thresh_frame = cv2.threshold(sharpened_frame, 125, 255, cv2.THRESH_BINARY) # threshold 
    
    kernel = np.ones((3,3), np.uint8)
    opened_frame = cv2.morphologyEx(thresh_frame, cv2.MORPH_OPEN, kernel) # apply opening

    mask_bool = opened_frame.astype(bool)
    result = np.zeros_like(img)

    # copy the pixels from the original image where the mask is true
    result[mask_bool] = img[mask_bool]

    return result
