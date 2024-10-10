import numpy as np
import cv2 as cv
from PIL import Image

def reduce_noise(image: Image) -> Image:
    """
    Reduces noise in an image using a Gaussian, median and bilateral filter consecutively.

    Details of the algorithm can be found at https://docs.opencv.org/4.x/d4/d13/tutorial_py_filtering.html
 
    Args:
        image (Image): The image to reduce noise in.
    Returns:
        (Image): The image with reduced noise.
    """
    image = np.array(image)
    gaussian = cv.GaussianBlur(image, (3, 3), 0)
    median = cv.medianBlur(gaussian, 3)
    bilateral = cv.bilateralFilter(median, 5, 100, 100)

    return Image.fromarray(bilateral)