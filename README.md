# CV-Drone-101
All Notebooks and submissions during the course of project

## Description
### Hybrid Image Generation Assignment

#### Objective
The goal of this assignment is to create hybrid images by combining the low-frequency components of one image with the high-frequency components of another image using Python. You will learn how to manipulate and transform images in the frequency domain using libraries such as OpenCV, NumPy, and Matplotlib.

#### Step 1: Load and Display the Mask
Create and display a mask that will be used to filter the images in the frequency domain.

```python
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# Load images
pil_im = Image.open('/content/drive/MyDrive/Computer vision/funny-cat-4.jpg')
pil_im2 = Image.open('/content/drive/MyDrive/Computer vision/image00049.jpg')

# Create a circular mask
rows, cols = 256, 256
crow, ccol = rows // 2, cols // 2
radius = 30  # Adjust this parameter to control the cutoff frequency
mask = np.zeros((rows, cols), np.uint8)
cv2.circle(mask, (ccol, crow), radius, 1, -1)

# Create a rectangular mask
mask1 = np.zeros((256, 256), dtype=np.uint8)
cv2.rectangle(mask1, (112, 112), (144, 144), 1, -1)

# Show the mask using Matplotlib
plt.imshow(mask1, cmap='gray')
plt.show()
```
### Step 2:Define and Apply the Hybrid Image Function
Define a function hybrid that combines the low-frequency components of one image with the high-frequency components of another image. This function will read the images, resize them, convert them to the frequency domain, apply the mask, and then combine the filtered images.

```python
def hybrid(s1, s2):
    # Open images
    image1 = cv2.imread(s1, cv2.IMREAD_GRAYSCALE)
    image2 = cv2.imread(s2, cv2.IMREAD_GRAYSCALE)

    # Resize images
    image1 = cv2.resize(image1, (256, 256))
    image2 = cv2.resize(image2, (256, 256))

    # Convert images to frequency domain
    image1_transform = np.fft.fft2(image1)
    image2_transform = np.fft.fft2(image2)

    # Shift zero frequency component to the center
    image1_transform_shifted = np.fft.fftshift(image1_transform)
    image2_transform_shifted = np.fft.fftshift(image2_transform)

    # Create a rectangular mask
    rows, cols = 256, 256
    crow, ccol = rows // 2, cols // 2
    radius = 10
    mask = np.zeros((rows, cols), np.uint8)
    cv2.rectangle(mask, (112, 112), (144, 144), 1, -1)

    # Apply the mask
    im1_filtered = image1_transform_shifted * mask
    im2_filtered = image2_transform_shifted * mask

    # Inverse FFT to get the filtered images
    final1 = np.abs(np.fft.ifft2(np.fft.ifftshift(im1_filtered)))
    transit = np.abs(np.fft.ifft2(np.fft.ifftshift(im2_filtered)))
    final2 = image2 - transit
    np.log1p(final2)

    # Display the results
    plt.subplot(121), plt.imshow(image1, cmap='gray'), plt.title('Original Image1')
    plt.subplot(122), plt.imshow(final1, cmap='gray'), plt.title('LPF Image')
    plt.show()

    plt.subplot(121), plt.imshow(image2, cmap='gray'), plt.title('Original Image2')
    plt.subplot(122), plt.imshow(final2, cmap='gray'), plt.title('HPF Image')
    plt.show()

    plt.subplot(121), plt.imshow((final1 + final2) / 2, cmap='gray'), plt.title('Hybrid Image')
    plt.subplot(122), plt.imshow(mask, cmap='gray'), plt.title('The Filter')
    plt.show()

    return 0

# Apply the hybrid function
hybrid('/content/drive/MyDrive/Computer vision/funny-cat-4.jpg', '/content/drive/MyDrive/Computer vision/image00049.jpg')
```
