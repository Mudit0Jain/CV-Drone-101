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
### Image Convolution and Sobel Edge Detection Assignment

#### Objective
The objective of this assignment is to implement image convolution and apply Sobel edge detection on a grayscale image using Python. This exercise will help you understand the basics of image processing and edge detection.

#### Step 1: Define the Convolution Function
Create a function `convolve` that takes an image and a kernel as inputs and returns the convolved image.

```python
import cv2
import numpy as np
import matplotlib.pyplot as plt

def convolve(image, kernel):
    # Get image dimensions
    rows, cols = image.shape

    # Get kernel dimensions
    krows, kcols = kernel.shape

    # Calculate the padding size
    pad_rows = krows // 2
    pad_cols = kcols // 2

    # Pad the image
    padded_image = np.pad(image, ((pad_rows, pad_rows), (pad_cols, pad_cols)), mode='edge')

    # Initialize the result image
    result = np.zeros_like(image, dtype=np.float64)

    # Perform convolution
    for i in range(rows):
        for j in range(cols):
            result[i, j] = np.sum(padded_image[i:i+krows, j:j+kcols] * kernel)

    return result
```
#### Step 2: Define the Sobel Edge Detection Function
Create a function Sobel that reads an image, applies the Sobel operator in both x and y directions, calculates the gradient magnitude, and displays the results.

```python
def Sobel(s1):
    image = cv2.imread(s1, cv2.IMREAD_GRAYSCALE)

    # Define Sobel kernels
    sobelk_x = np.array([[-1, 0, 1],
                         [-3, 0, 3],
                         [-1, 0, 1]])
    sobelk_y = np.array([[-1, -3, -1],
                         [ 0,  0,  0],
                         [ 1,  3,  1]])

    # Apply convolution with Sobel kernels
    sobel_x = convolve(image.astype(np.float64), sobelk_x)
    sobel_y = convolve(image.astype(np.float64), sobelk_y)

    # Calculate gradient magnitude
    magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
    normalized = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)

    # Display the results
    plt.subplot(2, 2, 1), plt.imshow(image, cmap='gray'), plt.title('Original Image')
    plt.subplot(2, 2, 2), plt.imshow(sobel_x, cmap='gray'), plt.title('Sobel X')
    plt.subplot(2, 2, 3), plt.imshow(sobel_y, cmap='gray'), plt.title('Sobel Y')
    plt.subplot(2, 2, 4), plt.imshow(magnitude, cmap='gray'), plt.title('Gradient Magnitude')
    plt.show()

    return 0
```
#### Step 3: Run the Sobel Edge Detection
Call the Sobel function with the path to your image to apply edge detection and visualize the results.
