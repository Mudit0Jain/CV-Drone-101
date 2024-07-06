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

### Camera Calibration Using OpenCV in Google Colab
#### Objective:
The objective of this code is to calibrate a camera using images of a chessboard pattern, captured in a Google Colab environment.

#### Steps:
##### Define Parameters
* chessboardSize: Size of the chessboard pattern (6x6 in this case).
* frameSize: Size of the image frame (640x480 in this case).
* criteria: Criteria for the termination of the iterative process in corner refinement (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER with iterations set to 30 and epsilon to 0.001).
* objp: 3D points of the chessboard corners in real-world coordinates, initialized as a grid of zeros and then filled with coordinates scaled by the size of each chessboard square (size_of_cb_sq_mm = 20).

##### Initialize Arrays:

* objpoints: List to store 3D points of chessboard corners in real-world coordinates.
* imgpoints: List to store 2D points of chessboard corners in image coordinates.

##### Load and Process Images:

* Load images using glob.glob().
* For each image, convert it to grayscale (cv.cvtColor(img, cv.COLOR_BGR2GRAY)).
* Use cv.findChessboardCorners() to find corners of the chessboard pattern in the grayscale image.
* If corners are found (ret == True), refine corner locations using cv.cornerSubPix(), store the 3D object points (objp), and store the refined 2D image points (corners2).
* Visualize the detected corners using cv.drawChessboardCorners() and cv2_imshow().

##### Camera Calibration:

* Use cv.calibrateCamera() to calibrate the camera using the collected object points (objpoints) and image points (imgpoints).
* Retrieve the camera matrix (cameraMatrix), distortion coefficients (dist), rotation vectors (rvecs), and translation vectors (tvecs).

##### Display Results:
Print the calibrated cameraMatrix which contains intrinsic parameters of the camera.

# Example Code:
```python
import numpy as np
import cv2 as cv
import glob
from google.colab.patches import cv2_imshow

# Parameters
chessboardSize = (6, 6)
frameSize = (640, 480)
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
size_of_cb_sq_mm = 20

# Create grid of points in real-world coordinates
objp = np.zeros((chessboardSize[0] * chessboardSize[1], 3), np.float32)
objp[:,:2] = np.mgrid[0:chessboardSize[0], 0:chessboardSize[1]].T.reshape(-1, 2)
objp = objp * size_of_cb_sq_mm

# Arrays to store object points and image points from all images
objpoints = []  # 3d point in real world space
imgpoints = []  # 2d points in image plane

# Load images
link = '/content/drive/MyDrive/Computer vision/WhatsApp Image 2023-12-20 at 17.17.12_9605c69f.jpg'
images = glob.glob(link)

# Process each image
for image in images:
    img = cv.imread(image)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    # Find chessboard corners
    ret, corners = cv.findChessboardCorners(gray, chessboardSize, None)

    if ret == True:
        # Refine corner locations
        corners2 = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

        # Store object points and image points
        objpoints.append(objp)
        imgpoints.append(corners2)

        # Visualize corners
        cv.drawChessboardCorners(img, chessboardSize, corners2, ret)
        cv2_imshow(img)
        cv.waitKey(500)

# Calibrate the camera
ret, cameraMatrix, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, frameSize, None, None)

# Print the camera matrix
print("cameraMatrix:", cameraMatrix)
This script demonstrates how to perform camera calibration using OpenCV in Google Colab. It detects a chessboard pattern in images, refines the corner positions, and calculates the camera's intrinsic parameters such as the camera matrix (cameraMatrix) and distortion coefficients (dist).
```
