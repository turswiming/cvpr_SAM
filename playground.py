import cv2
import os

# Specify the directory
source = 'output_data_2d/'
destination = 'playground/'

# Set the JPEG quality
jpeg_quality = 50

# Iterate over all files in the directory
for filename in os.listdir(source):
    # Check if the file is an image
    if filename.endswith('.jpg') or filename.endswith('.png'):
        # Construct the full file path
        filepath = os.path.join(source, filename)

        # Load the image
        image = cv2.imread(filepath)

        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(image, (15, 15), 0)

        # Save the blurred image as a JPEG file with the specified quality
        cv2.imwrite(os.path.join(destination, os.path.splitext(filename)[0]) + '.jpg', blurred, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])