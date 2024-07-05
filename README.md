# Gate Detection Using OpenCV
This project utilizes OpenCV to detect gates in a video. The script applies several image processing techniques such as sharpening, edge detection, and Hough Line Transformation to identify and highlight gates. It processes each frame of the video to detect parallel lines and draw bounding boxes around them.

## Requirements
* Python 3.x
* OpenCV
* NumPy

## Usage
1. Set the video path
2. Run the script

## Code Description
1. Import necessary libraries
2. Set the video path and initialize the video capture
3. Define kernels for image sharpening
4. Process each frame of the video
5. Combine color channels with different weights
6. Sharpen and threshold the combined image
7. Apply edge detection and Hough Line Transform
8. Detect and draw parallel lines
9. Draw bounding box around two longest parallel lines and track the center
10. Display results and handle exit

## Test Cases 

## Results

## Conclusion
This project demonstrates a method for detecting gates in videos using OpenCV. By combining various image processing techniques, the script effectively highlights gates and tracks their positions frame by frame.
