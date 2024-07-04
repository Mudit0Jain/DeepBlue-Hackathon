import cv2
import numpy as np

# Read the video file
#video_path = r"C:\Users\Mudit\Downloads\VScode\cv\Deepblu\Untitled video - Made with Clipchamp (3).mp4"#Very faint gate
#video_path = r"C:\Users\Mudit\Downloads\VScode\cv\Deepblu\Untitled video - Made with Clipchamp (2).mp4"#short video
#video_path = r"C:\Users\Mudit\Downloads\VScode\cv\Deepblu\Untitled video - Made with Clipchamp (1).mp4"# best test caase best performance
video_path = r"C:\Users\Mudit\Downloads\VScode\cv\Deepblu\Untitled video - Made with Clipchamp.mp4" #ideal
cap = cv2.VideoCapture(video_path)

def harris_corner_detection(image, k=0.04, threshold=0.01):
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Compute the Harris corner response
    harris_response = cv2.cornerHarris(gray, blockSize=2, ksize=3, k=k)

    # Normalize the Harris corner response
    normalized_response = cv2.normalize(harris_response, None, 0, 255, cv2.NORM_MINMAX)

    # Threshold the response to identify corners
    corners = np.zeros_like(gray)
    corners[normalized_response > threshold * normalized_response.max()] = 255

    # Draw circles around the detected corners
    result = image.copy()
    result[corners > 0] = [0, 0, 255]  # Set the color of corners to red

    return result

def sharpen_lines_near_center(edge_map, center_threshold=0.5, sharpening_factor=2.0):
    # Find the center of the image
    center_x, center_y = edge_map.shape[1] // 2, edge_map.shape[0] // 2

    # Create a mask to identify regions near the center
    mask = np.zeros_like(edge_map)
    mask[int(center_y - center_y * center_threshold):int(center_y + center_y * center_threshold),
         int(center_x - center_x * center_threshold):int(center_x + center_x * center_threshold)] = 1

    # Apply a sharpening kernel to the regions near the center
    sharpening_kernel = np.array([[-1, -1, -1],
                                  [-1,  9, -1],
                                  [-1, -1, -1]])
    sharpened_lines = cv2.filter2D(edge_map, -1, sharpening_factor * sharpening_kernel)

    # Combine the sharpened lines with the original edge map using the mask
    result = edge_map.copy()
    result[mask > 0] = sharpened_lines[mask > 0]

    return result

def blur_congested_lines(canny_edge_map, blur_kernel_size=5):
    # Apply Gaussian blur to the Canny edge map
    blurred_edges = cv2.GaussianBlur(canny_edge_map, (blur_kernel_size, blur_kernel_size), 0)
    return blurred_edges
def vert_edge(image, threshold=100):
    # Convert the image to grayscale
    gray = image

    # Apply Sobel operator in the x and y directions
    # sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)

    # Compute the magnitude of the gradient
    # magnitude = np.sqrt(sobel_x**2 + sobel_y**2)

    # Normalize the magnitude to 8-bit for display
    magnitude_normalized = np.uint8(255 * sobel_y / np.max(sobel_y))

    # Apply a threshold to obtain binary edges
    edges = np.zeros_like(magnitude_normalized)
    edges[magnitude_normalized > threshold] = 255

    return edges.astype(image.dtype)



while cap.isOpened():
    ret, frame = cap.read()
    
    if not ret:
        break
    
    frame = cv2.resize(frame, (640, 480))
    blue = frame[:, :, 2]
    green = frame[:, :, 1]
    red = frame[:, :, 0]
    #combine images with different weights
    combined = cv2.addWeighted( blue, 0.1, green, 0.6, 0)
    combined = cv2.addWeighted( combined, 0.7, red, 0.3, 0)
    #display the combined image
    # cv2.imshow('combined', combined)
    
    NOT_BLUE = cv2.bitwise_not(blue)
    # cv2.imshow('NOT_BLUE', NOT_BLUE)    
    kernel = np.array([[-1,-1,-1,-1,-1],[-1,1,2,1,-1],[-1,2,4,2,-1],[-1,1,2,1,-1],[-1,-1,-1,-1,-1]])
    sharpened = cv2.filter2D(combined, -1, kernel)
    # cv2.imshow('sharpened', sharpened)
    DILATED = cv2.dilate(sharpened, kernel, iterations=1)
    # cv2.imshow('DILATED', DILATED)
    ERODED = cv2.erode(DILATED, kernel, iterations=2)
    # cv2.imshow('ERODED', ERODED)
    #display the original image
    cv2.imshow('original', frame)
    # cent_sharp = sharpen_lines_near_center(sharpened)
    # cv2.imshow('sharped', cent_sharp)
    # corners  = harris_corner_detection(sharpened)
    cannied = cv2.Canny(sharpened,70, 110)
    # cv2.imshow('cannied', cannied)
    # cv2.imshow(corners)
    cong_blurred = blur_congested_lines(cannied)
    # cv2.imshow('congblur', cong_blurred)

    vert_lines = vert_edge(cannied)
    cv2.imshow('verts', vert_lines)

    vert_lines150 = vert_edge(cannied, threshold=150)
    cv2.imshow('verts150', vert_lines150)

    vert_lines190 = vert_edge(cannied, threshold=250)
    cv2.imshow('verts190', vert_lines190)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
