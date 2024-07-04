import cv2
import numpy as np

# Function for unsharp mask
def unsharp_mask(image, sigma=1.0, strength=1.5):
    blurred = cv2.bilateralFilter(image, 2, 75, 75)
    sharpened = cv2.addWeighted(image, 1.0 + strength, blurred, -strength, 0)
    return sharpened

# Read the video file

# Read the video file
#video_path = r"C:\Users\Mudit\Downloads\VScode\cv\Deepblu\Untitled video - Made with Clipchamp (3).mp4"#Very faint gate
#video_path = r"C:\Users\Mudit\Downloads\VScode\cv\Deepblu\Untitled video - Made with Clipchamp (2).mp4"#short video
video_path = r"C:\Users\Mudit\Downloads\VScode\cv\Deepblu\Untitled video - Made with Clipchamp (1).mp4"# best test caase best performance
#video_path = r"C:\Users\Mudit\Downloads\VScode\cv\Deepblu\Untitled video - Made with Clipchamp.mp4" #ideal
cap = cv2.VideoCapture(video_path)
# width = int(cap.get(3))
# height = int(cap.get(4))
# output_path = "C:/Users/Mudit/Downloads/VScode/cv"
# fourcc = cv2.VideoWriter_fourcc(*'XVID')
# output_video = cv2.VideoWriter(output_path, fourcc, 20.0, (width, height))

# Sharpen the combined image
kernel_5x5 = np.array(
    [
        [-1, -1, -1, -1, -1],
        [-1, 1, 2, 1, -1],
        [-1, 2, 5, 2, -1],
        [-1, 1, 2, 1, -1],
        [-1, -1, -1, -1, -1],
    ]
)

kernel_7x7 = np.array(
    [
        [-1, -1, -1, -1, -1, -1, -1],
        [-1, -1, -1, -1, -1, -1, -1],
        [-1, -1, -1, 8, -1, -1, -1],
        [-1, -1, 8, 13.5, 8, -1, -1],
        [-1, -1, -1, 8, -1, -1, -1],
        [-1, -1, -1, -1, -1, -1, -1],
        [-1, -1, -1, -1, -1, -1, -1]
    ]
)

kernel_7x7_hor = np.array(
    [
        [-1, -1, -1, 6, -1, -1, -1],
        [-1, -1, -1, 6, -1, -1, -1],
        [-1, -1, -1, 6, -1, -1, -1],
        [-1, -1, -1, 6, -1, -1, -1],
        [-1, -1, -1, 6, -1, -1, -1],
        [-1, -1, -1, 6, -1, -1, -1],
        [-1, -1, -1, 6, -1, -1, -1]
    ]
)

kernel_7x7_hor_thick = np.array(
    [
        [-1, -1, 4/3, 4/3, 4/3, -1, -1],
        [-1, -1, 4/3, 4/3, 4/3, -1, -1],
        [-1, -1, 4/3, 4/3, 4/3, -1, -1],
        [-1, -1, 4/3, 4/3, 4/3, -1, -1],
        [-1, -1, 4/3, 4/3, 4/3, -1, -1],
        [-1, -1, 4/3, 4/3, 4/3, -1, -1],
        [-1, -1, 4/3, 4/3, 4/3, -1, -1]
    ]
)

kernel_7x7_corners = np.array(
    [
        [-1, -1, -1, -1, -1, -1, -1],
        [-1, -1, -1, -1, -1, -1, -1],
        [-1, -1, 4/3, 4/3, 4/3, 4/3, 4/3],
        [-1, -1, 4/3, 4/3, 4/3, 4/3, 4/3],
        [-1, -1, 4/3, 4/3, 4/3, 4/3, 4/3],
        [-1, -1, 4/3, 4/3, 4/3, -1, -1],
        [-1, -1, 4/3, 4/3, 4/3, -1, -1]
    ]
)

while cap.isOpened():
    ret, frame = cap.read()

    if not ret:
        break

    frame = cv2.resize(frame, (640, 480))
    blue = frame[:, :, 2]
    green = frame[:, :, 1]
    red = frame[:, :, 0]

    # Combine images with different weights
    combined = cv2.addWeighted((blue), 0.05, green, 0.66, 0)
    combined = cv2.addWeighted(combined, 0.7, red, 0.52, 0)
    sharpened = cv2.filter2D(combined, -1, kernel_7x7)
    sharpened_hor= cv2.filter2D(combined, -1, kernel_7x7_hor)
    sharpened_hor_thick = cv2.filter2D(combined, -1 , kernel_7x7_hor_thick)
    sharpened_corners = cv2.filter2D(combined, -1, kernel_7x7_corners)
    sharpened = cv2.bilateralFilter(sharpened_hor_thick, 6, 80, 80)
    sharpened = cv2.bitwise_not(sharpened_hor_thick)
    ret, thresh = cv2.threshold(sharpened_hor_thick, 110, 255, cv2.THRESH_BINARY)
    kernel = np.ones((5 ,5 ), np.uint8)
    edges = cv2.Canny(thresh, 240, 245)
    cv2.imshow("edges", edges)
    cv2.imshow("hor_sharp", sharpened_hor_thick)
    edges = cv2.dilate(edges, kernel, iterations=1)
    # Probabilistic Hough Line Transform
    linesP = cv2.HoughLinesP(edges, 5, np.pi / 180, 50, 150, 40)
    cv2.imshow("thresh", thresh)
    # cv2.imshow("sharpened", sharpened)  
    # Detect and draw parallel lines
    if linesP is not None:
        cdstP = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)
        parallel_lines = []
        for line in linesP:
            x1, y1, x2, y2 = line[0]
            angle = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi
            if 60 < angle < 110 or -110 < angle < -70:  # Adjust angle range for parallel lines
                parallel_lines.append((x1, y1, x2, y2))
                cv2.line(cdstP, (x1, y1), (x2, y2), (0, 0, 255), 3, cv2.LINE_AA)

        # Draw bounding box around two longest parallel lines
        if len(parallel_lines) >= 2:
            # Sort by line length
            parallel_lines.sort(key=lambda l: ((l[2] - l[0]) ** 2 + (l[3] - l[1]) ** 2) ** 0.5, reverse=True)

            x1_1, y1_1, x2_1, y2_1 = parallel_lines[0]
            x1_2, y1_2, x2_2, y2_2 = parallel_lines[1]

            cv2.line(cdstP, (x1_1, y1_1), (x2_1, y2_1), (0, 255, 0), 2)
            cv2.line(cdstP, (x1_2, y1_2), (x2_2, y2_2), (0, 255, 0), 2)

            x_min = min(x1_1, x2_1, x1_2, x2_2)
            x_max = max(x1_1, x2_1, x1_2, x2_2)
            y_min = min(y1_1, y2_1, y1_2, y2_2)
            y_max = max(y1_1, y2_1, y1_2, y2_2)

            centre = ((x_max+x_min)/2, (y_max+y_min)/2)
            radius = 10

            #if ((y_max-y_min)/(x_max-x_min))<1:
                #cv2.circle(frame, centre, radius, (0,255,0), -1)
                #cv2.circle(cdstP, centre, radius, (0,255,0), -1)
            cv2.rectangle(cdstP, (x_min, y_min), (x_max, y_max), (255, 0, 0), 2)
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (90, 0, 0), 2)

        cv2.imshow("Detected Lines (in red) - Probabilistic Line Transform", cdstP)
        cv2.imshow("SAXX", frame)
        cv2.imshow("sharp corners", sharpened_corners)
        #cv2.waitKey(0)
        # output_video.write(frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()