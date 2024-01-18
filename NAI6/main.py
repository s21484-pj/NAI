import cv2
import numpy as np

# Define a video capture object
vid = cv2.VideoCapture(0)

# Load the face cascade classifier
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Check if the face cascade classifier is loaded
if face_cascade.empty():
    raise IOError("Unable to load the face cascade classifier.")

while True:
    # Capture the video frame by frame
    ret, frame = vid.read()
    result = frame.copy()

    # Change video to another color scale
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Detect faces
    face_rects = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.3, minNeighbors=5)

    # Draw circle on faces
    for (x, y, w, h) in face_rects:
        center_coordinates = x + w // 2, y + h // 2
        cv2.circle(result, center_coordinates, 10, (0, 0, 255), 20)

    # Define color range to find red
    lower = np.array([36, 25, 25])
    upper = np.array([70, 255, 255])

    # Create mask to show only red things
    mask = cv2.inRange(hsv, lower, upper)
    imask = mask > 0
    green = np.zeros_like(frame, np.uint8)
    green[imask] = frame[imask]

    # Find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Set a minimum size threshold for contours
    min_contour_size = 200  # Adjust this value according to your needs

    if contours:

        # Iterate through the contours and draw bounding boxes only for large contours
        for contour in contours:
            # Calculate bounding box parameters
            x, y, w, h = cv2.boundingRect(contour)

            # Check if the contour size exceeds the minimum threshold
            if w * h > min_contour_size:
                cv2.circle(result, (x + w // 2, y + h // 2), 10, (0, 255, 0), 20)

    # Show original video and the one with the mask
    cv2.imshow("Original Video", frame)
    cv2.imshow('Result', result)

    # Close the video if the ESC key is pressed
    if cv2.waitKey(1) == 27:  # Esc key
        break

# Release the video capture object
vid.release()
# Destroy all windows
cv2.destroyAllWindows()
