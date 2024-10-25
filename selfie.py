import cv2
import mediapipe as mp
import numpy as np

# Initialize MediaPipe Selfie Segmentation
mp_selfie_segmentation = mp.solutions.selfie_segmentation
selfie_segmentation = mp_selfie_segmentation.SelfieSegmentation(model_selection=0)

# Open the webcam
cap = cv2.VideoCapture(0)

# Set a custom background color (blue, in this case)
bg_color = (0, 0, 255)  # BGR format

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the image from BGR to RGB for MediaPipe processing
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the image to get the segmentation mask
    results = selfie_segmentation.process(rgb_frame)
    mask = results.segmentation_mask

    # Create a mask where pixel values > threshold are considered as the foreground
    condition = mask > 0.5  # You can adjust the threshold value if needed

    # Apply the mask to the original image
    foreground = np.where(condition[..., None], frame, bg_color).astype(np.uint8)

    # Display the result
    cv2.imshow('Background Removed', foreground)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
