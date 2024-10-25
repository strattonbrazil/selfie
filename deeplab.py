import cv2
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub

# Load the DeepLab model from TensorFlow Hub
model_url = "https://tfhub.dev/google/segmentation/deeplabv3/1"
model = hub.load(model_url).signatures['serving_default']

# Function to perform segmentation
def run_inference_for_single_image(model, image):
    input_tensor = tf.convert_to_tensor(image)
    input_tensor = input_tensor[tf.newaxis, ...]  # Create a batch

    # Run inference
    output_dict = model(input_tensor)

    return output_dict

# Open the webcam
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Resize frame for the model
    input_size = (512, 512)
    image = cv2.resize(frame, input_size)
    image = image / 255.0  # Normalize to [0, 1]

    # Run inference
    output_dict = run_inference_for_single_image(model, image)

    # Get the segmentation mask
    mask = output_dict['segmentation_mask'][0].numpy()  # Modify this based on the output format

    # Resize the mask to the original frame size
    mask_resized = cv2.resize(mask, (frame.shape[1], frame.shape[0]))

    # Create a binary mask
    binary_mask = mask_resized > 0.5

    # Apply the mask to the original frame
    foreground = np.where(binary_mask[..., None], frame, (0, 0, 255))  # Replace background with red

    # Display the result
    cv2.imshow('Background Removed', foreground)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
