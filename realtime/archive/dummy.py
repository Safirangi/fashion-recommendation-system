import cv2
import mediapipe as mp
import numpy as np

def normalize_image(image, target_width=640, target_height=None):
    # Calculate aspect ratio of original image
    height, width, _ = image.shape
    aspect_ratio = width / height
    
    if target_height is None:
        # Calculate height based on target width and aspect ratio
        target_height = int(target_width / aspect_ratio)
    
    # Resize the image to target width and height while maintaining aspect ratio
    normalized_image = cv2.resize(image, (target_width, target_height))
    return normalized_image

def calculate_person_height(image):
    # Convert BGR image to RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Initialize MediaPipe Pose
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose( min_detection_confidence=0.5, min_tracking_confidence=0.5)

    # Process the image with MediaPipe Pose
    results = pose.process(image_rgb)

    # Extract landmark coordinates
    landmarks = []
    if results.pose_landmarks:
        for lm in results.pose_landmarks.landmark:
            landmarks.append((lm.x, lm.y))
    
    return landmarks

# Initialize camera
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture frame.")
        break
    
    landmarks = calculate_person_height(frame)
    
    if landmarks:
        # Draw landmarks on the frame
        for landmark in landmarks:
            x = int(landmark[0] * frame.shape[1])
            y = int(landmark[1] * frame.shape[0])
            cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)

        # Display the frame
    
        # Check if any landmark is visible
        any_landmark_visible = any(landmarks)
        
        # If no landmark is visible, display "No person"
        if not any_landmark_visible:
            cv2.putText(frame, "No person", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
            # cv2.imshow('Frame', frame)
    
        # If all landmarks are visible, capture the image
        elif all(landmarks):
            # Capture the image
            # cv2.imwrite("captured_image.jpg", frame)
            cv2.putText(frame, "all landmarks found", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
            # print("Image captured!")
            # break
    
    cv2.imshow('Frame', frame)
    # Exit loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release video capture and close windows
cap.release()
cv2.destroyAllWindows()
