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
    # Load image
    # image = cv2.imread(image_path)
    
    # Normalize image
    normalized_image = normalize_image(image)

    # Convert BGR image to RGB
    image_rgb = cv2.cvtColor(normalized_image, cv2.COLOR_BGR2RGB)

    # Initialize MediaPipe Pose
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(static_image_mode=True)

    # Process the normalized image with MediaPipe Pose
    results = pose.process(image_rgb)

    # Extract landmark coordinates
    if results.pose_landmarks:
        landmarks = np.array([(lm.x, lm.y) for lm in results.pose_landmarks.landmark])
        
        # Landmarks for head and toes
        head = landmarks[mp_pose.PoseLandmark.NOSE.value]
        left_ankle = landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value]
        right_ankle = landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value]
        left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
        right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
        left_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP.value]
        right_hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value]

        # Calculate distance between shoulders (assuming person is standing straight)
        shoulder_distance = np.linalg.norm(np.array(left_shoulder) - np.array(right_shoulder))
        
        # Calculate distance between hips (assuming person is standing straight)
        waist_distance = np.linalg.norm(np.array(left_hip) - np.array(right_hip))

        
        # Calculate the midpoint of the two toes
        midpoint_ankle = (left_ankle + right_ankle) / 2
        mid_hip = (left_hip + right_hip) / 2
        mid_shoulder = (left_shoulder + right_shoulder) / 2

        
        upperbody_length = np.linalg.norm(mid_hip - mid_shoulder)
        lowerbody_length = np.linalg.norm(mid_hip - midpoint_ankle)
        height = np.linalg.norm(np.array(head) - np.array(midpoint_ankle))
        top_pixel_distance = head[1] 
        
        return top_pixel_distance, head
    else:
        return None, None
# Initialize camera
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture frame.")
        break
    
    top_pix_distance, nose_pos = calculate_person_height(frame)
    
    # If no person detected, display "no person" in the video
    if top_pix_distance is None:
        cv2.putText(frame, "No person", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
        cv2.imshow('Frame', frame)
    else:
        # If top pixel distance is within the desired range, capture the image
        # min_desired_distance = 0.0825911462306976
        min_desired_distance = 0.0925911462306976
        max_desired_distance = 0.253641217947006
    
        if min_desired_distance <= top_pix_distance <= max_desired_distance:
            # Draw a circle to mark the nose position
            cv2.circle(frame, (int(nose_pos[0] * frame.shape[1]), int(nose_pos[1] * frame.shape[0])), 5, (0, 255, 0), -1)
            cv2.imwrite("captured_image.jpg", frame)
            print("Image captured!")
            break
    
        cv2.imshow('Frame', frame)
    
    # Exit loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release video capture and close windows
cap.release()
cv2.destroyAllWindows()