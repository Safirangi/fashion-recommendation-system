import cv2
import numpy as np
import mediapipe as mp
import pandas as pd
import os
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

def calculate_person_height(image_path):
    # Load image
    image = cv2.imread(image_path)
    
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
        
        return height,shoulder_distance,waist_distance,top_pixel_distance,upperbody_length,lowerbody_length


if __name__=="__main__":

    # # Read CSV file
    data = pd.read_csv("/home/saiganesh.s/ML/body_measurements/updated_csv_file51.csv")

    # # Folder containing images
    image_folder = "/home/saiganesh.s/ML/body_measurements/Dataset"

    # Initialize MediaPipe Pose
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(static_image_mode=True)

    # Initialize lists to store measurements
    pix_heights = []
    pix_shoulder_distances = []
    pix_waist_distances = []
    top_pix_distances=[]
    upperbody_distances=[]
    lowerbody_distances=[]

    # Calculate pixel height, shoulder distance, and waist distance for each image and append to respective lists
    for index, row in data.iterrows():
        image_path = os.path.join(image_folder, row['Filename'])
        
        if os.path.exists(image_path):
            height, shoulder, waist ,top_pix_distance,upperbody_distance,lowerbody_distance= calculate_person_height(image_path)
            if upperbody_distance is not None and lowerbody_distance is not None :
                # pix_heights.append(height)
                # pix_shoulder_distances.append(shoulder)
                # pix_waist_distances.append(waist)
                # top_pix_distances.append(top_pix_distance)
                upperbody_distances.append(upperbody_distance)
                lowerbody_distances.append(lowerbody_distance)
            else:
                # Append None if any of the measurements couldn't be calculated
                # pix_heights.append(None)
                # pix_shoulder_distances.append(None)
                # pix_waist_distances.append(None)
                # top_pix_distances.append(None)
                upperbody_distances.append(None)
                lowerbody_distances.append(None)
        else:
            # Append None if the image doesn't exist
            # pix_heights.append(None)
            # pix_shoulder_distances.append(None)
            # pix_waist_distances.append(None)
            # top_pix_distances.append(None)
            upperbody_distances.append(None)
            lowerbody_distances.append(None)
        
    # Add the lists of measurements to the DataFrame
    # data['pix_height'] = pix_heights
    # data['pix_shoulder_distance'] = pix_shoulder_distances
    # data['pix_waist_distance'] = pix_waist_distances
    # data['top_pix_distance']=top_pix_distances
    data['upperbody_distance']=upperbody_distances
    data['lowerbody_distance']=lowerbody_distances
    # Save updated DataFrame to CSV
    data.to_csv("updated_csv_file5.csv", index=False)
