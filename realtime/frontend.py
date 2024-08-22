import streamlit as st
import cv2
import numpy as np
from PIL import Image
import mediapipe as mp
import os
from feature_extraction_mp import calculate_person_height
from white_bg_no_crop import get_clothes
from test_rs import extract_features, load_rs_resnet_model,load_rs_knn_model,load_rs_weights
from body_measurements import get_body_measurements_models
from size_chart import male_size_chart,female_size_chart

# Global variable to control video capture
global video_capture_running
video_capture_running = False

def main():
    global video_capture_running
    mp_drawing = mp.solutions.drawing_utils
    mp_pose = mp.solutions.pose
    
    pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
    cnt=0
    captureImg=False
    # Initialize video capture
    cap = cv2.VideoCapture(2)
    
    while cap.isOpened() and video_capture_running:
        ret, frame = cap.read()
        # Rotate the frame 90 degrees anti-clockwise
        frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)

        if(captureImg==True):
            cv2.imwrite("captured_image.jpg", frame)
            break
            
        if not ret:
            print("Unable to capture video")
            break

        # Convert BGR image to RGB
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False

        # Detect pose landmarks
        results = pose.process(image)

        # If pose detected
        if results.pose_landmarks:
            # Draw landmarks on the frame
            mp_drawing.draw_landmarks(
                frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

            # Check if all landmarks are at their correct position
            all_visible = True
            for landmark in results.pose_landmarks.landmark:
                if landmark.visibility < 0.5:
                    all_visible = False
                    break

            if all_visible:
                # time.sleep(3)
                cnt=cnt+1
                cv2.putText(frame, "Stand Still", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
                
                # print("correct position")
                if cnt == 100:
                    captureImg=True
                    
                
            else:
                cv2.putText(frame, "Full Body Not Visible", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
                # print("wrong position")

        else:
            # No person detected
            cv2.putText(frame, "No person", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
            
            # print("No person")
            
        # Get the screen resolution
        # screen_width = 1920  # Set your screen resolution width
        # screen_height = 1080
        # Show the frame
        # frame = cv2.resize(frame, (screen_width, screen_height))

        # Show the frame in full screen mode
        # cv2.namedWindow('MediaPipe Pose', cv2.WND_PROP_FULLSCREEN)
        # cv2.setWindowProperty('MediaPipe Pose', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        cv2.imshow('MediaPipe Pose', frame)
        # cv2.imshow('MediaPipe Pose', frame)

        # Break the loop if 'q' key is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release video capture and close all windows
    cap.release()
    cv2.destroyAllWindows()

if __name__=='__main__':
    img_path='/home/saiganesh.s/ML/realtime/captured_image.jpg'
    st.title('Real-Time Clothing Recommendation System')
    top_size='S'
    bottom_size='28'
    # Take choice from user
    choice = st.radio("Select your choice:", ('top', 'bottom'))
    gender = st.radio("Select your gender:", ('male', 'female'))

    if st.button('Start Video Capture'):
        # global video_capture_running
        video_capture_running = True
        main()
        st.success('Video capture finished successfully!')

        # Pixel Distances
        pix_height,pix_shoulder_distance,pix_waist_distance,top_pixel_distance,upperbody_length,lowerbody_length=calculate_person_height(img_path)
        # st.write("Pixel Distances Calculated")
        
        # Add Real Measurements calculation model
        rf_height_model,rf_shoulder_model,rf_waist_model=get_body_measurements_models()

        real_height=rf_height_model.predict([[top_pixel_distance,pix_height]])
        real_shoulder=rf_shoulder_model.predict([[top_pixel_distance,pix_height,pix_shoulder_distance]])
        real_waist=rf_waist_model.predict([[top_pixel_distance,pix_height,pix_waist_distance]])

        if(gender=='male'):
            top_size,bottom_size=male_size_chart(real_shoulder,real_waist)
        else:
            top_size,bottom_size=female_size_chart(real_shoulder,real_waist)

        if(choice=='top'):
            st.write("Top Size:",top_size)
        else:
            st.write("Bottom Size:",bottom_size)
        # st.write("Height: ",real_height)
        # st.write("Shoulder: ",real_shoulder)
        # st.write("Waist: ",real_waist)

        # Clothes Detection and extraction
        get_clothes(img_path, choice)
        # st.write("Clothes Extracted")

        # Recommendation System
        rs_img_path="/home/saiganesh.s/ML/realtime/masked_area.jpg"

        model=load_rs_resnet_model()
        neighbors_male=load_rs_knn_model()
        neighbors_female=load_rs_knn_model()
        # st.write("Recommendation Model Loaded")
        feature_list_male,outfit_ids_male,item_ids_male,feature_list_female,outfit_ids_female,item_ids_female=load_rs_weights()
        # st.write("Weights Loaded")

        neighbors_male.fit(feature_list_male)
        # st.write("Male KNN Model Trained")
        
        input_features = extract_features(rs_img_path, model)
        # st.write("Feature Extracted From the masked cloth")

        if(gender=='female'):
            neighbors_female.fit(feature_list_female)
            # st.write("Female KNN Model Trained")
            distances, indices = neighbors_female.kneighbors([input_features])

            for i, index in enumerate(indices[0]):
                st.write(f"Neighbor {i+1}:")
                # st.write(f" Outfit ID: {outfit_ids_female[index]}, Item ID: {item_ids_female[index]}")
                # st.write(f"Distance: {distances[0][i]}")

                # Construct the directory path for the outfit
                outfit_dir = f'/home/saiganesh.s/ML/realtime/outfit-dataset/female/{outfit_ids_female[index]}'

                # List all files in the outfit directory
                files = os.listdir(outfit_dir)
                images = []
                # Display all images except for the item itself
                for file in files:
                    if file.endswith(".jpg") and file != f'{item_ids_female[index]}':
                        img_path = os.path.join(outfit_dir, file)
                        images.append(img_path)

                st.image(images, width=200)
                st.write()
                
        distances, indices = neighbors_male.kneighbors([input_features])

        for i, index in enumerate(indices[0]):
            st.write(f"Neighbor {i+1}:")
            # st.write(f" Outfit ID: {outfit_ids_male[index]}, Item ID: {item_ids_male[index]}")
            # st.write(f"Distance: {distances[0][i]}")

            # Construct the directory path for the outfit
            outfit_dir = f'/home/saiganesh.s/ML/realtime/outfit-dataset/male/{outfit_ids_male[index]}'

            # List all files in the outfit directory
            files = os.listdir(outfit_dir)
            images = []
            # Display all images except for the item itself
            for file in files:
                if file.endswith(".jpg") and file != f'{item_ids_male[index]}':
                    # print("file=",file)
                    # print("Item_ids_male=",item_ids_male[index])
                    img_path = os.path.join(outfit_dir, file)
                    images.append(img_path)

            st.image(images, width=200)
            st.write()

    if st.button('Stop Video Capture'):
        # global video_capture_running
        video_capture_running = False
        st.success('Video capture stopped successfully!')
