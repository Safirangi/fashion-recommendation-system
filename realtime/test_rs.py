import pickle
import pandas as pd
import tensorflow as tf
import numpy as np
from numpy.linalg import norm
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from sklearn.neighbors import NearestNeighbors
import cv2

# Function to extract features from an image
def extract_features(img_path, model):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    expanded_img_array = np.expand_dims(img_array, axis=0)
    preprocessed_img = preprocess_input(expanded_img_array)
    result = model.predict(preprocessed_img).flatten()
    normalized_result = result / norm(result)
    return normalized_result

def load_rs_resnet_model():
# Load pre-trained ResNet50 model
    model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    model.trainable = False

    # Build a Sequential model for feature extraction
    model = tf.keras.Sequential([
        model,
        GlobalMaxPooling2D()
    ])

    # Initialize NearestNeighbors model
    
    return model

def load_rs_knn_model():
    neighbors = NearestNeighbors(n_neighbors=5, algorithm='brute', metric='euclidean')
    return neighbors

def load_rs_weights():
    # Load features and metadata from numpy array
    data_male = np.load('rs_extracted_features_male.npz')
    feature_list_male = data_male['features']
    # user_ids = data['user_id']
    outfit_ids_male = data_male['outfit_id']
    item_ids_male = data_male['item_id']
    data_female = np.load('rs_extracted_features_female.npz')
    feature_list_female = data_female['features']
    # user_ids = data['user_id']
    outfit_ids_female = data_female['outfit_id']
    item_ids_female = data_female['item_id']
    return feature_list_male,outfit_ids_male,item_ids_male,feature_list_female,outfit_ids_female,item_ids_female

if __name__=='__main__':
    # Specify the path to the image file
    img_path = r'D:\BMS\8th sem\PW\Fashion Recommendation System\IQON3000\1938\1129626\2598501_m.jpg'
    model,neighbors=load_rs_model()
    # Extract features from the input image
    input_features = extract_features(img_path, model)

    # Find nearest neighbors
    distances, indices = neighbors.kneighbors([input_features])

    feature_list,user_ids,outfit_ids,item_ids=load_rs_weights()

    # Print the nearest neighbors along with their metadata
    for i, index in enumerate(indices[0]):
        print(f"Neighbor {i+1}:")
        print(f"User ID: {user_ids[index]}, Outfit ID: {outfit_ids[index]}, Item ID: {item_ids[index]}")
        print(f"Distance: {distances[0][i]}")
        print()