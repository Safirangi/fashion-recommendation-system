import pickle

def get_body_measurements_models():
    with open('rf_height_model.pkl', 'rb') as file:
        rf_height_model = pickle.load(file)
    
    with open('rf_shoulder_model.pkl', 'rb') as file:
        rf_shoulder_model = pickle.load(file)

    with open('rf_waist_model.pkl', 'rb') as file:
        rf_waist_model = pickle.load(file)
        
    return rf_height_model,rf_shoulder_model,rf_waist_model

if __name__=='__main__':
    pass