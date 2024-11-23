import streamlit as st
import joblib
from tensorflow.keras.models import load_model
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
import os
import tensorflow as tf
from utils import readmask , normalizeImageIntensityRange , voxel2volume , scaleImg , predictVolume

# Initialize session state variables to persist data between interactions
if 'prediction' not in st.session_state:
    st.session_state.prediction = 0

if 'temp_filename' not in st.session_state:
    st.session_state.temp_filename = ''

# Load the trained model
def load_model2(grid_filename):
    """Load the trained GridSearchCV model from the specified file."""
    grid = joblib.load(grid_filename)
    best_model = grid.best_estimator_
    return best_model

def make_single_prediction(best_model, preprocessed_sample):
    """Make a single prediction using the best model."""
    prediction = best_model.predict(preprocessed_sample)
    return prediction

# Load model file
grid_filename = 'GSCV_NOFS.pkl'
best_model = load_model2(grid_filename)

# Streamlit app interface
st.title("MRS Prediction App")
st.write("""
    Please input the following features for the prediction.
""")

# File upload section
st.write("Upload a NIfTI File for Volume Estimation")

uploaded_file = st.file_uploader('Upload a .nii or .nii.gz file', type=['nii', 'nii.gz'])

if uploaded_file:
    # Save the uploaded file temporarily if not done yet
    if not st.session_state.temp_filename:
        st.session_state.temp_filename = f'./temp/{uploaded_file.name}'
        os.makedirs('./temp/', exist_ok=True)
        
        with open(st.session_state.temp_filename, 'wb') as f:
            f.write(uploaded_file.getbuffer())

    # Load NIfTI file
    nifti_data = nib.load(st.session_state.temp_filename)
    img_array = nifti_data.get_fdata()

    # Preprocess data if necessary and run the model
    imgTarget = normalizeImageIntensityRange(img_array)
    predImg = predictVolume(imgTarget)
    st.session_state.prediction = voxel2volume(nifti_data, predImg)

    st.write("Stroke Volume (in ml):", st.session_state.prediction)
    
    if st.button('More Details'):
        # Create the figure for more details
        fig = plt.figure(figsize=(15, 175))
        for i in range(predImg.shape[2]):
            # Plot the prediction image
            plt.subplot(predImg.shape[2], 2, 2 * i + 1)
            plt.imshow(predImg[:, :, i], cmap='gray')
            plt.title(f'Pred {i + 1}')
            plt.axis('off')

            # Plot the target image
            plt.subplot(predImg.shape[2], 2, 2 * i + 2)
            plt.imshow(imgTarget[:, :, i], cmap='gray')
            plt.title(f'Target {i + 1}')
            plt.axis('off')

        plt.tight_layout()
        # Use Streamlit's function to render the figure
        st.pyplot(fig)

# Create input fields for additional features
stroke_volume = st.session_state.prediction
age = st.number_input('Age', min_value=0.0, format="%.1f")
gender = st.selectbox('Gender', ['Female', 'Male'])
gender = 0 if gender == 'Female' else 1
nihss = st.number_input('NIHSS Score', min_value=0.0, format="%.1f")
sht = st.selectbox('SHT (0 for No, 1 for Yes)', [0, 1])
dm = st.selectbox('DM (0 for No, 1 for Yes)', [0, 1])
alcohol = st.selectbox('Alcohol (0 for No, 1 for Yes)', [0, 1])
smoking = st.selectbox('Smoking (0 for No, 1 for Yes)', [0, 1])
tobacco = st.selectbox('Tobacco (0 for No, 1 for Yes)', [0, 1])
dyslipidaemia = st.selectbox('Dyslipidaemia (0 for No, 1 for Yes)', [0, 1])
af = st.selectbox('Atrial Fibrillation (0 for No, 1 for Yes)', [0, 1])
ihd = st.selectbox('IHD (0 for No, 1 for Yes)', [0, 1])
rhd = st.selectbox('Rheumatic Heart Disease (0 for No, 1 for Yes)', [0, 1])
haemoglobin = st.number_input('Haemoglobin', min_value=0.0, format="%.1f")
pcv = st.number_input('PCV', min_value=0.0, format="%.1f")
mcv = st.number_input('MCV', min_value=0.0, format="%.1f")
homocystiene = st.number_input('Homocystiene', min_value=0.0, format="%.2f")
hba1c = st.number_input('HbA1C', min_value=0.0, format="%.1f")
cholesterol = st.number_input('Cholesterol', min_value=0.0, format="%.1f")
hdl = st.number_input('HDL Cholesterol', min_value=0.0, format="%.1f")
triglycerides = st.number_input('Triglycerides', min_value=0.0, format="%.1f")
vldl = st.number_input('VLDL', min_value=0.0, format="%.1f")
b12 = st.number_input('Vitamin B12', min_value=0.0, format="%.1f")
ct_aspects = st.number_input('CT ASPECTS', min_value=0.0, format="%.1f")
tan = st.number_input('TAN', min_value=0.0, format="%.1f")
mas = st.number_input('MAS', min_value=0.0, format="%.1f")
miteff = st.number_input('MITEFF', min_value=0.0, format="%.1f")
mcta = st.number_input('MCTA', min_value=0.0, format="%.1f")
mechanical_thrombectomy = st.selectbox('Mechanical Thrombectomy (0 for No, 1 for Yes)', [0, 1])
decompressive_hemicranectomy = st.selectbox('Decompressive Hemicranectomy (0 for No, 1 for Yes)', [0, 1])

# Prediction button
if st.button('Predict MRS'):
    # Gather all inputs into a single array
    input_features = np.array([[stroke_volume, age, gender, nihss, sht, dm, alcohol, smoking, tobacco, dyslipidaemia, 
                                af, ihd, rhd, haemoglobin, pcv, mcv, homocystiene, hba1c, cholesterol, hdl, triglycerides, 
                                vldl, b12, ct_aspects, tan, mas, miteff, mcta, mechanical_thrombectomy, 
                                decompressive_hemicranectomy]])
    
    # Make prediction using the best model
    prediction = make_single_prediction(best_model, input_features)
    
    # Display the prediction result
    if prediction[0] == 0:
        st.error(f"Predicted MRS for the sample: {prediction[0]} -> BAD [mRS >= 3]")
    else:
        st.success(f"Predicted MRS for the sample: {prediction[0]} -> GOOD [mRS <= 2]")
        
    # mRS scale image
    st.subheader("mRS Scale:")
    st.image("mRS_scale.jpg")
