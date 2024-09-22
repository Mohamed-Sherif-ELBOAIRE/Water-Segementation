import os
import numpy as np
from PIL import Image
import tensorflow as tf
from flask import Flask, request, render_template, redirect, url_for
from werkzeug.utils import secure_filename
import tifffile as tiff
import matplotlib.pyplot as plt
from io import BytesIO
import io
import base64
from sklearn.preprocessing import MinMaxScaler



def ndwi(image):
    green_band = image[:, :, 2]  # Assuming Band 2 is Green
    nir_band = image[:, :, 4]    # Assuming Band 4 is NIR
    multispectral_images = (green_band - nir_band) / (green_band + nir_band + 1e-10)  # Avoid division by zero
    return multispectral_images



# Initialize the Flask app
app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load the trained model
model_path = "D:/Mohamed Sheriff/Projects/Computer Vision Internship - Cellula Technologies/Water Segementation/Model/9channels.keras"
model = tf.keras.models.load_model(model_path)




# Ensure the uploads folder exists
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Route to the homepage
@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict', methods=['POST'])
def predict():
    scaler = MinMaxScaler()

    #if request.method == 'POST':
    # Handle the file upload
    if 'file' not in request.files:
        return 'Upload a File', 400

    file = request.files['file']
    if file.filename == '':
        return 'Select a File', 400

    if file and file.filename.endswith('.tif'):
        # Save the uploaded file
        filepath = os.path.join(UPLOAD_FOLDER, secure_filename(file.filename))
        file.save(filepath)

    # Load and process the image
    image = tiff.imread(filepath)
    channels_9_image = image[:, :, 3:]  # Use only the last 9 channels
    channels_9_image = channels_9_image.reshape(-1, 9)
    channels_9_image = scaler.fit_transform(channels_9_image).reshape(128, 128, 9)

    # Predict the mask
    image_for_model = np.expand_dims(channels_9_image, axis=0)
    predicted_mask = model.predict(image_for_model)
    predicted_mask = (predicted_mask.squeeze() > 0.5).astype(np.uint8)
    predicted_mask_no_threshold = predicted_mask.squeeze()  # No threshold applied, plot the raw prediction

    
    ndwi_image = ndwi(image)
    # Plot the original and predicted images
    fig, ax = plt.subplots(2, 2, figsize=(12, 10))
    ax[0,0].imshow(image[:,1:, 2:3])  # Show RGB channels
    ax[0,0].set_title('Original Image (RGB)', fontsize=18)
    
    ax[0,1].imshow(ndwi_image, cmap='Blues')
    ax[0,1].set_title('NDWI Composite Image', fontsize=18)
    
    ax[1,0].imshow(predicted_mask_no_threshold, cmap='gray')
    ax[1,0].set_title('Predicted Mask (No Threshold)', fontsize=18)
    
    ax[1,1].imshow(predicted_mask, cmap='binary')
    ax[1,1].set_title('Predicted Mask (With Threshold)', fontsize=18)
    plt.tight_layout()  # Adjusts the padding between subplots
    plt.subplots_adjust(wspace=0.3)
    # Convert input image to base64
    img_io = io.BytesIO()
    #Image.fromarray(image[:, :, :3].astype('uint8')).save(img_io, 'PNG')
    plt.savefig(img_io, format='png')
    img_io.seek(0)
    encoded_input_img = base64.b64encode(img_io.getvalue()).decode('utf-8')



    return render_template('home.html', output=encoded_input_img)

    #return render_template('home.html', image=image_base64)
    #return redirect(request.url)
# Run the app
if __name__ == '__main__':
    app.run(debug=True)