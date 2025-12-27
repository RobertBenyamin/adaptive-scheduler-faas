import os
import time
import numpy as np
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.preprocessing import image as keras_image
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
from storage_helper import download_file
from dnld_blob import download_blob_new

TMP_DIR = "/tmp/"

# Global model - lazy loaded
net = None

def _initialize_model():
    """Lazy initialization of ResNet50 model"""
    global net
    
    if net is None:
        print("Loading ResNet50 model...", flush=True)
        net = ResNet50(weights='imagenet')
        print("Model loaded successfully", flush=True)

def lambda_handler(event):
    # Initialize model on first call
    _initialize_model()
    
    blobName = event.get("input_file", "img10.jpg")
    pid = str(os.getpid())
    base = os.path.basename(blobName)
    name, ext = os.path.splitext(base)
    proc_blob_name = f"{name}_{pid}{ext}"
    local_file_path = os.path.join(TMP_DIR, proc_blob_name)
    
    # Request centralized IO server to fetch the object for this pid
    try:
        download_blob_new(blobName)
        timeout = 2.0
        t0 = time.time()
        while not os.path.exists(local_file_path) and (time.time() - t0) < timeout:
            time.sleep(0.01)
        if not os.path.exists(local_file_path):
            download_file(blobName, local_file_path)
    except Exception:
        download_file(blobName, local_file_path)
    
    # Load and preprocess image using Keras/TensorFlow
    img = keras_image.load_img(local_file_path, target_size=(224, 224))
    img_array = keras_image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    
    # Make prediction
    preds = net.predict(img_array, verbose=0)
    
    # Decode predictions (top 5)
    decoded = decode_predictions(preds, top=5)[0]
    
    # Format output similar to original
    inference = ''
    for _, label, prob in decoded:
        inference += f'With prob = {prob:.5f}, it contains {label}. '
    
    # Clean up temp file
    try:
        if os.path.exists(local_file_path):
            os.remove(local_file_path)
    except:
        pass
    
    return {"result = ": inference, "output_file": blobName}