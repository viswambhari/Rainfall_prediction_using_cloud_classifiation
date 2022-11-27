from __future__ import division, print_function
# coding=utf-8
import os
import numpy as np

from tensorflow import keras
from PIL import Image, ImageOps
import numpy as np

from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename


app = Flask(__name__)


def model_predict(img_path):
    np.set_printoptions(suppress=True)
    
    # Create the array of the right shape to feed into the keras model
    # The 'length' or number of images you can put into the array is
    # determined by the first position in the shape tuple, in this case 1.
    data = np.ndarray(shape=(1, 150, 150, 3), dtype=np.float32)
    
    # Replace this with the path to your image
    image = Image.open(img_path)
    #resizing the image to be at least 224x224 
    
    size = (150, 150)
    image = ImageOps.fit(image, size, Image.ANTIALIAS)
    
    #turn the image into a numpy array
    image_array = np.asarray(image)
    
    # Normalize the image
    normalized_image_array = (image_array.astype(np.float32) / 255.0) 
    
    # Load the image into the array
    data[0] = normalized_image_array
    
        
    # Load the model
    model = keras.models.load_model('model_2.h5')
    
    # run the inference
    preds = ""
    prediction = model.predict(data)
    # max_val = "%.2f" % max_val
    if np.argmax(prediction)==0:
        #preds = f"It will rain in 12 to 24 hours"
        preds=f'cloud0.html'
    elif np.argmax(prediction)==1:
        #preds = f" No rain"
        preds=f'cloud1.html'
    elif np.argmax(prediction)==2:
        #preds = f" No rain"
        preds=f'cloud2.html'
    elif np.argmax(prediction)==3:
       #preds = f" Rain"
       preds=f'cloud3.html'

    
    return preds  

@app.route('/', methods=['GET','POST'])
def index():
    # Main page
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']#file

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath,secure_filename(f.filename))
        f.save(file_path)

        # Make prediction
        preds = model_predict(file_path)
        return render_template(preds)
    return None

if __name__ == '__main__':
    app.run(debug=True)