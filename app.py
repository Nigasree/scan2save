from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os

app = Flask(__name__)

# Load your trained model
model = load_model('model/scan2save_tomato_model.h5')

@app.route('/', methods=['GET', 'POST'])
def index():
    result = None
    if request.method == 'POST':
        # Check if the file is uploaded
        if 'file' not in request.files:
            return "No file uploaded", 400

        img_file = request.files['file']

        if img_file.filename == '':
            return "No selected file", 400

        # Save image to static/test.jpg
        img_path = os.path.join('static', 'test.jpg')
        img_file.save(img_path)

        # Preprocess the image to match model input (128x128x3)
        img = image.load_img(img_path, target_size=(128, 128))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0) / 255.0

        # Make prediction
        prediction = model.predict(img_array)[0][0]
        result = "Spoiled ! Don't consume" if prediction > 0.5 else "Fresh....safe to consume"

    return render_template('index.html', result=result)

if __name__ == '__main__':
    app.run(debug=True)
