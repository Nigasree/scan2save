from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

# Load the trained model
model = load_model("scan2save_tomato_model.h5")

# Load an image to test
img_path = "test.jpg"  # <- Replace this with your test image name
img = image.load_img(img_path, target_size=(128, 128))
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0) / 255.0

# Predict
prediction = model.predict(img_array)
class_name = "Fresh" if prediction < 0.5 else "Spoiled"

print(f"🧠 Prediction: {class_name}")
