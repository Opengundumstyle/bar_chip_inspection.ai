import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# Load trained model
model = load_model("../models/defect_detector.h5")

def predict_chip(img_path):
    img = image.load_img(img_path, target_size=(128, 128))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)
    return "Defective" if prediction[0][0] > 0.5 else "Good"

# Example usage
if __name__ == "__main__":
    test_image = "../dataset/defective_chips/sample1.jpg"  # Replace with actual image
    print(f"Prediction: {predict_chip(test_image)}")
