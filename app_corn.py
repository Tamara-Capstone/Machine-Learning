from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
import numpy as np
from keras.models import load_model
from tensorflow.keras.preprocessing import image
from keras.applications.imagenet_utils import preprocess_input

app = Flask(__name__)

# Load the trained model
MODEL_PATH = 'models_mango.h5'
model = load_model(MODEL_PATH)
model.make_predict_function()  # Necessary

label_map = {
    0: "Mango_anthracnose_disease",
    1: "Mango_bacterial_canker_disease",
    2: "Mango_cutting_weevil_disease",
    3: "Mango_dieback_disease",
    4: "Mango_gall_midge_disease",
    5: "Mango_healthy",
    6: "Mango_powdery_mildew_disease",
    7: "Mango_sooty_mold_disease",
}

def model_predict(img_path, model):
    """
    Predict the disease of a mango leaf image.

    Args:
        img_path (str): The path to the image file.
        model (keras.Model): The trained model.

    Returns:
        A tuple containing the predicted class index and its probability.
    """
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)

    # Apply preprocessing (e.g., normalization)
    x = preprocess_input(x)

    preds = model.predict(x)
    predicted_class_index = np.argmax(preds)
    predicted_class_prob = preds[0][predicted_class_index]
    return predicted_class_index, predicted_class_prob

@app.route('/predict', methods=['POST'])
def predict():
    """
    Predict the disease of a mango leaf image uploaded by the user.

    Returns:
        A JSON response containing the predicted disease label and its probability.
    """
    # Check if an image file was uploaded
    if 'file' not in request.files:
        return jsonify({'error': 'No file found'})

    file = request.files['file']

    # Check if the file is a valid image
    if file.filename == '':
        return jsonify({'error': 'No image file selected'})

    if not file.filename.lower().endswith(('.jpg', '.jpeg', '.png', '.gif')):
        return jsonify({'error': 'Invalid image file'})

    # Save the file
    file_path = secure_filename(file.filename)
    file.save(file_path)

    # Make prediction
    predicted_class_index, predicted_class_prob = model_predict(file_path, model)

    # Process the result
    if predicted_class_prob < 0.3:
        predicted_label = "Unclear"
        predicted_prob = predicted_class_prob * 100
    else:
        predicted_label = label_map[predicted_class_index]
        predicted_prob = predicted_class_prob * 100

    # Return the result as JSON
    return jsonify({'label': predicted_label, 'probability': round(predicted_prob, 2)})

if __name__ == '__main__':
    app.run(debug=True)
