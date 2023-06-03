from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.imagenet_utils import preprocess_input


app = Flask(__name__)

# Load the trained models
MODEL_PATHS = {
    "mango_model": {
        "path": "models_mango.h5",
        "label_map": {
            0: "Mango_anthracnose_disease",
            1: "Mango_bacterial_canker_disease",
            2: "Mango_cutting_weevil_disease",
            3: "Mango_dieback_disease",
            4: "Mango_gall_midge_disease",
            5: "Mango_healthy",
            6: "Mango_powdery_mildew_disease",
            7: "Mango_sooty_mold_disease",
        },
        "input_shape" : (224, 224, 3)
    },
    "chili_model": {
        "path": "Chili_model.h5",
        "label_map": {
            2: "chili_healthy",
            0: "chili_leaf_curl",
            1: "chili_leaf_spot",
            3: "chili_whitefly",
            4: "chili_yellowish"
        },
        "input_shape" : (220, 220, 3)
    },
    # "cassava_model": {
    #     "path": "./Cassava/",
    #     "label_map": {
    #         0: "cassava_bacterial_blight",
    #         1: "cassava_brown_streak_disease",
    #         2: "cassava_green_mottle",
    #         3: "cassava_healthy",
    #         4: "cassava_mosaic_disease"
    #     },
    #     "input_shape" : (224, 224, 3)
    # },
    "guava_model": {
        "path": "guavas_model2.h5",
        "label_map": {
            0: "guava_canker",
            1: "guava_dot",
            2: "guava_healthy",
            3: "guava_mummification",
            4: "guava_rust"
        },
        "input_shape" : (224, 224, 3)
    },
    "potato_model": {
        "path": "Potato_models.h5",
        "label_map": {
            0: "potato_early_blight",
            1: "potato_healthy",
            2: "potato_late_blight"
        },
        "input_shape" : (256, 256, 3)
    },
    "tomato_models": {
        "path": "tomato_models2.h5",
        "label_map": {
            0: "Tomato_bacterial_spot",
            1: "Tomato_early_blight",
            3: "Tomato_late_blight",
            4: "Tomato_leaf_mold",
            5: "Tomato_septoria_leaf",
            6: "Tomato_spider_mites",
            7: "Tomato_target_spot",
            8: "Tomato_mosaic_virus",
            9: "Tomato_healthy",
        },
        "input_shape" : (112, 112, 3)
    },
    "tea_model": {
        "path": "Tea_models.h5",
        "label_map": {
            0: "Tea_algal_leaf",
            1: "Tea_anthracnose",
            2: "Tea_bird_eye_spot",
            3: "Tea_brown_blight",
            4: "Tea_healthy",
            5: "Tea_red_leaf_spot"
        },
        "input_shape" : (250, 250, 3)
    },
    # "corn_model": {
    #     "path": "corn_model.h5",
    #     "label_map": {
    #         0: "corn_common_rust",
    #         1: "corn_gray_leaf_spot",
    #         2: "corn_healthy",
    #         3: "corn_northern_leaf_blight"
    #     },
    #     "input_shape" : (224, 224, 3)
    # },
}

models = {}
for model_name, model_info in MODEL_PATHS.items():
    model = load_model(model_info["path"])
    models[model_name] = model

def model_predict(img_path, model, label_map, input_shape):
    """
    Predict the label and confidence of an image using a specific model.

    Args:
        img_path (str): The path to the image file.
        model (keras.Model): The trained model.

    Returns:
        A tuple containing the predicted label and its confidence.
    """
    img = image.load_img(img_path, target_size=input_shape[:2])
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)

    preds = model.predict(x)
    label_index = np.argmax(preds)
    label = label_map[label_index]
    confidence = preds[0, label_index] * 100

    return label, confidence

@app.route('/predict/<model_name>', methods=['POST'])
def predict(model_name):
    """
    Predict the label and confidence of an image uploaded by the user.

    Returns:
        A JSON response containing the predicted label and its confidence.
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

    # Load the models
    model_info = MODEL_PATHS.get(model_name)
    if model_info is None:
        return jsonify({'error': 'Invalid model specified'})

    model = models[model_name]
    label_map = model_info["label_map"]
    input_shape = model_info["input_shape"]

    # Make predictions with the model
    preds, confidence = model_predict(file_path, model, label_map, input_shape)

    # Determine if the image is unclear
    unclear_threshold = 40
    is_unclear = bool(confidence < unclear_threshold)  # Convert to regular boolean type

    # Prepare the result
    result = {
        'label': preds,
        'confidence': confidence,
        'is_unclear': is_unclear,
        'label_map': label_map
    }

    # Return the result as JSON
    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True)
