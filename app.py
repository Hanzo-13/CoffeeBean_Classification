import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import io
import os # For listing models

# --- Configuration ---
MODEL_DIR = "models"
# Define the expected input size for your models (assuming they are consistent)
# You might need to adjust these based on what your specific models were trained with.
TARGET_SIZE = (224, 224) # Common size for many vision models

# Define the ripeness labels based on your model's output
# This is a placeholder; replace with your actual class names
RIPENESS_LABELS = ["Unripe", "Ripe", "Overripe"]
# You might have more specific labels or fewer, depending on your model.
# Example: If your model outputs 0 for unripe, 1 for ripe, 2 for overripe.

# --- Helper Functions ---
@st.cache_resource # Cache the model loading to avoid reloading on every rerun
def load_keras_model(model_path):
    """Loads a Keras model from a .keras file."""
    try:
        model = tf.keras.models.load_model(model_path)
        return model
    except Exception as e:
        st.error(f"Error loading Keras model {model_path}: {e}")
        return None

@st.cache_resource # Cache the model loading
def load_tflite_model(model_path):
    """Loads a TFLite model and allocates tensors."""
    try:
        interpreter = tf.lite.Interpreter(model_path=model_path)
        interpreter.allocate_tensors()
        return interpreter
    except Exception as e:
        st.error(f"Error loading TFLite model {model_path}: {e}")
        return None

def preprocess_image(image, target_size):
    """Preprocesses an image for model inference."""
    # Ensure image is RGB
    if image.mode != "RGB":
        image = image.convert("RGB")

    image = image.resize(target_size)
    image_array = np.asarray(image)
    image_array = image_array / 255.0 # Normalize pixel values to [0, 1]
    image_array = np.expand_dims(image_array, axis=0) # Add batch dimension
    return image_array.astype(np.float32) # TFLite often expects float32

def predict_with_keras(model, processed_image):
    """Makes a prediction using a Keras model."""
    predictions = model.predict(processed_image)
    return predictions[0] # Get the first (and only) sample's predictions

def predict_with_tflite(interpreter, processed_image):
    """Makes a prediction using a TFLite interpreter."""
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    interpreter.set_tensor(input_details[0]['index'], processed_image)
    interpreter.invoke()
    predictions = interpreter.get_tensor(output_details[0]['index'])
    return predictions[0] # Get the first (and only) sample's predictions

# --- Streamlit App Layout ---
st.set_page_config(page_title="Ripeness Detector", layout="centered")

st.title("🌱 Fruit/Vegetable Ripeness Detector")
st.markdown("""
Upload an image of a fruit or vegetable, and our models will try to predict its ripeness!
""")

# --- Model Selection ---
st.sidebar.header("Model Selection")

# Get a list of available models
available_models = [f for f in os.listdir(MODEL_DIR) if f.endswith(('.keras', '.tflite'))]
if not available_models:
    st.error(f"No models found in the '{MODEL_DIR}' directory. Please check your setup.")
    st.stop()

selected_model_name = st.sidebar.selectbox(
    "Choose a model:",
    available_models
)

selected_model_path = os.path.join(MODEL_DIR, selected_model_name)
model = None

if selected_model_name.endswith('.keras'):
    model = load_keras_model(selected_model_path)
elif selected_model_name.endswith('.tflite'):
    model = load_tflite_model(selected_model_path)

if model is None:
    st.warning("Please select a valid model from the sidebar to proceed.")
    st.stop()


# --- Image Upload ---
uploaded_file = st.file_uploader(
    "Upload an image here",
    type=["jpg", "jpeg", "png"],
    help="Drag and drop your image file or click to browse."
)

if uploaded_file is not None:
    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
    st.write("")
    st.write("Detecting ripeness...")

    # Load and preprocess the image
    image = Image.open(uploaded_file)
    processed_image = preprocess_image(image, TARGET_SIZE)

    # Make prediction
    predictions = None
    if selected_model_name.endswith('.keras'):
        predictions = predict_with_keras(model, processed_image)
    elif selected_model_name.endswith('.tflite'):
        predictions = predict_with_tflite(model, processed_image)

    if predictions is not None:
        predicted_class_idx = np.argmax(predictions)
        confidence = predictions[predicted_class_idx] * 100

        st.subheader(f"Prediction: {RIPENESS_LABELS[predicted_class_idx]}")
        st.write(f"Confidence: {confidence:.2f}%")

        # Display all probabilities
        st.subheader("All Probabilities:")
        for i, prob in enumerate(predictions):
            st.write(f"- {RIPENESS_LABELS[i]}: {prob*100:.2f}%")
    else:
        st.error("Could not get predictions from the model.")

st.sidebar.markdown("---")
st.sidebar.info("This application uses machine learning models to predict ripeness.")
