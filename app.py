import streamlit as st
import tensorflow as tf
from tensorflow.keras.applications.resnet50 import preprocess_input as resnet50_preprocess
from PIL import Image, ImageEnhance
import numpy as np
import time

# --- Page Configuration ---
st.set_page_config(
    page_title="Coffee Bean Quality Classifier",
    page_icon="☕",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Custom CSS for Better UI ---
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        color: #8B4513;
        margin-bottom: 0.5rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    .sub-header {
        text-align: center;
        color: #5D4037;
        font-size: 1.2rem;
        margin-bottom: 2rem;
    }
    .prediction-good {
        background: linear-gradient(135deg, #d4edda 0%, #c3e6cb 100%);
        border: 2px solid #28a745;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        font-weight: bold;
        color: #155724;
    }
    .prediction-bad {
        background: linear-gradient(135deg, #f8d7da 0%, #f1b0b7 100%);
        border: 2px solid #dc3545;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        font-weight: bold;
        color: #721c24;
    }
    .prediction-uncertain {
        background: linear-gradient(135deg, #fff3cd 0%, #fce4ec 100%);
        border: 2px solid #ffc107;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        font-weight: bold;
        color: #856404;
    }
    .confidence-metric {
        background: #f8f9fa;
        padding: 0.5rem 1rem;
        border-radius: 8px;
        margin: 0.25rem 0;
        border-left: 4px solid #007bff;
    }
    .metric-good {
        border-left-color: #28a745;
    }
    .metric-bad {
        border-left-color: #dc3545;
    }
</style>
""", unsafe_allow_html=True)


# -----------------------------
# Sidebar controls (UI options)
# -----------------------------
st.sidebar.header("🛠️ Settings")
enhance_contrast = st.sidebar.checkbox("Enhance image contrast", value=False)
confidence_threshold_high = st.sidebar.slider("High certainty threshold (%)", 60, 95, 75)
show_raw_data = st.sidebar.checkbox("Show raw model outputs", value=False)

softmax_label_order = st.sidebar.radio(
    "Softmax label order (choose based on how you set class_indices at training)",
    options=["bad, good (0=bad,1=good)", "good, bad (0=good,1=bad)"],
    index=0
)
sigmoid_positive_is_good = st.sidebar.checkbox("If model outputs sigmoid, treat value as probability of GOOD", value=True)

transfer_preprocess_mode = st.sidebar.selectbox(
    "Transfer-model preprocessing mode (try different if predictions look wrong)",
    options=["resnet50", "tf", "simple", "auto"],
    index=0
)
run_preprocess_diagnostics = st.sidebar.checkbox("Run transfer-model preprocess diagnostics", value=False)

# NEW: manual flip toggle for transfer model outputs
flip_transfer_manual = st.sidebar.checkbox("Flip Transfer Model Output (manual)", value=False)

# initialize session state for auto-detect flip
if "flip_transfer" not in st.session_state:
    st.session_state["flip_transfer"] = False


# -----------------------------
# Model loader
# -----------------------------
@st.cache_resource
def load_models():
    models = {}
    model_files = {
        "Custom CNN": "custom_model.h5",
        "Transfer Learning": "CoffeeBeanbest_model.h5"
    }
    for name, fname in model_files.items():
        try:
            models[name] = tf.keras.models.load_model(fname)
            st.sidebar.success(f"✅ {name} loaded")
        except Exception as e:
            models[name] = None
            st.sidebar.error(f"⚠️ Failed to load {name}: {e}")
    return models


# -----------------------------
# Preprocessing helper
# -----------------------------
def preprocess_image(image: Image.Image, target_size=(224, 224), enhance_contrast=False, model_name="Custom CNN", mode="auto"):
    if image.mode != "RGB":
        image = image.convert("RGB")
    if enhance_contrast:
        enhancer = ImageEnhance.Contrast(image)
        image = enhancer.enhance(1.2)

    image_resized = image.resize(target_size, Image.Resampling.LANCZOS)
    arr = np.asarray(image_resized, dtype=np.float32)
    arr = np.expand_dims(arr, axis=0)  # batch dim

    if mode == "auto":
        mode = "resnet50" if model_name == "Transfer Learning" else "simple"

    if mode == "resnet50":
        return resnet50_preprocess(arr.copy())
    elif mode == "tf":
        return (arr / 127.5) - 1.0
    elif mode == "simple":
        return arr / 255.0
    else:
        raise ValueError("Unknown preprocess mode")


# -----------------------------
# Prediction helper
# -----------------------------
def make_enhanced_prediction(model, processed_image, softmax_order, sigmoid_positive_good, high_thresh_pct):
    try:
        raw = model.predict(processed_image, verbose=0)[0]
    except Exception as e:
        return {"error": f"Prediction failed: {e}"}

    raw = np.asarray(raw).flatten()

    if raw.size == 1:
        if sigmoid_positive_good:
            confidence_good = float(raw[0]) * 100.0
            confidence_bad = 100.0 - confidence_good
        else:
            confidence_bad = float(raw[0]) * 100.0
            confidence_good = 100.0 - confidence_bad
    else:
        if softmax_order == "bad_good":
            confidence_bad = float(raw[0]) * 100.0
            confidence_good = float(raw[1]) * 100.0
        else:
            confidence_good = float(raw[0]) * 100.0
            confidence_bad = float(raw[1]) * 100.0

    medium_thresh = max(50, high_thresh_pct - 15)

    if confidence_good >= confidence_bad:
        chosen_label = "Good Bean"
        chosen_conf = confidence_good
    else:
        chosen_label = "Bad Bean"
        chosen_conf = confidence_bad

    if chosen_conf >= high_thresh_pct:
        certainty = "High"
    elif chosen_conf >= medium_thresh:
        certainty = "Medium"
    else:
        certainty = "Low"
        chosen_label = "Uncertain"

    return {
        "prediction": chosen_label,
        "confidence_good": confidence_good,
        "confidence_bad": confidence_bad,
        "certainty": certainty,
        "raw_output": raw.tolist()
    }


# -----------------------------
# Utility to flip a model result (swap confidences & label)
# -----------------------------
def maybe_flip_result(result):
    if not result or "raw_output" not in result:
        return result
    r = result.copy()
    # swap numerical confidences
    r["confidence_good"], r["confidence_bad"] = result["confidence_bad"], result["confidence_good"]
    # swap label
    if result["prediction"] == "Good Bean":
        r["prediction"] = "Bad Bean"
    elif result["prediction"] == "Bad Bean":
        r["prediction"] = "Good Bean"
    return r


# -----------------------------
# Confidence display helper
# -----------------------------
def display_confidence_metrics(result, model_name):
    st.markdown(f"**📊 {model_name} Confidence:**")
    st.markdown(f"<div class='confidence-metric metric-good'>🟢 Good Bean: <strong>{result['confidence_good']:.1f}%</strong></div>", unsafe_allow_html=True)
    st.markdown(f"<div class='confidence-metric metric-bad'>🔴 Bad Bean: <strong>{result['confidence_bad']:.1f}%</strong></div>", unsafe_allow_html=True)
    st.progress(min(max(result['confidence_good'] / 100.0, 0.0), 1.0))
    st.progress(min(max(result['confidence_bad'] / 100.0, 0.0), 1.0))


# -----------------------------
# Main app
# -----------------------------
def main():
    st.markdown('<h1 class="main-header">☕ Coffee Bean Quality Classifier</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Advanced AI-powered coffee bean quality assessment with model comparison</p>', unsafe_allow_html=True)

    models = load_models()

    col_upload, col_info = st.columns([2, 1])

    with col_upload:
        uploaded_file = st.file_uploader("📸 Upload your coffee bean image", type=["jpg", "jpeg", "png", "bmp", "tiff"])
    with col_info:
        with st.expander("ℹ️ How to get best results"):
            st.write("""
            **For optimal predictions:**
            - Use clear, well-lit images
            - Ensure beans are clearly visible
            - Avoid blurry or dark images
            - Single bean or small groups work best
            - Try contrast enhancement for low-light images
            """)

    if not uploaded_file:
        st.info("👆 Upload a coffee bean image to start the analysis")
        return

    image = Image.open(uploaded_file)
    st.subheader("📷 Uploaded Image")
    col_img, col_details = st.columns([2, 1])
    with col_img:
        st.image(image, caption="Original Image", use_column_width=True)
    with col_details:
        st.write("**Image Details:**")
        st.write(f"- Size: {image.size[0]} × {image.size[1]} pixels")
        st.write(f"- Mode: {image.mode}")
        st.write(f"- Format: {image.format}")
        if enhance_contrast:
            enhanced = ImageEnhance.Contrast(image).enhance(1.2)
            st.image(enhanced, caption="Enhanced Image", use_column_width=True)

    # Auto-detect button: tries to determine whether Transfer mapping should be flipped
    st.markdown("---")
    st.write("🔎 Mapping helper")
    if st.button("Auto-detect Transfer label mapping (compare Transfer vs Custom for this image)"):
        # need both models
        if models.get("Custom CNN") is None or models.get("Transfer Learning") is None:
            st.warning("Both models must be loaded for auto-detect.")
        else:
            try:
                proc_custom = preprocess_image(image, enhance_contrast=enhance_contrast, model_name="Custom CNN", mode="simple")
                r_custom = make_enhanced_prediction(
                    models["Custom CNN"], proc_custom,
                    softmax_order="good_bad",  # custom model mapping won't matter here; we assume custom trained mapping is consistent
                    sigmoid_positive_good=sigmoid_positive_is_good,
                    high_thresh_pct=confidence_threshold_high
                )
                proc_transfer = preprocess_image(image, enhance_contrast=enhance_contrast, model_name="Transfer Learning", mode=transfer_preprocess_mode)
                r_transfer = make_enhanced_prediction(
                    models["Transfer Learning"], proc_transfer,
                    softmax_order=("bad_good" if softmax_label_order.startswith("bad") else "good_bad"),
                    sigmoid_positive_good=sigmoid_positive_is_good,
                    high_thresh_pct=confidence_threshold_high
                )
                st.write("Custom model raw:", r_custom["raw_output"], "->", r_custom["prediction"])
                st.write("Transfer model raw:", r_transfer["raw_output"], "->", r_transfer["prediction"])

                # Decide: if they are opposite and both confident, flip suggested
                conf_custom = max(r_custom["confidence_good"], r_custom["confidence_bad"])
                conf_transfer = max(r_transfer["confidence_good"], r_transfer["confidence_bad"])
                if (r_custom["prediction"] in ["Good Bean", "Bad Bean"] and
                    r_transfer["prediction"] in ["Good Bean", "Bad Bean"] and
                    r_custom["prediction"] != r_transfer["prediction"] and
                    conf_custom >= 60 and conf_transfer >= 60):
                    st.session_state["flip_transfer"] = True
                    st.success("Auto-detect suggests flipping Transfer model mapping and has been applied.")
                else:
                    st.session_state["flip_transfer"] = False
                    st.info("Auto-detect did not find a confident opposite mapping (no flip applied).")
            except Exception as e:
                st.error(f"Auto-detect failed: {e}")

    st.markdown("---")

    # net effective flip flag: auto-detect override else manual checkbox
    flip_transfer_effective = st.session_state.get("flip_transfer", False) or flip_transfer_manual

    st.subheader("🤖 Model Predictions")
    results = {}
    prediction_cols = st.columns(2)

    soft_order_str = "bad_good" if softmax_label_order.startswith("bad") else "good_bad"

    for i, (model_name, model) in enumerate(models.items()):
        with prediction_cols[i]:
            st.markdown(f"### {model_name}")

            if model is None:
                st.error(f"❌ {model_name} not available")
                continue

            chosen_mode = transfer_preprocess_mode if model_name == "Transfer Learning" else "simple"

            if model_name == "Transfer Learning" and run_preprocess_diagnostics:
                st.info("Running preprocess diagnostics for Transfer model...")
                modes = ["resnet50", "tf", "simple"]
                diag = {}
                for mode in modes:
                    try:
                        proc = preprocess_image(image, enhance_contrast=enhance_contrast, model_name=model_name, mode=mode)
                        raw = model.predict(proc, verbose=0)[0]
                        diag[mode] = np.asarray(raw).tolist()
                    except Exception as e:
                        diag[mode] = f"error: {e}"
                st.expander("🔬 Preprocess diagnostics (raw outputs for each mode)").write(diag)

            try:
                processed = preprocess_image(image, enhance_contrast=enhance_contrast, model_name=model_name, mode=chosen_mode)
            except Exception as e:
                st.error(f"Preprocessing failed for {model_name}: {e}")
                continue

            with st.spinner(f"🔄 Analyzing with {model_name}..."):
                time.sleep(0.3)
                result = make_enhanced_prediction(
                    model,
                    processed,
                    softmax_order=soft_order_str,
                    sigmoid_positive_good=sigmoid_positive_is_good,
                    high_thresh_pct=confidence_threshold_high
                )

                if "error" in result:
                    st.error(result["error"])
                    results[model_name] = None
                    continue

                # Apply flip only to Transfer model if effective flag set
                if model_name == "Transfer Learning" and flip_transfer_effective:
                    result = maybe_flip_result(result)

                results[model_name] = result

                # Show prediction card
                if result['prediction'] == "Good Bean":
                    st.markdown(f'<div class="prediction-good">🟢 {result["prediction"]}<br>Confidence: {result["confidence_good"]:.1f}%<br>Certainty: {result["certainty"]}</div>', unsafe_allow_html=True)
                elif result['prediction'] == "Bad Bean":
                    st.markdown(f'<div class="prediction-bad">🔴 {result["prediction"]}<br>Confidence: {result["confidence_bad"]:.1f}%<br>Certainty: {result["certainty"]}</div>', unsafe_allow_html=True)
                else:
                    st.markdown(f'<div class="prediction-uncertain">🟡 {result["prediction"]}<br>Certainty: {result["certainty"]}</div>', unsafe_allow_html=True)

                with st.expander("📊 View Detailed Confidence"):
                    display_confidence_metrics(result, model_name)

                if show_raw_data:
                    with st.expander("🔧 Raw Model Output"):
                        st.write(result['raw_output'])

    # Comparative analysis
    valid_results = {k: v for k, v in results.items() if v}
    if len(valid_results) > 1:
        st.divider()
        st.subheader("🔍 Model Comparison")
        predictions_list = [r['prediction'] for r in valid_results.values()]
        if len(set(predictions_list)) == 1:
            st.success(f"✅ Model Agreement: Both models agree — {predictions_list[0]}")
        else:
            st.warning("⚠️ Model Disagreement: Models have different predictions. Check raw outputs/diagnostics.")

        comp_cols = st.columns(len(valid_results))
        for idx, (mname, res) in enumerate(valid_results.items()):
            with comp_cols[idx]:
                st.write(f"**{mname}**")
                st.write(f"Prediction: {res['prediction']}")
                st.write(f"Good: {res['confidence_good']:.1f}%  |  Bad: {res['confidence_bad']:.1f}%")
                st.write(f"Certainty: {res['certainty']}")

if __name__ == "__main__":
    main()
