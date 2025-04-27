import streamlit as st
import cv2
import numpy as np
import mss
import tempfile
from PIL import Image
import time
import torch
from torchvision import transforms
import platform
from CNN_opt import CNN
import warnings
warnings.filterwarnings("ignore", message="Tried to instantiate class .*torch.classes.*")

# Utility to check if running in WSL
def is_wsl():
    return "microsoft" in platform.uname().release.lower()

# Load face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Define and cache model loader
model = CNN()

@st.cache_resource
def load_model(model_path):
    model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
    model.eval()
    return model

# Image/frame preprocessing
def preprocess_frame(face):
    face = Image.fromarray(face)

    mean = (0.5,0.5,0.5)
    std = mean

    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(256),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    return val_transform(face).unsqueeze(0)

# Binary classifier
def classify_frame(model, tensor):
    with torch.no_grad():
        logits = model(tensor)  # shape: [1, 2]
        probs = torch.softmax(logits, dim=1)  # shape: [1, 2]
        fake_prob = probs[0, 1].item()  # probability of "fake"
        return fake_prob, fake_prob > 0.5


# Face detection + classification
def detect_and_classify(model, frame):
    predictions = []
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    for idx, (x, y, w, h) in enumerate(faces):
        face_crop_bgr = frame[y:y+h, x:x+w]
        # <-- convert to RGB so PIL/Image knows the right channels:
        face_crop = cv2.cvtColor(face_crop_bgr, cv2.COLOR_BGR2RGB)
        tensor = preprocess_frame(face_crop)

        with torch.no_grad():
            logits = model(tensor)
            probs = torch.softmax(logits, dim=1).squeeze()
            real_prob = probs[0].item()
            fake_prob = probs[1].item()

        label = f"Real: {real_prob:.2f} | Fake: {fake_prob:.2f}"
        color = (0, 255, 0) if real_prob >= fake_prob else (0, 0, 255)
        final_class = "Real" if real_prob >= fake_prob else "Fake"

        # Draw on image
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
        cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        predictions.append({
            "Face": idx + 1,
            "Real %": f"{real_prob:.2%}",
            "Fake %": f"{fake_prob:.2%}",
            "Prediction": final_class
        })

    return frame, predictions



# Screen grab
def capture_screen():
    with mss.mss() as sct:
        monitor = sct.monitors[1]
        screenshot = np.array(sct.grab(monitor))
        return cv2.cvtColor(screenshot, cv2.COLOR_BGRA2BGR)

# UI Layout
st.title("🧠 Real-Time Deepfake Detection")
st.markdown("""
Welcome to the Deepfake Detector App.  
You can run real-time deepfake analysis on:
- **Webcam feed** (Windows only)
- **Screen capture**
- **Uploaded images or videos**

Simply upload a trained model and choose your input source. The app will detect faces, run binary classification, and label them as **Real** or **Fake**.
""")

# Centered UI inputs
st.markdown("### 🔧 Configuration")
col1, col2 = st.columns(2)
with col1:
    input_options = ["Screen", "Upload Image", "Upload Video"]
    if not is_wsl():
        input_options.insert(0, "Webcam")
    input_mode = st.selectbox("Select Input Source", input_options)
with col2:
    nth_frame = st.slider("Process every Nth frame (video and screen capture only)", 1, 30, 5)

model_path = st.text_input("Model path", "/home/jpcha/TFM/src/preprocessing/extraction/best_CNN.pth")
load_model_btn = st.button("📥 Load Model")

# Load model if button clicked
if "model" not in st.session_state:
    st.session_state.model = None

if load_model_btn:
    try:
        st.session_state.model = load_model(model_path)
        st.success("✅ Model loaded successfully!")
    except Exception as e:
        st.error(f"❌ Failed to load model: {e}")

model = st.session_state.model
frame_area = st.image([])

# Start processing
if model:
    if input_mode == "Webcam":
        run = st.checkbox("🎥 Start Webcam")
        cap = cv2.VideoCapture(0)
        frame_count = 0
        while run:
            ret, frame = cap.read()
            if not ret:
                break
            frame_count += 1
            if frame_count % nth_frame == 0:
                processed_frame, predictions = detect_and_classify(model, frame)
            frame_area.image(processed_frame, channels="BGR")
            if predictions:
                st.markdown("### 🧠 Model Predictions")
                st.dataframe(predictions, use_container_width=True)
            else:
                st.info("No faces detected.")
        cap.release()

    elif input_mode == "Screen":
        run = st.checkbox("🖥️ Start Screen Capture")
        frame_count = 0
        while run:
            frame = capture_screen()
            frame_count += 1
            if frame_count % nth_frame == 0:
                processed_frame, predictions = detect_and_classify(model, frame)
            frame_area.image(processed_frame, channels="BGR")
            if predictions:
                st.markdown("### 🧠 Model Predictions")
                st.dataframe(predictions, use_container_width=True)
            else:
                st.info("No faces detected.")
            time.sleep(0.03)

    elif input_mode == "Upload Image":
        uploaded_image = st.file_uploader("📁 Upload an image", type=["jpg", "png", "jpeg"])
        if uploaded_image:
            img = np.asarray(bytearray(uploaded_image.read()), dtype=np.uint8)
            frame = cv2.imdecode(img, 1)
            processed_frame, predictions = detect_and_classify(model, frame)
            frame_area.image(processed_frame, channels="BGR")
            if predictions:
                st.markdown("### 🧠 Model Predictions")
                st.dataframe(predictions, use_container_width=True)
            else:
                st.info("No faces detected.")

    elif input_mode == "Upload Video":
        uploaded_video = st.file_uploader("📁 Upload a video", type=["mp4", "avi", "mov"])
        if uploaded_video:
            temp = tempfile.NamedTemporaryFile(delete=False)
            temp.write(uploaded_video.read())
            cap = cv2.VideoCapture(temp.name)
            frame_count = 0
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                frame_count += 1
                if frame_count % nth_frame == 0:
                    processed_frame, predictions = detect_and_classify(model, frame)
                frame_area.image(frame, channels="BGR")
                time.sleep(0.03)
            cap.release()
else:
    st.info("Please load a model to get started.")
