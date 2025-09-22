import streamlit as st
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import tensorflow as tf
import cv2
import os
from tqdm import tqdm

# --- Configuration & Model Loading ---
st.set_page_config(layout="wide", page_title="Face Recognition and Emotion Analysis")
st.title("Face Recognition & Emotion Analysis Framework")

# --- Model Paths ---
RECOGNITION_MODEL_PATH = 'lfw_recognition_model.h5'
EMOTION_MODEL_PATH = 'affectnet_emotion_model.h5'
CASCADE_PATH = 'haarcascade_frontalface_default.xml'

# --- Emotion Recognition Configuration ---
EMOTION_IMG_SIZE = (48, 48)
EMOTION_LABELS = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

# --- Face Recognition Configuration ---
RECOGNITION_IMG_SIZE = (105, 105)
LFW_DATASET_DIR = os.path.join('lfw_dataset', 'lfw-deepfunneled', 'lfw-deepfunneled')
RECOGNITION_THRESHOLD = 0.8 # Lower value means stricter matching

@st.cache_resource
def load_models():
    """Load all models and classifiers into memory and cache them."""
    if not os.path.exists(CASCADE_PATH):
        st.error(f"Error: Cascade classifier not found at '{CASCADE_PATH}'")
        return None, None, None
    if not os.path.exists(EMOTION_MODEL_PATH):
        st.error(f"Error: Emotion model not found at '{EMOTION_MODEL_PATH}'. Please train it first.")
        return None, None, None
    if not os.path.exists(RECOGNITION_MODEL_PATH):
        st.error(f"Error: Recognition model not found at '{RECOGNITION_MODEL_PATH}'. Please train it first.")
        return None, None, None
        
    face_cascade = cv2.CascadeClassifier(CASCADE_PATH)
    emotion_model = tf.keras.models.load_model(EMOTION_MODEL_PATH)
    recognition_model = tf.keras.models.load_model(RECOGNITION_MODEL_PATH)
    return face_cascade, emotion_model, recognition_model

face_cascade, emotion_model, recognition_model = load_models()

# --- Preprocessing Functions ---
def preprocess_image_for_emotion(face_roi):
    """Preprocess a single face image for the emotion model."""
    gray_face = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
    resized_face = cv2.resize(gray_face, EMOTION_IMG_SIZE)
    processed_face = resized_face.astype('float32') / 255.0
    processed_face = np.expand_dims(processed_face, axis=0)
    processed_face = np.expand_dims(processed_face, axis=-1)
    return processed_face

def preprocess_image_for_recognition(face_roi):
    """Preprocess a single face image for the recognition model."""
    resized_face = cv2.resize(face_roi, RECOGNITION_IMG_SIZE)
    processed_face = resized_face.astype('float32') / 255.0
    processed_face = np.expand_dims(processed_face, axis=0)
    return processed_face
    
@st.cache_data
def create_known_face_encodings(_recognition_model):
    """Create a database of known face embeddings from the LFW dataset."""
    if not os.path.exists(LFW_DATASET_DIR):
        st.warning("LFW dataset directory not found. Recognition will not be available.")
        return {}, {}
        
    known_face_encodings = {}
    person_image_paths = {}

    people_list = [p for p in os.listdir(LFW_DATASET_DIR) if os.path.isdir(os.path.join(LFW_DATASET_DIR, p))]
    
    progress_bar = st.progress(0, text="Building face recognition database...")
    
    for i, person_name in enumerate(people_list):
        person_dir = os.path.join(LFW_DATASET_DIR, person_name)
        person_images = [os.path.join(person_dir, img) for img in os.listdir(person_dir)]
        
        if not person_images:
            continue
            
        # Use the first image as the reference for this person
        first_image_path = person_images[0]
        try:
            img = cv2.imread(first_image_path)
            if img is None: continue
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            processed_img = preprocess_image_for_recognition(img_rgb)
            embedding = _recognition_model.predict(processed_img, verbose=0)[0]
            known_face_encodings[person_name] = embedding
            person_image_paths[person_name] = first_image_path
        except Exception:
            continue # Skip if an image is corrupted
            
        progress_bar.progress((i + 1) / len(people_list), text=f"Processing {person_name}...")
        
    progress_bar.empty()
    return known_face_encodings, person_image_paths


if recognition_model:
    known_face_encodings, person_image_paths = create_known_face_encodings(recognition_model)
else:
    known_face_encodings, person_image_paths = {}, {}


def find_match(embedding, known_encodings, threshold):
    """Find the best match for an embedding in the known encodings database."""
    distances = {}
    for name, known_embedding in known_encodings.items():
        distance = np.linalg.norm(embedding - known_embedding)
        distances[name] = distance
        
    if not distances:
        return "Unknown", float('inf')
        
    best_match_name = min(distances, key=distances.get)
    best_match_distance = distances[best_match_name]

    if best_match_distance < threshold:
        return best_match_name, best_match_distance
    else:
        return "Unknown", best_match_distance

# --- Streamlit UI ---
col1, col2 = st.columns(2)

with col1:
    st.header("Upload Image")
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    
if uploaded_file is not None and all([face_cascade, emotion_model, recognition_model]):
    # Convert the uploaded file to an OpenCV image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image_bgr = cv2.imdecode(file_bytes, 1)
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    gray_image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    
    with col1:
        st.image(image_rgb, caption='Uploaded Image', use_column_width=True)

    # Detect faces
    faces = face_cascade.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    img_pil = Image.fromarray(image_rgb)
    draw = ImageDraw.Draw(img_pil)
    
    try:
        font = ImageFont.truetype("arial.ttf", 20)
    except IOError:
        font = ImageFont.load_default()

    with col2:
        st.header("Analysis Results")
        if len(faces) == 0:
            st.warning("No faces detected in the image.")
        else:
            st.info(f"Detected {len(faces)} face(s).")
            # Create a layout for each face
            for i, (x, y, w, h) in enumerate(faces):
                st.markdown(f"---")
                st.subheader(f"Face #{i+1}")
                
                # --- Emotion Analysis ---
                face_roi_bgr = image_bgr[y:y+h, x:x+w]
                processed_emotion_face = preprocess_image_for_emotion(face_roi_bgr)
                emotion_prediction = emotion_model.predict(processed_emotion_face, verbose=0)[0]
                dominant_emotion = EMOTION_LABELS[np.argmax(emotion_prediction)]

                # --- Recognition Analysis ---
                if known_face_encodings:
                    face_roi_rgb = image_rgb[y:y+h, x:x+w]
                    processed_recognition_face = preprocess_image_for_recognition(face_roi_rgb)
                    embedding = recognition_model.predict(processed_recognition_face, verbose=0)[0]
                    name, distance = find_match(embedding, known_face_encodings, RECOGNITION_THRESHOLD)
                    label_text = f"{name} ({dominant_emotion.capitalize()})"
                else:
                    name = "N/A"
                    distance = float('inf')
                    label_text = f"{dominant_emotion.capitalize()}"
                
                # Display text results
                st.write(f"**Recognized As:** {name} (Distance: {distance:.2f})")
                st.write(f"**Predicted Emotion:** {dominant_emotion.capitalize()}")

                # Draw on the image
                draw.rectangle([(x, y), (x+w, y+h)], outline="lime", width=3)
                text_size = draw.textlength(label_text, font)
                draw.rectangle([(x, y-25), (x + text_size + 8, y)], fill="lime")
                draw.text((x + 5, y - 22), label_text, fill="black", font=font)
            
            st.markdown(f"---")
            st.image(img_pil, caption='Processed Image', use_column_width=True)

