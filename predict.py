import cv2
import numpy as np
import tensorflow as tf
import os
from tqdm import tqdm

# --- Configuration & Model Loading ---

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
RECOGNITION_THRESHOLD = 0.8  # Lower value means stricter matching

def load_all_models():
    """Load all necessary models and classifiers."""
    if not all(os.path.exists(p) for p in [RECOGNITION_MODEL_PATH, EMOTION_MODEL_PATH, CASCADE_PATH]):
        print("Error: One or more model/cascade files are missing. Please ensure all are trained/present.")
        return None, None, None
        
    face_cascade = cv2.CascadeClassifier(CASCADE_PATH)
    emotion_model = tf.keras.models.load_model(EMOTION_MODEL_PATH)
    recognition_model = tf.keras.models.load_model(RECOGNITION_MODEL_PATH)
    print("All models loaded successfully.")
    return face_cascade, emotion_model, recognition_model

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

def create_known_face_encodings(recognition_model):
    """Create a database of known face embeddings from the LFW dataset."""
    if not os.path.exists(LFW_DATASET_DIR):
        print("Warning: LFW dataset directory not found. Recognition will not be available.")
        return {}
        
    known_face_encodings = {}
    people_list = [p for p in os.listdir(LFW_DATASET_DIR) if os.path.isdir(os.path.join(LFW_DATASET_DIR, p))]
    
    print("Building face recognition database from LFW dataset...")
    for person_name in tqdm(people_list):
        person_dir = os.path.join(LFW_DATASET_DIR, person_name)
        person_images = [os.path.join(person_dir, img) for img in os.listdir(person_dir)]
        
        if not person_images:
            continue
            
        first_image_path = person_images[0]
        try:
            img = cv2.imread(first_image_path)
            if img is None: continue
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            processed_img = preprocess_image_for_recognition(img_rgb)
            embedding = recognition_model.predict(processed_img, verbose=0)[0]
            known_face_encodings[person_name] = embedding
        except Exception:
            continue
    print("Face database built successfully.")
    return known_face_encodings

def find_match(embedding, known_encodings, threshold):
    """Find the best match for an embedding in the known encodings database."""
    distances = {name: np.linalg.norm(embedding - known_embedding) for name, known_embedding in known_encodings.items()}
    if not distances:
        return "Unknown", float('inf')
        
    best_match_name = min(distances, key=distances.get)
    best_match_distance = distances[best_match_name]

    if best_match_distance < threshold:
        return best_match_name, best_match_distance
    else:
        return "Unknown", best_match_distance

# --- Main Application Logic ---
def main():
    face_cascade, emotion_model, recognition_model = load_all_models()
    if not all([face_cascade, emotion_model, recognition_model]):
        return
        
    known_face_encodings = create_known_face_encodings(recognition_model)
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Flip the frame horizontally
        frame = cv2.flip(frame, 1)
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        for (x, y, w, h) in faces:
            # Crop face ROI for both models
            face_roi_bgr = frame[y:y+h, x:x+w]
            face_roi_rgb = cv2.cvtColor(face_roi_bgr, cv2.COLOR_BGR2RGB)

            # --- Emotion Analysis ---
            processed_emotion_face = preprocess_image_for_emotion(face_roi_bgr)
            emotion_prediction = emotion_model.predict(processed_emotion_face, verbose=0)[0]
            dominant_emotion = EMOTION_LABELS[np.argmax(emotion_prediction)]

            # --- Recognition Analysis ---
            if known_face_encodings:
                processed_recognition_face = preprocess_image_for_recognition(face_roi_rgb)
                embedding = recognition_model.predict(processed_recognition_face, verbose=0)[0]
                name, distance = find_match(embedding, known_face_encodings, RECOGNITION_THRESHOLD)
                label_text = f"{name} ({dominant_emotion.capitalize()})"
            else:
                label_text = f"Emotion: {dominant_emotion.capitalize()}"
            
            # --- Draw on Frame ---
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame, label_text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # Display the resulting frame
        cv2.imshow('Real-Time Face Recognition and Emotion Analysis', frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the capture and destroy all windows
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()