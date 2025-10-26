import streamlit as st
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import os
import pandas as pd
from deepface import DeepFace
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, RTCConfiguration
import av
import cv2

# ==============================================================================
# --- 1. CONFIGURATION ---
# ==============================================================================
st.set_page_config(layout="wide", page_title="Advanced Face Analysis")

st.title("✨ Advanced Multi-Face & Real-Time Analysis")
st.write("Upload an image or use your webcam to detect and analyze multiple faces for identity, emotion, age, and gender.")

# --- Paths & Models ---
DATABASE_PATH = "lfw-subset-large/"

# --- DeepFace Configuration ---
RECOGNITION_MODEL = "ArcFace"

# ==============================================================================
# --- 2. WEBCAM & HELPER CLASSES ---
# ==============================================================================

class VideoTransformer(VideoTransformerBase):
    """
    Processes video frames from the webcam for real-time analysis.
    """
    def __init__(self):
        self.frame_counter = 0

    def transform(self, frame: av.VideoFrame) -> np.ndarray:
        image_np = frame.to_ndarray(format="bgr24")
        self.frame_counter += 1

        # Process every 5th frame for performance
        if self.frame_counter % 5 == 0:
            try:
                analysis_results = DeepFace.analyze(
                    img_path=image_np,
                    actions=['emotion', 'age', 'gender'],
                    enforce_detection=False,
                    silent=True
                )
                analysis_results = [res for res in analysis_results if res is not None and res.get('region')]

                for face in analysis_results:
                    box = face['region']
                    x, y, w, h = box['x'], box['y'], box['w'], box['h']
                    
                    label = f"{face['dominant_emotion']}"
                    cv2.rectangle(image_np, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    cv2.putText(image_np, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                    
                    sub_label = f"Age: {face['age']}, {face['dominant_gender']}"
                    cv2.putText(image_np, sub_label, (x, y + h + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

                return image_np
            except Exception:
                return image_np
        
        return image_np

@st.cache_resource
def get_font():
    """Load a font for drawing on images, with a fallback."""
    try:
        return ImageFont.truetype("arial.ttf", 22)
    except IOError:
        return ImageFont.load_default()

def resize_image(image, max_size=(800, 800)):
    """
    Resizes a PIL Image to fit within a max_size box while maintaining aspect ratio.
    """
    img_copy = image.copy()
    img_copy.thumbnail(max_size, Image.Resampling.LANCZOS)
    return img_copy

# ==============================================================================
# --- 3. CORE PROCESSING FUNCTION ---
# ==============================================================================

def process_and_display_results(image_np, analysis_results, distance_threshold):
    """
    Processes analysis results, finds identities, draws on the image, and displays it.
    """
    original_image = Image.fromarray(image_np)
    img_to_draw = original_image.copy()
    draw = ImageDraw.Draw(img_to_draw)
    font = get_font()

    all_find_results = []

    for face in analysis_results:
        box = face['region']
        x, y, w, h = box['x'], box['y'], box['w'], box['h']
        
        cropped_face_np_rgb = image_np[y:y+h, x:x+w]
        cropped_face_np_bgr = cv2.cvtColor(cropped_face_np_rgb, cv2.COLOR_RGB2BGR)

        try:
            result_df_list = DeepFace.find(
                img_path=cropped_face_np_bgr,
                db_path=DATABASE_PATH,
                model_name=RECOGNITION_MODEL,
                enforce_detection=False,
                silent=True
            )
            
            if result_df_list and not result_df_list[0].empty:
                top_match = result_df_list[0].iloc[0]
                distance = top_match['distance']
                
                if distance < distance_threshold:
                    identity_path = top_match['identity']
                    identity_name = os.path.basename(os.path.dirname(identity_path))
                    all_find_results.append({"identity": identity_name, "distance": distance})
                else:
                    closest_identity_path = top_match['identity']
                    closest_name = os.path.basename(os.path.dirname(closest_identity_path))
                    all_find_results.append({"identity": "Unknown", "distance": distance, "closest_match": closest_name})
            else:
                all_find_results.append({"identity": "Unknown", "distance": float('inf')})

        except Exception as e:
             st.toast(f"Error during face search: {e}", icon="⚠️")
             all_find_results.append({"identity": "Error", "distance": float('inf')})

    for i, face in enumerate(analysis_results):
        box = face['region']
        x, y, w, h = box['x'], box['y'], box['w'], box['h']
        identity_info = all_find_results[i]
        identity = identity_info['identity']
        
        box_color = "lime" if identity != "Unknown" else "red"
        label = f"{identity} ({face['dominant_emotion']})"
        
        draw.rectangle([x, y, x + w, y + h], outline=box_color, width=3)
        text_bbox = draw.textbbox((x, y - 30), label, font=font)
        draw.rectangle(text_bbox, fill=box_color)
        draw.text((x + 5, y - 28), label, fill="black", font=font)

    col1, col2 = st.columns([2, 1])
    with col1:
        st.header("🖼️ Processed Image")
        # --- MODIFIED: Resize the image before displaying for a cleaner UI ---
        resized_image = resize_image(img_to_draw)
        st.image(resized_image, caption="Analysis Result")

    with col2:
        st.header("🔍 Analysis Details")
        if not analysis_results:
            st.warning("No faces were detected.")
        else:
            st.info(f"Detected {len(analysis_results)} face(s).")
            for i, face in enumerate(analysis_results):
                identity_info = all_find_results[i]
                with st.expander(f"**Face #{i+1}**: {identity_info['identity']}"):
                    st.write(f"**Identity:** {identity_info['identity']}")
                    if identity_info['identity'] != "Unknown":
                        st.write(f"**Match Confidence (Distance):** {identity_info['distance']:.4f}")
                    elif "closest_match" in identity_info:
                         st.warning(f"Closest match was '{identity_info['closest_match']}' with distance {identity_info['distance']:.4f}, which is above the threshold.")
                    
                    st.write(f"**Dominant Emotion:** {face['dominant_emotion'].capitalize()}")
                    st.write(f"**Estimated Age:** {face['age']}")
                    st.write(f"**Estimated Gender:** {face['dominant_gender'].capitalize()}")
                    
                    st.write("**Emotion Scores:**")
                    emotion_df = pd.DataFrame(face['emotion'].items(), columns=['Emotion', 'Confidence'])
                    st.bar_chart(emotion_df.set_index('Emotion'))

# ==============================================================================
# --- 4. STREAMLIT UI ---
# ==============================================================================
if not os.path.exists(DATABASE_PATH):
    st.error(f"Database path not found: '{DATABASE_PATH}'. Please ensure the directory exists.")
    st.stop()

st.sidebar.header("⚙️ Settings")

distance_threshold = st.sidebar.slider(
    'Recognition Confidence Threshold',
    min_value=0.0, max_value=2.0, value=0.68, step=0.01,
    help="Lower values mean stricter matching. ArcFace models work best around 0.68."
)

tab1, tab2 = st.tabs(["📤 Upload Image", "📹 Live Webcam Analysis"])

with tab1:
    st.sidebar.header("Image Upload")
    uploaded_file = st.sidebar.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    
    st.sidebar.info(
        "**How It Works:**\n\n"
        "1. **Detection & Analysis:** Finds all faces and analyzes their attributes.\n"
        "2. **Recognition:** Each face is individually compared against the database.\n"
        f"3. **Verification:** A match is confirmed if its distance is below the set threshold."
    )

    if uploaded_file is not None:
        try:
            image = Image.open(uploaded_file).convert("RGB")
            image_np = np.array(image)

            with st.spinner('Analyzing image... This may take a moment.'):
                analysis_results = DeepFace.analyze(
                    img_path=image_np,
                    actions=['emotion', 'age', 'gender'],
                    enforce_detection=False,
                    silent=True
                )
                analysis_results = [res for res in analysis_results if res is not None and res.get('region')]
                process_and_display_results(image_np, analysis_results, distance_threshold)
                
        except Exception as e:
            st.error(f"An error occurred during analysis: {e}")
    else:
        st.info("Please upload an image to begin analysis.")

with tab2:
    st.header("Live Webcam Feed")
    st.info("Allow webcam access, then click 'Start' to begin real-time analysis.")
    st.warning("Recognition is disabled in live mode for performance.")

    webrtc_streamer(
        key="webcam",
        video_transformer_factory=VideoTransformer,
        rtc_configuration=RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]})
    )

