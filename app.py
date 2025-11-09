import os
import tempfile
import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, WebRtcMode, RTCConfiguration

# -------------------------------
# Config
# -------------------------------
st.set_page_config(page_title="Mask Detector — Full App", layout="wide")

MODEL_PATH = "mask_detector_v1.h5"
HAAR_PATH = "haarcascade_frontalface_default.xml"
HAAR_URL = "https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml"

CLASS_NAMES = ['mask_weared_incorrect', 'with_mask', 'without_mask']
INPUT_SIZE = (128, 128)

# Download Haar cascade if missing
if not os.path.exists(HAAR_PATH):
    import urllib.request
    urllib.request.urlretrieve(HAAR_URL, HAAR_PATH)

@st.cache_resource
def load_resources():
    cascade = cv2.CascadeClassifier(HAAR_PATH)
    model = load_model(MODEL_PATH, compile=False)
    return cascade, model

face_cascade, model = load_resources()

# -------------------------------
# Helper Class for Live Camera
# -------------------------------
class MaskTransformer(VideoTransformerBase):
    def __init__(self):
        self.frame_counter = 0

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        self.frame_counter += 1

        # Process every 3 frames to reduce lag
        if self.frame_counter % 3 == 0:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.1, 4)

            for (x, y, w, h) in faces:
                face = img[y:y+h, x:x+w]
                face = cv2.resize(face, INPUT_SIZE)
                face = face / 255.0
                face = np.expand_dims(face, axis=0)

                preds = model.predict(face, verbose=0)[0]
                idx = np.argmax(preds)
                label = CLASS_NAMES[idx]
                conf = preds[idx]

                color = (0, 255, 0) if label=="with_mask" else (0,165,255) if label=="mask_weared_incorrect" else (0,0,255)
                cv2.rectangle(img, (x, y), (x+w, y+h), color, 2)
                cv2.putText(img, f"{label} {conf*100:.1f}%", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        return img

# -------------------------------
# App UI
# -------------------------------
st.title("😷 Full Mask Detector App")
st.write("Choose an option below:")

option = st.radio("Mode:", ["Live Camera", "Upload Image", "Upload Video"])

# -------------------------------
# Live Camera
# -------------------------------
if option == "Live Camera":
    st.header("🎥 Live Camera Mask Detection")
    RTC_CONFIGURATION = RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]})
    webrtc_streamer(
        key="mask-detector",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration=RTC_CONFIGURATION,
        video_transformer_factory=MaskTransformer,
        media_stream_constraints={"video": True, "audio": False},
        async_transform=True
    )

# -------------------------------
# Upload Image
# -------------------------------
elif option == "Upload Image":
    st.header("📁 Upload Image for Mask Detection")
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)

        for (x, y, w, h) in faces:
            face = img[y:y+h, x:x+w]
            face = cv2.resize(face, INPUT_SIZE)
            face = face / 255.0
            face = np.expand_dims(face, axis=0)

            preds = model.predict(face, verbose=0)[0]
            idx = np.argmax(preds)
            label = CLASS_NAMES[idx]
            conf = preds[idx]

            color = (0, 255, 0) if label=="with_mask" else (0,165,255) if label=="mask_weared_incorrect" else (0,0,255)
            cv2.rectangle(img, (x,y), (x+w, y+h), color, 2)
            cv2.putText(img, f"{label} {conf*100:.1f}%", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        st.image(img, channels="BGR")

# -------------------------------
# Upload Video
# -------------------------------
elif option == "Upload Video":
    st.header("🎬 Upload Video for Mask Detection")
    uploaded_video = st.file_uploader("Choose a video...", type=["mp4", "mov", "avi"])
    if uploaded_video is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_video.read())
        
        cap = cv2.VideoCapture(tfile.name)
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.1, 4)

            for (x, y, w, h) in faces:
                face = frame[y:y+h, x:x+w]
                face = cv2.resize(face, INPUT_SIZE)
                face = face / 255.0
                face = np.expand_dims(face, axis=0)

                preds = model.predict(face, verbose=0)[0]
                idx = np.argmax(preds)
                label = CLASS_NAMES[idx]
                conf = preds[idx]

                color = (0, 255, 0) if label=="with_mask" else (0,165,255) if label=="mask_weared_incorrect" else (0,0,255)
                cv2.rectangle(frame, (x,y), (x+w, y+h), color, 2)
                cv2.putText(frame, f"{label} {conf*100:.1f}%", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

            st.image(cv2.resize(frame, (640,480)), channels="BGR")
        cap.release()
