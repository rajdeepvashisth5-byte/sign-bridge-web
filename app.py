import streamlit as st
import numpy as np
import mediapipe as mp
from PIL import Image

MODEL_FILE = "model.npz"
MATCH_THRESHOLD = 0.75

st.set_page_config(page_title="Sign Bridge AI", page_icon="✦")

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
RunningMode = mp.tasks.vision.RunningMode


@st.cache_resource
def load_landmarker():
    options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path="hand_landmarker.task"),
        running_mode=RunningMode.IMAGE,
        num_hands=1
    )
    return HandLandmarker.create_from_options(options)


@st.cache_resource
def load_model():
    data = np.load(MODEL_FILE, allow_pickle=True)
    return data["X"], data["y"]


def normalize_landmarks(pts_xyz):
    pts = np.array(pts_xyz, dtype=np.float32)
    wrist = pts[0].copy()
    pts = pts - wrist
    dists = np.sqrt((pts[:, 0] ** 2) + (pts[:, 1] ** 2) + (pts[:, 2] ** 2))
    scale = dists.max()
    if scale < 1e-6:
        scale = 1.0
    pts = pts / scale
    return pts.flatten()


def predict(vector63, X, y):
    diffs = X - vector63
    dists = np.sqrt((diffs ** 2).sum(axis=1))
    best_idx = int(np.argmin(dists))
    best_dist = float(dists[best_idx])
    label = str(y[best_idx])
    confidence = max(0.0, 1.0 - (best_dist / 2.0)) * 100
    if best_dist > MATCH_THRESHOLD:
        return None, confidence
    return label, confidence


landmarker = load_landmarker()
X, y = load_model()
sign_list = sorted(set(y.tolist()))

st.title("✦ Sign Bridge AI")
st.write(
    f"Real-time Indian Sign Language recognizer. "
    f"Trained signs: **{', '.join(sign_list)}**"
)
st.write("Show a sign clearly to your camera, then click **Take Photo**.")

img_file = st.camera_input("Camera")

if img_file is not None:
    image = Image.open(img_file).convert("RGB")
    rgb = np.array(image)

    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    result = landmarker.detect(mp_image)

    if not result.hand_landmarks:
        st.error("NO HAND DETECTED — try again with your hand clearly in frame.")
    else:
        hand = result.hand_landmarks[0]
        pts_xyz = [[lm.x, lm.y, lm.z] for lm in hand]
        vector = normalize_landmarks(pts_xyz)
        label, confidence = predict(vector, X, y)

        if label is None:
            st.warning(f"UNKNOWN SIGN — confidence too low ({confidence:.0f}%)")
        else:
            st.success(f"Detected sign: **{label}**")
            st.write(f"Confidence: {confidence:.0f}%")
