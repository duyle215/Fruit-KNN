import streamlit as st
import cv2
import numpy as np
import joblib
from skimage.feature import hog, local_binary_pattern
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline

# ==== CONFIG ====
IMG_SIZE = 100
MODEL_PATH = "knn_image_model.pkl"

# ==== LOAD MODEL ====
(model, label_names) = joblib.load(MODEL_PATH)

# ==== FEATURE EXTRACTION ====
def extract_features(img):
    # Resize về 100x100x3
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # --- HOG ---
    hog_feat = hog(gray, orientations=9, pixels_per_cell=(8, 8),
                   cells_per_block=(2, 2), block_norm='L2-Hys',
                   transform_sqrt=True, feature_vector=True)

    # --- LBP ---
    lbp = local_binary_pattern(gray, P=24, R=3, method="uniform")
    (hist_lbp, _) = np.histogram(lbp.ravel(),
                                 bins=np.arange(0, 24 + 3),
                                 range=(0, 24 + 2))
    hist_lbp = hist_lbp.astype("float")
    hist_lbp /= (hist_lbp.sum() + 1e-6)

    # --- SURF (hoặc SIFT nếu SURF không có) ---
    try:
        surf = cv2.xfeatures2d.SURF_create(400)
    except:
        surf = cv2.SIFT_create()
    kp, des = surf.detectAndCompute(gray, None)
    if des is None:
        des = np.zeros((1, 64))  # tránh None
    surf_feat = des.mean(axis=0)  # lấy mean descriptor

    # --- Color Histogram ---
    chans = cv2.split(img)
    hist_color = []
    for chan in chans:
        hist = cv2.calcHist([chan], [0], None, [32], [0, 256])
        hist = cv2.normalize(hist, hist).flatten()
        hist_color.extend(hist)
    hist_color = np.array(hist_color)

    # --- Gộp tất cả ---
    feature_vector = np.hstack([hog_feat, hist_lbp, surf_feat, hist_color])
    return feature_vector


# ==== STREAMLIT APP ====
st.title("🍎🥭🍌 Phân loại trái cây với KNN")

uploaded_file = st.file_uploader("Tải lên một ảnh trái cây", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), caption="Ảnh đã tải lên", use_column_width=True)

    # Dự đoán
    feats = extract_features(img).reshape(1, -1)  # 1 sample
    probs = model.predict_proba(feats)[0]        # predict_proba tự động scale + knn
    pred_idx = np.argmax(probs)
    pred_label = label_names[pred_idx]
    pred_conf = probs[pred_idx] * 100

    st.markdown(f"### ✅ Dự đoán: **{pred_label}**")
    st.markdown(f"### 📊 Độ tự tin: **{pred_conf:.2f}%**")
