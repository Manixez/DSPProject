# rppg_processor.py
import numpy as np
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# POS method (Plane-Orthogonal-to-Skin)
def POS(signal, fps):
    eps = 1e-9
    X = signal  # shape: [1, 3, n]
    e, c, f = X.shape
    w = int(1.6 * fps)
    P = np.array([[0, 1, -1], [-2, 1, 1]])
    Q = np.stack([P for _ in range(e)], axis=0)
    H = np.zeros((e, f))

    for n in np.arange(w, f):
        m = n - w + 1
        Cn = X[:, :, m:(n+1)]
        M = 1.0 / (np.mean(Cn, axis=2) + eps)
        M = np.expand_dims(M, axis=2)
        Cn = np.multiply(Cn, M)
        S = np.dot(Q, Cn)
        S = S[0, :, :, :]
        S = np.swapaxes(S, 0, 1)
        S1 = S[:, 0, :]
        S2 = S[:, 1, :]
        alpha = np.std(S1, axis=1) / (eps + np.std(S2, axis=1))
        alpha = np.expand_dims(alpha, axis=1)
        Hn = S1 + alpha * S2
        Hnm = Hn - np.expand_dims(np.mean(Hn, axis=1), axis=1)
        H[:, m:(n + 1)] += Hnm
    return H

# BlazeFace face detector setup
base_model = "Model/blaze_face_short_range.tflite"
base_options = python.BaseOptions(model_asset_path=base_model)
FaceDetectorOptions = vision.FaceDetectorOptions
VisionRunningMode = vision.RunningMode

options = FaceDetectorOptions(
    base_options=base_options,
    running_mode=VisionRunningMode.IMAGE,
)
face_detector = vision.FaceDetector.create_from_options(options)

# Ekstraksi ROI dahi dari deteksi wajah
def extract_forehead_roi(frame, detection):
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    bboxC = detection.bounding_box
    x, y, w, h = bboxC.origin_x, bboxC.origin_y, bboxC.width, bboxC.height

    scale_w = 0.85
    scale_h = 1.1
    offset_y = 3
    new_w = int(w * scale_w)
    new_h = int(h * scale_h)
    center_x = int(x + w / 2)
    center_y = int(y + h * 0.35 + offset_y)
    new_x = max(0, center_x - new_w // 2)
    new_y = max(0, center_y - new_h // 2)
    new_x = min(new_x, frame.shape[1] - new_w)
    new_y = min(new_y, frame.shape[0] - new_h)
    roi = rgb_frame[new_y:new_y + new_h, new_x:new_x + new_w]
    return roi, (new_x, new_y, new_w, new_h)
