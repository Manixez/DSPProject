# --- rppg_processor.py ---
import numpy as np
import cv2
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

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
