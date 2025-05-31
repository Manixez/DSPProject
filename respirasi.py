import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# Inisialisasi PoseLandmarker
model_path = "Model/pose_landmarker.task"
BaseOptions = mp.tasks.BaseOptions
PoseLandmarkerOptions = vision.PoseLandmarkerOptions
VisionRunningMode = vision.RunningMode

options = PoseLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.IMAGE,
    num_poses=1,
    min_pose_detection_confidence=0.5,
    min_pose_presence_confidence=0.5,
    min_tracking_confidence=0.5,
    output_segmentation_masks=False
)
pose_landmarker = vision.PoseLandmarker.create_from_options(options)

def get_respiration_roi(frame, scale_x=1.5, roi_height=120, shift_y=30, draw_bahu=True):
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    height, width = frame.shape[:2]

    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
    result = pose_landmarker.detect(mp_image)

    if not result.pose_landmarks:
        raise ValueError("Pose tidak terdeteksi.")

    landmarks = result.pose_landmarks[0]
    center_0 = landmarks[0]
    left_shoulder = landmarks[11]
    right_shoulder = landmarks[12]

    left_x = int(left_shoulder.x * width)
    right_x = int(right_shoulder.x * width)
    shoulder_width = abs(right_x - left_x)
    if shoulder_width < 20:
        raise ValueError("Pose gagal: bahu terlalu dekat")

    center_dada_x = (left_shoulder.x + right_shoulder.x) / 2
    center_dada_y = (left_shoulder.y + right_shoulder.y) / 2

    final_center_x = int(((center_0.x + center_dada_x) / 2) * width)
    final_center_y = int(((center_0.y + center_dada_y) / 2) * height) + shift_y

    roi_width = int(shoulder_width * scale_x)
    left_roi = max(0, final_center_x - roi_width // 2)
    right_roi = min(width, final_center_x + roi_width // 2)
    top_roi = max(0, final_center_y - roi_height // 2)
    bottom_roi = min(height, final_center_y + roi_height // 2)

    if draw_bahu:
        for pt in [left_shoulder, right_shoulder]:
            px = int(pt.x * width)
            py = int(pt.y * height)
            cv2.circle(frame, (px, py), 6, (0, 255, 0), -1)

    return (left_roi, top_roi, right_roi, bottom_roi)
