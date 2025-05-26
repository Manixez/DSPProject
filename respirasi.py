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

def get_respiration_roi(frame, scale_x=1.5, roi_height=120, shift_y=70, draw_bahu=True):
    """
    ROI respirasi dengan pusat dihitung dari gabungan landmark[0] (tengah tubuh)
    dan titik tengah antara bahu kiri-kanan (landmark 11 dan 12).

    Lebar ROI berdasarkan lebar bahu.
    Tinggi dan offset bisa dikendalikan.

    Parameters:
    - frame: input frame (BGR format)
    - scale_x: faktor skala lebar ROI berdasarkan lebar bahu
    - roi_height: tinggi ROI dalam pixel
    - shift_y: pergeseran vertikal dari titik tengah bahu
    - draw_bahu: apakah menggambar titik bahu untuk referensi

    Returns:
    - (left_x, top_y, right_x, bottom_y): koordinat ROI
    """
    try:
        # Konversi ke RGB untuk Mediapipe
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        height, width = frame.shape[:2]

        # Deteksi pose
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
        result = pose_landmarker.detect(mp_image)

        if not result.pose_landmarks or len(result.pose_landmarks) == 0:
            raise ValueError("Pose tidak terdeteksi")
        
        landmarks = result.pose_landmarks[0]

        # Landmark bahu kiri dan kanan
        left_shoulder = landmarks[11]
        right_shoulder = landmarks[12]

        # Validasi visibility landmarks
        if left_shoulder.visibility < 0.5 or right_shoulder.visibility < 0.5:
            raise ValueError("Landmark bahu tidak terlihat dengan jelas")

        # Koordinat bahu
        left_shoulder_x = int(left_shoulder_x * width)
        left_shoulder_y = int(left_shoulder_y * height)
        right_shoulder_x = int(right_shoulder.x * width)
        right_shoulder_y = int(right_shoulder.y * height)

        # Hitung lebar bahu
        shoulder_width = abs(right_shoulder_x - left_shoulder_x)
        if shoulder_width < 30:
            raise ValueError("Pose gagal: bahu terlalu dekat atau tidak terdeteksi dengan baik")
        
        # Titik tengah bahu
        center_x = int((left_shoulder_x + right_shoulder_x) / 2 * width)
        center_y = int((left_shoulder_y + right_shoulder_y) / 2 * height) + shift_y

        # Hitung dimensi ROI
        roi_width = int(shoulder_width * scale_x)

        # Pastikan ROI tidak keluar dari batas frame
        left_roi = max(0, center_x - roi_width // 2)
        right_roi = min(width, center_x + roi_width // 2)
        top_roi = max(0, center_y - roi_height // 2)
        bottom_roi = min(height, center_y + roi_height // 2)

        # Validasi ukuran ROI minimum
        if (right_roi - left_roi) < 50 or (bottom_roi - top_roi) < 50:
            raise ValueError("ROI terlalu kecil untuk dianalisis")
        
        # Tampilkan titik bahu untuk referensi visual
        if draw_bahu:
            cv2.circle(frame, (left_shoulder_x, left_shoulder_y), 5, (0, 255, 0), -1)
            cv2.circle(frame, (right_shoulder_x, right_shoulder_y), 5, (0, 255, 0), -1)
            cv2.circle(frame, (center_x, center_y - shift_y), 3, (255, 0, 0), -1)

        return (left_roi, top_roi, right_roi, bottom_roi)
    
    except Exception as e:
        print(f"Error dalam mendapatkan ROI respirasi: {e}")
        return