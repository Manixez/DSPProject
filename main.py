import cv2
import threading
import time
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import mediapipe as mp
from collections import deque
from rPPG import extract_forehead_roi, face_detector
from respirasi import get_respiration_roi
from signal_processing import estimate_bpm

# Global states
fps = 30
buffer_len = 300

rgb_buffer = deque(maxlen=buffer_len)  # Untuk POS
r_signal = deque(maxlen=buffer_len)
g_signal = deque(maxlen=buffer_len)
b_signal = deque(maxlen=buffer_len)
resp_signal = deque(maxlen=buffer_len)

features = None
old_gray = None
hr = 0.0
rr = 0.0
monitoring_active = True
resp_roi_coords = None
frame_display = None  # Untuk GUI

lk_params = dict(winSize=(15, 15), maxLevel=2,
                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

def apply_pos(rgb_matrix):
    mean_centered = rgb_matrix - np.mean(rgb_matrix, axis=0)
    S = np.array([[0, 1, -1], [-2, 1, 1]])
    X = np.dot(S, mean_centered.T)
    h = X[0] + X[1]
    return h

def update_metrics():
    global hr, rr
    while monitoring_active:
        if len(rgb_buffer) >= 100:
            rgb_np = np.array(rgb_buffer)
            h = apply_pos(rgb_np)
            hr, _ = estimate_bpm(h, fs=fps, lowcut=0.7, highcut=3.0)
        if len(resp_signal) >= 100:
            rr, _ = estimate_bpm(list(resp_signal), fs=fps, lowcut=0.1, highcut=0.7)
        time.sleep(2)

def run_main():
    global features, old_gray, resp_roi_coords, monitoring_active, frame_display
    cap = cv2.VideoCapture(0)
    threading.Thread(target=update_metrics, daemon=True).start()

    while monitoring_active and cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # rPPG ROI
        mp_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = face_detector.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=mp_image))
        if result.detections:
            roi, (x, y, w, h) = extract_forehead_roi(frame, result.detections[0])
            if roi.size > 0:
                mean_rgb = cv2.mean(roi)[:3]
                rgb_buffer.append(mean_rgb)
                r_signal.append(mean_rgb[0])
                g_signal.append(mean_rgb[1])
                b_signal.append(mean_rgb[2])
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Respirasi ROI
        try:
            lx, ty, rx, by = get_respiration_roi(frame)
            resp_roi_coords = (lx, ty, rx, by)
            cv2.rectangle(frame, (lx, ty), (rx, by), (0, 0, 255), 2)
        except:
            continue

        # Optical Flow
        if resp_roi_coords:
            rx, ry, rrx, rby = resp_roi_coords
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            roi_gray = gray[ry:rby, rx:rrx]

            if features is None:
                features = cv2.goodFeaturesToTrack(roi_gray, maxCorners=50, qualityLevel=0.2, minDistance=5, blockSize=3)
                if features is not None:
                    features = np.float32(features)
                    features[:, :, 0] += rx
                    features[:, :, 1] += ry
                    old_gray = gray.copy()

            elif old_gray is not None and features is not None:
                new_features, status, _ = cv2.calcOpticalFlowPyrLK(old_gray, gray, features, None, **lk_params)
                good_new = new_features[status == 1]
                if len(good_new) > 0:
                    avg_y = np.mean(good_new[:, 1])
                    resp_signal.append(avg_y)
                    features = good_new.reshape(-1, 1, 2)
                    old_gray = gray.copy()

        frame_display = frame.copy()

    cap.release()
    cv2.destroyAllWindows()
