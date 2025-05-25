import cv2
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
from rPPG import POS, extract_forehead_roi, face_detector
from respirasi import get_respiration_roi
import mediapipe as mp
from scipy.signal import butter, filtfilt, find_peaks
import threading
import time

# Parameter umum
fps = 30
window_size = 300

# Buffer sinyal
r_signal, g_signal, b_signal = deque(maxlen=window_size), deque(maxlen=window_size), deque(maxlen=window_size)
resp_signal = deque(maxlen=window_size)

# Optical Flow
features = None
lk_params = dict(winSize=(15, 15), maxLevel=2,
                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
old_gray = None
hr, rr = 0, 0
resp_roi_coords = None

def normalize(x):
    x = np.array(x)
    return (x - np.mean(x)) / (np.std(x) + 1e-6)

def bandpass_filter(data, fs, lowcut, highcut, order=3):
    b, a = butter(order, [lowcut, highcut], btype='band', fs=fs)
    return filtfilt(b, a, data)

def estimate_rate(signal, fs, lowcut, highcut):
    if len(signal) < fs * 3:
        return 0
    filtered = bandpass_filter(signal, fs, lowcut, highcut)
    peaks, _ = find_peaks(filtered, distance=fs//2)
    duration = len(filtered) / fs
    return 60 * len(peaks) / duration

def update_metrics():
    global hr, rr
    while True:
        if len(r_signal) > 60:
            rgb = np.array([r_signal, g_signal, b_signal]).reshape(1, 3, -1)
            rppg = POS(rgb, fps=fps).reshape(-1)
            hr = estimate_rate(rppg, fs=fps, lowcut=0.7, highcut=3.0)
        if len(resp_signal) > 60:
            rr = estimate_rate(resp_signal, fs=fps, lowcut=0.1, highcut=0.7)
        time.sleep(5)

def run_main():
    global features, old_gray, resp_roi_coords
    cap = cv2.VideoCapture(0)
    threading.Thread(target=update_metrics, daemon=True).start()

    plt.ion()
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6))
    line1, = ax1.plot([], [], label='Respirasi')
    line2, = ax2.plot([], [], label='rPPG')
    ax1.set_ylim(-2, 2)
    ax2.set_ylim(-2, 2)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # rPPG ROI (wajah) dengan BlazeFace
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        result = face_detector.detect(mp_image)
        if result.detections:
            roi, (x, y, w, h) = extract_forehead_roi(frame, result.detections[0])
            mean_rgb = cv2.mean(roi)[:3]
            r_signal.append(mean_rgb[0])
            g_signal.append(mean_rgb[1])
            b_signal.append(mean_rgb[2])
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        # ROI respirasi dari pose landmark[0]
        try:
            lx, ty, rx, by = get_respiration_roi(frame, scale_x=1.5, roi_height=120, shift_y=70)
            resp_roi_coords = (lx, ty, rx, by)
            cv2.rectangle(frame, (lx, ty), (rx, by), (0, 0, 255), 2)
        except:
            continue

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

        # Visualisasi sinyal
        line1.set_ydata(normalize(resp_signal))
        line1.set_xdata(np.arange(len(resp_signal)))
        line2.set_ydata(normalize(g_signal))
        line2.set_xdata(np.arange(len(g_signal)))
        ax1.relim(); ax2.relim()
        ax1.autoscale_view(); ax2.autoscale_view()
        ax1.set_title(f"Respiration (RR ~ {rr:.1f} bpm)")
        ax2.set_title(f"rPPG (HR ~ {hr:.1f} bpm)")
        plt.pause(0.001)

        cv2.imshow("Monitoring", frame)
        if cv2.getWindowProperty("Monitoring", cv2.WND_PROP_VISIBLE) < 1:
            break

    cap.release()
    cv2.destroyAllWindows()
    plt.ioff()
    plt.close()
