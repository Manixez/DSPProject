import cv2
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
import mediapipe as mp
from rPPG import POS, extract_forehead_roi, face_detector
from respirasi import get_respiration_roi
from scipy.signal import butter, filtfilt, find_peaks
import threading
import time
import sys

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
monitoring_active = True

def normalize(x):
    x = np.array(x)
    if len(x) == 0:
        return x
    std_x = np.std(x)
    if std_x == 0:
        return np.zeros_like(x)
    return (x - np.mean(x)) / std_x

def bandpass_filter(data, fs, lowcut, highcut, order=3):
    try:
        nyquist = fs / 2
        low = lowcut / nyquist
        high = highcut / nyquist

        if high >= 1.0:
            high = 0.99
        if low <= 0 :
            low = 0.01

        b, a = butter(order, [lowcut, highcut], btype='band', fs=fs)
        return filtfilt(b, a, data)
    except Exception as e:
        print(f"Error in bandpass_filter: {e}")
        return data

def estimate_rate(signal, fs, lowcut, highcut):
    try:
        if len(signal) < fs * 3:
            return 0
        
        signal = np.array(signal)
        filtered = bandpass_filter(signal, fs, lowcut, highcut)

        min_distance = int(fs * 60 / 120)
        peaks, properties = find_peaks(filtered, distance=min_distance, height=np.std(filtered) * 0.3)
        
        if len(peaks) <2:
            return 0
        
        duration = len(filtered) / fs
        rate = 60 * len(peaks) / duration

        if lowcut == 0.7:
            if rate < 40 or rate > 180:
                return 0
        else:
            if rate < 8 or rate > 40:
                return 0
        
        return rate
    except Exception as e:
        print(f"Error in estimate_rate: {e}")
        return 0

def update_metrics():
    global hr, rr, monitoring_active
    while monitoring_active:
        try:
            if len(r_signal) > 60:
                rgb = np.array([r_signal, g_signal, b_signal]).reshape(1, 3, -1)
                try:
                    rppg = POS(rgb, fps=fps).reshape(-1)
                    hr = estimate_rate(rppg, fs=fps, lowcut=0.7, highcut=3.0)
                except Exception as e:
                    print(f"Kalkulasi rPPG error: {e}")
                    hr = 0
            
            if len(resp_signal) > 60:
                rr = estimate_rate(resp_signal, fs=fps, lowcut=0.1, highcut=0.7)
            
            time.sleep(3)
        
        except Exception as e:
            print(f"Error pada update_metrics: {e}")
            time.sleep(1)

def run_main():
    global features, old_gray, resp_roi_coords, monitoring_active, hr, rr

    features = None
    old_gray = None
    resp_roi_coords = None
    hr, rr = 0, 0
    monitoring_active = True

    r_signal.clear()
    g_signal.clear()
    b_signal.clear()
    resp_signal.clear()

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Tidak dapat membuka kamera.")
        return
    
    # Set camera properties
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, fps)
    
    # Start metrics update thread
    metrics_thread = threading.Thread(target=update_metrics, daemon=True)
    metrics_thread.start()

    # Setup matplotlib for real-time plotting
    plt.ion()
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6))
    line1, = ax1.plot([], [], 'b-', label='Respirasi')
    line2, = ax2.plot([], [], 'r-', label='rPPG')
    
    ax1.set_ylim(-3, 3)
    ax2.set_ylim(-3, 3)
    ax1.set_xlim(0, window_size)
    ax2.set_xlim(0, window_size)
    ax1.legend()
    ax2.legend()
    ax1.grid(True)
    ax2.grid(True)
    
    print("Monitoring started. Press 'q' to quit.")
    
    try:
        while monitoring_active:
            ret, frame = cap.read()
            if not ret:
                print("Error: Could not read frame")
                break
            
            # Flip frame horizontally for mirror effect
            frame = cv2.flip(frame, 1)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # rPPG ROI (forehead) detection using BlazeFace
            try:
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
                result = face_detector.detect(mp_image)
                
                if result.detections:
                    roi, (x, y, w, h) = extract_forehead_roi(frame, result.detections[0])
                    if roi is not None and roi.size > 0:
                        mean_rgb = cv2.mean(roi)[:3]
                        r_signal.append(mean_rgb[2])  # BGR to RGB
                        g_signal.append(mean_rgb[1])
                        b_signal.append(mean_rgb[0])
                        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                        cv2.putText(frame, "Face ROI", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            except Exception as e:
                print(f"Face detection error: {e}")

            # Respiration ROI detection using pose landmarks
            try:
                lx, ty, rx, by = get_respiration_roi(frame, scale_x=1.5, roi_height=120, shift_y=70)
                resp_roi_coords = (lx, ty, rx, by)
                cv2.rectangle(frame, (lx, ty), (rx, by), (0, 0, 255), 2)
                cv2.putText(frame, "Chest ROI", (lx, ty-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            except Exception as e:
                # If pose detection fails, skip this frame
                pass

            # Optical flow tracking for respiration
            if resp_roi_coords:
                lx, ty, rx, by = resp_roi_coords
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                roi_gray = gray[ty:by, lx:rx]

                if features is None:
                    # Initialize feature points
                    features = cv2.goodFeaturesToTrack(roi_gray, maxCorners=50, 
                                                     qualityLevel=0.2, minDistance=5, blockSize=3)
                    if features is not None and len(features) > 0:
                        features = np.float32(features)
                        # Convert to global coordinates
                        features[:, :, 0] += lx
                        features[:, :, 1] += ty
                        old_gray = gray.copy()

                elif old_gray is not None and features is not None and len(features) > 0:
                    # Track features using Lucas-Kanade optical flow
                    new_features, status, _ = cv2.calcOpticalFlowPyrLK(old_gray, gray, features, None, **lk_params)
                    
                    # Select good points
                    good_new = new_features[status == 1]
                    
                    if len(good_new) > 5:  # Need minimum points for reliable tracking
                        # Calculate average Y movement (breathing motion)
                        avg_y = np.mean(good_new[:, 1])
                        resp_signal.append(avg_y)
                        
                        # Draw tracked points
                        for point in good_new:
                            cv2.circle(frame, (int(point[0]), int(point[1])), 2, (255, 255, 0), -1)
                        
                        # Update features
                        features = good_new.reshape(-1, 1, 2)
                    else:
                        # Reset if too few points
                        features = None
                    
                    old_gray = gray.copy()

            # Update signal plots
            if len(resp_signal) > 1:
                line1.set_ydata(normalize(list(resp_signal)))
                line1.set_xdata(np.arange(len(resp_signal)))
                ax1.relim()
                ax1.autoscale_view()
            
            if len(g_signal) > 1:
                line2.set_ydata(normalize(list(g_signal)))
                line2.set_xdata(np.arange(len(g_signal)))
                ax2.relim()
                ax2.autoscale_view()
            
            ax1.set_title(f"Respiration Signal (RR ~ {rr:.1f} bpm)")
            ax2.set_title(f"rPPG Signal (HR ~ {hr:.1f} bpm)")
            
            plt.pause(0.001)

            # Display monitoring info on frame
            cv2.putText(frame, f"HR: {hr:.1f} bpm", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.putText(frame, f"RR: {rr:.1f} bpm", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            cv2.putText(frame, f"Signals: R={len(r_signal)}, Resp={len(resp_signal)}", (10, 90), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            # Show frame
            cv2.imshow("Real-time Monitoring", frame)
            
            # Check for exit conditions
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:  # 'q' or ESC
                break
            
            # Check if window is closed
            if cv2.getWindowProperty("Real-time Monitoring", cv2.WND_PROP_VISIBLE) < 1:
                break

    except KeyboardInterrupt:
        print("Monitoring interrupted by user")
    except Exception as e:
        print(f"Error in main loop: {e}")
    finally:
        # Cleanup
        monitoring_active = False
        cap.release()
        cv2.destroyAllWindows()
        plt.ioff()
        plt.close('all')
        print("Monitoring stopped")