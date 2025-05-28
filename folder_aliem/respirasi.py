import cv2
import numpy as np
import time
from collections import deque
from scipy import signal
from scipy.fft import fft, fftfreq
import mediapipe as mp
import warnings
warnings.filterwarnings('ignore')

# MediaPipe Pose setup
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=1,
    enable_segmentation=False,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
mp_drawing = mp.solutions.drawing_utils

# Optical Flow Lucas-Kanade parameters
lk_params = dict(winSize=(15, 15),
                 maxLevel=2,
                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

# Feature detection parameters
feature_params = dict(maxCorners=100,
                     qualityLevel=0.3,
                     minDistance=7,
                     blockSize=7)

# Global variables untuk tracking
chest_points = None
chest_tracks = deque(maxlen=300)
prev_gray = None

# Respiratory signal processing
shoulder_landmarks = deque(maxlen=300)
respiratory_data = deque(maxlen=300)
time_stamps = deque(maxlen=300)
filtered_signal = deque(maxlen=300)
optical_flow_data = deque(maxlen=300)

# Parameters untuk filtering
fs = 30  # Sampling frequency (FPS)
lowcut = 0.1  # Hz
highcut = 0.8  # Hz

# Analysis parameters
method = "combined"  # "mediapipe", "optical_flow", "combined"

def reset_tracking():
    """Reset tracking points"""
    global chest_points, chest_tracks, optical_flow_data, shoulder_landmarks
    global respiratory_data, time_stamps, filtered_signal, prev_gray
    
    chest_points = None
    chest_tracks.clear()
    optical_flow_data.clear()
    shoulder_landmarks.clear()
    respiratory_data.clear()
    time_stamps.clear()
    filtered_signal.clear()
    prev_gray = None

def update_parameters(new_lowcut=None, new_highcut=None, quality=None, corners=None, new_method=None):
    """Update analysis parameters"""
    global lowcut, highcut, feature_params, method
    
    if new_lowcut is not None:
        lowcut = new_lowcut
    if new_highcut is not None:
        highcut = new_highcut
    if quality is not None:
        feature_params['qualityLevel'] = quality
    if corners is not None:
        feature_params['maxCorners'] = corners
    if new_method is not None:
        method = new_method

def extract_shoulder_landmarks(landmarks, frame):
    """Extract shoulder landmarks (left dan right shoulder saja)"""
    global shoulder_landmarks
    
    # Shoulder landmarks: left shoulder (11), right shoulder (12)
    left_shoulder_idx = 11
    right_shoulder_idx = 12
    
    h, w, _ = frame.shape
    shoulder_points = []
    
    # Extract left shoulder
    if left_shoulder_idx < len(landmarks.landmark):
        landmark = landmarks.landmark[left_shoulder_idx]
        x = int(landmark.x * w)
        y = int(landmark.y * h)
        shoulder_points.append([x, y])
        
        # Draw left shoulder point
        cv2.circle(frame, (x, y), 8, (255, 0, 0), -1)
        cv2.putText(frame, 'L-Shoulder', (x-30, y-10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
    
    # Extract right shoulder
    if right_shoulder_idx < len(landmarks.landmark):
        landmark = landmarks.landmark[right_shoulder_idx]
        x = int(landmark.x * w)
        y = int(landmark.y * h)
        shoulder_points.append([x, y])
        
        # Draw right shoulder point
        cv2.circle(frame, (x, y), 8, (0, 0, 255), -1)
        cv2.putText(frame, 'R-Shoulder', (x-30, y-10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    
    if len(shoulder_points) == 2:
        # Hitung jarak antara kedua bahu
        shoulder_points = np.array(shoulder_points)
        left_shoulder = shoulder_points[0]
        right_shoulder = shoulder_points[1]
        
        # Hitung midpoint antara kedua bahu
        midpoint = (left_shoulder + right_shoulder) / 2
        
        # Hitung jarak antara kedua bahu sebagai indikator respirasi
        shoulder_distance = np.linalg.norm(right_shoulder - left_shoulder)
        
        # Hitung pergerakan midpoint jika ada data sebelumnya
        movement = 0
        if len(shoulder_landmarks) > 0:
            prev_midpoint = shoulder_landmarks[-1]['midpoint']
            movement = np.linalg.norm(midpoint - prev_midpoint)
        
        # Store shoulder data
        shoulder_data = {
            'left_shoulder': left_shoulder,
            'right_shoulder': right_shoulder,
            'midpoint': midpoint,
            'distance': shoulder_distance,
            'movement': movement
        }
        shoulder_landmarks.append(shoulder_data)
        
        # Draw line between shoulders
        cv2.line(frame, tuple(left_shoulder.astype(int)), tuple(right_shoulder.astype(int)), 
                (0, 255, 0), 3)
        
        # Draw midpoint
        cv2.circle(frame, tuple(midpoint.astype(int)), 6, (0, 255, 0), -1)
        cv2.putText(frame, 'Shoulder Center', (int(midpoint[0])-50, int(midpoint[1])+20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Return combined signal (distance variation + movement)
        return shoulder_distance + movement * 10  # Scale movement
    
    return None

def process_optical_flow(gray_frame, display_frame):
    """Process Optical Flow Lucas-Kanade"""
    global prev_gray, chest_points, feature_params, chest_tracks, optical_flow_data
    
    if prev_gray is None:
        prev_gray = gray_frame.copy()
        return None
    
    # Initialize atau update tracking points
    if chest_points is None or len(chest_points) < 10:
        # Detect good features to track di area chest (fokus area antara bahu)
        h, w = gray_frame.shape
        chest_mask = np.zeros_like(gray_frame)
        
        # Jika ada shoulder landmarks, fokus area di antara bahu
        if len(shoulder_landmarks) > 0:
            shoulder_data = shoulder_landmarks[-1]
            left_shoulder = shoulder_data['left_shoulder'].astype(int)
            right_shoulder = shoulder_data['right_shoulder'].astype(int)
            midpoint = shoulder_data['midpoint'].astype(int)
            
            # Buat ROI berdasarkan posisi bahu
            roi_width = int(abs(right_shoulder[0] - left_shoulder[0]) * 1.2)
            roi_height = int(roi_width * 0.8)
            
            x1 = max(0, midpoint[0] - roi_width//2)
            y1 = max(0, midpoint[1] - roi_height//4)
            x2 = min(w, midpoint[0] + roi_width//2)
            y2 = min(h, midpoint[1] + roi_height*3//4)
            
            chest_mask[y1:y2, x1:x2] = 255
            
            # Draw ROI
            cv2.rectangle(display_frame, (x1, y1), (x2, y2), (255, 255, 0), 2)
            cv2.putText(display_frame, 'OF ROI', (x1, y1-5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        else:
            # Default ROI jika belum ada shoulder detection
            chest_roi = chest_mask[h//3:2*h//3, w//3:2*w//3]
            chest_roi[:] = 255
        
        corners = cv2.goodFeaturesToTrack(gray_frame, mask=chest_mask, **feature_params)
        if corners is not None:
            chest_points = corners
    
    if chest_points is not None and len(chest_points) > 0:
        # Calculate optical flow
        new_points, status, error = cv2.calcOpticalFlowPyrLK(
            prev_gray, gray_frame, chest_points, None, **lk_params)
        
        # Select good points
        if new_points is not None:
            good_new = new_points[status == 1]
            good_old = chest_points[status == 1]
            
            if len(good_new) > 5:  # Minimal points untuk tracking
                # Calculate movement vectors
                movement_vectors = good_new - good_old
                
                # Hitung rata-rata magnitude pergerakan
                movements = np.linalg.norm(movement_vectors, axis=1)
                avg_movement = np.mean(movements)
                
                # Filter outliers
                valid_movements = movements[movements < np.percentile(movements, 95)]
                if len(valid_movements) > 0:
                    avg_movement = np.mean(valid_movements)
                
                # Draw tracking points dan vectors
                for i, (new, old) in enumerate(zip(good_new, good_old)):
                    a, b = new.ravel().astype(int)
                    c, d = old.ravel().astype(int)
                    
                    cv2.circle(display_frame, (a, b), 3, (0, 255, 255), -1)
                    cv2.line(display_frame, (a, b), (c, d), (0, 255, 255), 1)
                
                # Update tracking points
                chest_points = good_new.reshape(-1, 1, 2)
                
                # Store tracking data
                chest_tracks.append(avg_movement)
                optical_flow_data.append(avg_movement)
                
                cv2.putText(display_frame, f'OF Points: {len(good_new)}', (10, 90), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                
                return avg_movement
            else:
                # Re-detect features jika terlalu sedikit
                chest_points = None
    
    return None

def combine_signals(mediapipe_data, optical_flow_data):
    """Combine MediaPipe dan Optical Flow signals"""
    global method
    
    if method == "mediapipe":
        return mediapipe_data
    elif method == "optical_flow":
        return optical_flow_data
    elif method == "combined":
        if mediapipe_data is not None and optical_flow_data is not None:
            # Weighted combination (bisa disesuaikan)
            return 0.7 * mediapipe_data + 0.3 * optical_flow_data
        elif mediapipe_data is not None:
            return mediapipe_data
        elif optical_flow_data is not None:
            return optical_flow_data
    
    return None

def apply_filters():
    """Apply bandpass filter untuk sinyal respirasi"""
    global lowcut, highcut, fs, respiratory_data
    
    if len(respiratory_data) < 60:
        return respiratory_data[-1] if respiratory_data else 0
    
    # Convert ke numpy array
    data = np.array(list(respiratory_data))
    
    # Bandpass filter
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    
    try:
        b, a = signal.butter(4, [low, high], btype='band')
        filtered_data = signal.filtfilt(b, a, data)
        return filtered_data[-1]
    except:
        return data[-1]

def analyze_respiratory_pattern():
    """Analisis pola respirasi"""
    global filtered_signal, fs, method, shoulder_landmarks, optical_flow_data
    
    if len(filtered_signal) < 90:  # 3 detik data
        return None
    
    # Convert ke numpy array
    data = np.array(list(filtered_signal))
    
    # FFT untuk analisis frekuensi
    fft_data = fft(data)
    freqs = fftfreq(len(data), 1/fs)
    
    # Cari frekuensi dominan dalam range respirasi (0.1-0.8 Hz)
    valid_indices = (freqs >= 0.1) & (freqs <= 0.8)
    if np.any(valid_indices):
        dominant_freq_idx = np.argmax(np.abs(fft_data[valid_indices]))
        dominant_freq = freqs[valid_indices][dominant_freq_idx]
        
        # Konversi ke respirasi per menit
        respiratory_rate = dominant_freq * 60
        
        # Peak detection untuk menghitung siklus
        peaks, properties = signal.find_peaks(data, height=np.mean(data), distance=15)
        if len(peaks) > 1:
            avg_interval = np.mean(np.diff(peaks)) / fs
            calculated_rate = 60 / avg_interval if avg_interval > 0 else 0
            
            # Analisis kualitas sinyal
            signal_quality = assess_signal_quality(data, peaks)
            
            # Get shoulder info
            shoulder_info = {}
            if len(shoulder_landmarks) > 0:
                latest_shoulder = shoulder_landmarks[-1]
                shoulder_info = {
                    'distance': latest_shoulder['distance'],
                    'movement': latest_shoulder['movement']
                }
            
            # Return hasil analisis
            result = {
                'method': method.upper(),
                'respiratory_rate_fft': respiratory_rate,
                'respiratory_rate_peaks': calculated_rate,
                'dominant_frequency': dominant_freq,
                'num_peaks': len(peaks),
                'signal_quality': signal_quality,
                'shoulder_info': shoulder_info,
                'shoulder_landmarks_count': len(shoulder_landmarks),
                'optical_flow_points': len(optical_flow_data)
            }
            
            return result
    
    return None

def assess_signal_quality(data, peaks):
    """Assess kualitas sinyal respirasi"""
    if len(peaks) < 2:
        return "Poor - Too few peaks"
    
    # Hitung regularity dari peak intervals
    intervals = np.diff(peaks)
    if len(intervals) > 1:
        cv = np.std(intervals) / np.mean(intervals)  # Coefficient of variation
        if cv < 0.2:
            return "Excellent"
        elif cv < 0.4:
            return "Good"
        elif cv < 0.6:
            return "Fair"
        else:
            return "Poor - Irregular"
    
    return "Fair"

def display_info_on_frame(frame):
    """Display informasi pada frame"""
    global fs, method, respiratory_data, filtered_signal, shoulder_landmarks, optical_flow_data
    
    # FPS dan status
    cv2.putText(frame, f'FPS: {int(fs)}', (10, 30), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    cv2.putText(frame, f'Method: {method}', (10, 60), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    # Current signal values
    if len(respiratory_data) > 0:
        cv2.putText(frame, f'Respiratory: {respiratory_data[-1]:.3f}', 
                   (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    if len(filtered_signal) > 0:
        cv2.putText(frame, f'Filtered: {filtered_signal[-1]:.3f}', 
                   (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    # Shoulder info
    if len(shoulder_landmarks) > 0:
        latest_shoulder = shoulder_landmarks[-1]
        cv2.putText(frame, f'Shoulder Dist: {latest_shoulder["distance"]:.1f}px', 
                   (10, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    
    cv2.putText(frame, f'OF Points: {len(optical_flow_data)}', 
               (10, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

def process_frame(frame, current_time):
    """Process single frame untuk analisis respirasi"""
    global pose, prev_gray, respiratory_data, time_stamps, filtered_signal
    
    # Convert BGR to RGB untuk MediaPipe
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # MediaPipe Pose detection (fokus shoulder saja)
    results = pose.process(rgb_frame)
    
    # Extract shoulder landmarks dari MediaPipe
    shoulder_signal = None
    if results.pose_landmarks:
        shoulder_signal = extract_shoulder_landmarks(results.pose_landmarks, frame)
    
    # Optical Flow Lucas-Kanade
    optical_flow_signal = process_optical_flow(gray_frame, frame)
    
    # Combine signals berdasarkan metode yang dipilih
    respiratory_signal = combine_signals(shoulder_signal, optical_flow_signal)
    
    analysis_result = None
    if respiratory_signal is not None:
        respiratory_data.append(respiratory_signal)
        time_stamps.append(current_time)
        
        # Apply filtering
        if len(respiratory_data) > 30:
            filtered_value = apply_filters()
            filtered_signal.append(filtered_value)
            
            # Analisis respirasi
            if len(filtered_signal) > 90:  # 3 detik data
                analysis_result = analyze_respiratory_pattern()
    
    # Display info pada frame
    display_info_on_frame(frame)
    
    # Update previous frame untuk optical flow
    prev_gray = gray_frame.copy()
    
    return analysis_result