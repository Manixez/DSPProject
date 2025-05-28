import cv2
import mediapipe as mp
import numpy as np
import time
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from scipy import signal
from scipy.signal import butter, filtfilt
import warnings
warnings.filterwarnings('ignore')

class RespirationDetector:
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            enable_segmentation=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.mp_drawing = mp.solutions.drawing_utils
        
        # Data storage
        self.left_shoulder_data = []
        self.right_shoulder_data = []
        self.timestamps = []
        self.breathing_signal = []
        
        # Recording parameters
        self.duration = 30  # 30 seconds
        self.start_time = None
        self.recording = False
        
    def calculate_distance(self, point1, point2):
        """Menghitung jarak euclidean antara dua titik"""
        return np.sqrt((point1.x - point2.x)**2 + (point1.y - point2.y)**2)
    
    def butter_bandpass_filter(self, data, lowcut=0.1, highcut=0.5, fs=30, order=4):
        """Filter bandpass untuk sinyal pernapasan"""
        nyquist = 0.5 * fs
        low = lowcut / nyquist
        high = highcut / nyquist
        b, a = butter(order, [low, high], btype='band')
        filtered_data = filtfilt(b, a, data)
        return filtered_data
    
    def moving_average_filter(self, data, window_size=5):
        """Filter moving average untuk menghaluskan sinyal"""
        if len(data) < window_size:
            return data
        return np.convolve(data, np.ones(window_size)/window_size, mode='same')
    
    def detect_breathing_rate(self, signal_data, sample_rate=30):
        """Mendeteksi laju pernapasan dari sinyal"""
        if len(signal_data) < sample_rate:
            return 0
        
        # Menggunakan FFT untuk mencari frekuensi dominan
        fft = np.fft.fft(signal_data)
        freqs = np.fft.fftfreq(len(signal_data), 1/sample_rate)
        
        # Fokus pada rentang frekuensi pernapasan normal (0.1 - 0.5 Hz atau 6-30 BPM)
        valid_idx = (freqs > 0.1) & (freqs < 0.5)
        if np.any(valid_idx):
            dominant_freq = freqs[valid_idx][np.argmax(np.abs(fft[valid_idx]))]
            breathing_rate = dominant_freq * 60  # Konversi ke BPM
            return max(0, breathing_rate)
        return 0
    
    def process_frame(self, frame):
        """Memproses frame untuk deteksi pose"""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(rgb_frame)
        
        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            
            # Dapatkan koordinat bahu kiri dan kanan
            left_shoulder = landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER]
            right_shoulder = landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER]
            
            if self.recording:
                current_time = time.time() - self.start_time
                
                # Simpan data koordinat bahu
                self.left_shoulder_data.append([left_shoulder.x, left_shoulder.y])
                self.right_shoulder_data.append([right_shoulder.x, right_shoulder.y])
                self.timestamps.append(current_time)
                
                # Hitung sinyal pernapasan (jarak antara kedua bahu)
                shoulder_distance = self.calculate_distance(left_shoulder, right_shoulder)
                self.breathing_signal.append(shoulder_distance)
            
            # Gambar landmark pose
            self.mp_drawing.draw_landmarks(
                frame, results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS)
            
            # Highlight bahu kiri dan kanan
            h, w, _ = frame.shape
            left_x, left_y = int(left_shoulder.x * w), int(left_shoulder.y * h)
            right_x, right_y = int(right_shoulder.x * w), int(right_shoulder.y * h)
            
            cv2.circle(frame, (left_x, left_y), 8, (0, 255, 0), -1)  # Hijau untuk bahu kiri
            cv2.circle(frame, (right_x, right_y), 8, (0, 0, 255), -1)  # Merah untuk bahu kanan
            
            # Tambahkan label
            cv2.putText(frame, 'L', (left_x-10, left_y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, 'R', (right_x+10, right_y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        return frame
    
    def generate_pdf_report(self, filename='hasil.pdf'):
        """Membuat laporan PDF dengan analisis sinyal"""
        if len(self.breathing_signal) == 0:
            print("Tidak ada data untuk dianalisis")
            return
        
        # Konversi ke numpy array
        timestamps = np.array(self.timestamps)
        raw_signal = np.array(self.breathing_signal)
        
        # Proses filtering
        smoothed_signal = self.moving_average_filter(raw_signal, window_size=5)
        
        if len(raw_signal) > 60:  # Minimal data untuk filtering
            filtered_signal = self.butter_bandpass_filter(smoothed_signal)
        else:
            filtered_signal = smoothed_signal
        
        # Deteksi laju pernapasan
        breathing_rate = self.detect_breathing_rate(filtered_signal)
        
        # Buat PDF
        with PdfPages(filename) as pdf:
            # Halaman 1: Sinyal Mentah
            fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10))
            
            # Plot sinyal mentah
            ax1.plot(timestamps, raw_signal, 'b-', linewidth=1, alpha=0.7)
            ax1.set_title('Sinyal Pernapasan Mentah', fontsize=14, fontweight='bold')
            ax1.set_xlabel('Waktu (detik)')
            ax1.set_ylabel('Jarak Bahu (normalized)')
            ax1.grid(True, alpha=0.3)
            ax1.set_xlim(0, 30)
            
            # Plot sinyal setelah moving average
            ax2.plot(timestamps, smoothed_signal, 'g-', linewidth=1.5)
            ax2.set_title('Sinyal Setelah Moving Average Filter', fontsize=14, fontweight='bold')
            ax2.set_xlabel('Waktu (detik)')
            ax2.set_ylabel('Jarak Bahu (normalized)')
            ax2.grid(True, alpha=0.3)
            ax2.set_xlim(0, 30)
            
            # Plot sinyal setelah bandpass filter
            ax3.plot(timestamps, filtered_signal, 'r-', linewidth=2)
            ax3.set_title('Sinyal Setelah Bandpass Filter (0.1-0.5 Hz)', fontsize=14, fontweight='bold')
            ax3.set_xlabel('Waktu (detik)')
            ax3.set_ylabel('Jarak Bahu (normalized)')
            ax3.grid(True, alpha=0.3)
            ax3.set_xlim(0, 30)
            
            plt.tight_layout()
            pdf.savefig(fig, dpi=300, bbox_inches='tight')
            plt.close()
            
            # Halaman 2: Analisis Spektral
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
            
            # FFT Analysis
            if len(filtered_signal) > 1:
                fft = np.fft.fft(filtered_signal)
                freqs = np.fft.fftfreq(len(filtered_signal), 1/30)  # 30 FPS
                
                # Plot hanya frekuensi positif
                positive_freqs = freqs[:len(freqs)//2]
                positive_fft = np.abs(fft[:len(fft)//2])
                
                ax1.plot(positive_freqs * 60, positive_fft)  # Konversi ke cycles per minute
                ax1.set_title('Analisis Spektral Sinyal Pernapasan', fontsize=14, fontweight='bold')
                ax1.set_xlabel('Frekuensi (cycles per minute)')
                ax1.set_ylabel('Amplitudo')
                ax1.set_xlim(0, 60)
                ax1.grid(True, alpha=0.3)
                
                # Highlight area pernapasan normal (6-30 BPM)
                ax1.axvspan(6, 30, alpha=0.2, color='green', label='Range Normal (6-30 BPM)')
                ax1.legend()
            
            # Perbandingan sinyal
            ax2.plot(timestamps, raw_signal, 'b-', alpha=0.5, label='Sinyal Mentah', linewidth=1)
            ax2.plot(timestamps, filtered_signal, 'r-', label='Sinyal Filtered', linewidth=2)
            ax2.set_title('Perbandingan Sinyal Mentah vs Filtered', fontsize=14, fontweight='bold')
            ax2.set_xlabel('Waktu (detik)')
            ax2.set_ylabel('Amplitudo')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            ax2.set_xlim(0, 30)
            
            plt.tight_layout()
            pdf.savefig(fig, dpi=300, bbox_inches='tight')
            plt.close()
            
            # Halaman 3: Statistik dan Ringkasan
            fig, ax = plt.subplots(figsize=(12, 8))
            ax.axis('off')
            
            # Hitung statistik
            mean_raw = np.mean(raw_signal)
            std_raw = np.std(raw_signal)
            mean_filtered = np.mean(filtered_signal)
            std_filtered = np.std(filtered_signal)
            
            # Teks ringkasan
            summary_text = f"""
LAPORAN ANALISIS PERNAPASAN

PARAMETER RECORDING:
• Durasi: {len(timestamps):.1f} detik
• Sample Rate: ~{len(timestamps)/30:.1f} FPS
• Total Samples: {len(timestamps)}

ANALISIS SINYAL MENTAH:
• Mean: {mean_raw:.6f}
• Standard Deviation: {std_raw:.6f}
• Range: {np.min(raw_signal):.6f} - {np.max(raw_signal):.6f}

ANALISIS SINYAL FILTERED:
• Mean: {mean_filtered:.6f}
• Standard Deviation: {std_filtered:.6f}
• Range: {np.min(filtered_signal):.6f} - {np.max(filtered_signal):.6f}

ESTIMASI LAJU PERNAPASAN:
• Breathing Rate: {breathing_rate:.1f} BPM
• Status: {'Normal' if 12 <= breathing_rate <= 20 else 'Perlu Perhatian'}

METODE FILTERING:
1. Moving Average Filter (Window: 5)
2. Butterworth Bandpass Filter (0.1-0.5 Hz, Order: 4)

DETEKSI LANDMARK:
• Bahu Kiri (LEFT_SHOULDER): MediaPipe Landmark
• Bahu Kanan (RIGHT_SHOULDER): MediaPipe Landmark
• Sinyal: Jarak Euclidean antara kedua bahu
            """
            
            ax.text(0.05, 0.95, summary_text, transform=ax.transAxes, fontsize=12,
                   verticalalignment='top', fontfamily='monospace',
                   bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray"))
            
            plt.tight_layout()
            pdf.savefig(fig, dpi=300, bbox_inches='tight')
            plt.close()
        
        print(f"Laporan PDF telah disimpan sebagai '{filename}'")
        print(f"Laju pernapasan estimasi: {breathing_rate:.1f} BPM")
    
    def run(self):
        """Menjalankan deteksi pernapasan"""
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("Error: Tidak dapat membuka kamera")
            return
        
        print("Tekan SPACE untuk memulai recording 30 detik")
        print("Tekan 'q' untuk keluar")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Tidak dapat membaca frame")
                break
            
            # Flip frame horizontal untuk efek mirror
            frame = cv2.flip(frame, 1)
            
            # Proses frame
            frame = self.process_frame(frame)
            
            # Tampilkan status
            if not self.recording:
                cv2.putText(frame, 'Tekan SPACE untuk mulai recording', 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            else:
                elapsed = time.time() - self.start_time
                remaining = max(0, self.duration - elapsed)
                cv2.putText(frame, f'Recording: {remaining:.1f}s tersisa', 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                cv2.putText(frame, f'Samples: {len(self.breathing_signal)}', 
                           (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                
                # Berhenti setelah 30 detik
                if elapsed >= self.duration:
                    self.recording = False
                    print("Recording selesai! Membuat laporan PDF...")
                    self.generate_pdf_report()
                    break
            
            cv2.imshow('Deteksi Pernapasan - Bahu Kiri & Kanan', frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord(' ') and not self.recording:
                # Mulai recording
                self.recording = True
                self.start_time = time.time()
                self.left_shoulder_data = []
                self.right_shoulder_data = []
                self.timestamps = []
                self.breathing_signal = []
                print("Mulai recording untuk 30 detik...")
        
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    detector = RespirationDetector()
    detector.run()