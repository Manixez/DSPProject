import cv2
import tkinter as tk
from tkinter import ttk, messagebox
import threading
import time
import respirasi

# Global variables
cap = None
is_running = False

# GUI variables
root = None
status_var = None
results_text = None
lowcut_var = None
highcut_var = None
quality_var = None
corners_var = None
method_var = None

def setup_gui():
    """Setup GUI menggunakan Tkinter"""
    global root, status_var, results_text, lowcut_var, highcut_var
    global quality_var, corners_var, method_var
    
    root = tk.Tk()
    root.title("Sistem Respirasi - MediaPipe Shoulder + Optical Flow Lucas-Kanade")
    root.geometry("900x700")
    
    # Frame untuk kontrol
    control_frame = ttk.Frame(root)
    control_frame.pack(pady=10)
    
    # Tombol kontrol
    start_btn = ttk.Button(control_frame, text="Mulai Kamera", command=start_camera)
    start_btn.pack(side=tk.LEFT, padx=5)
    
    stop_btn = ttk.Button(control_frame, text="Stop Kamera", command=stop_camera)
    stop_btn.pack(side=tk.LEFT, padx=5)
    
    reset_btn = ttk.Button(control_frame, text="Reset Tracking", command=reset_tracking)
    reset_btn.pack(side=tk.LEFT, padx=5)
    
    # Frame untuk parameter
    param_frame = ttk.LabelFrame(root, text="Parameter Filter dan Tracking")
    param_frame.pack(pady=10, padx=10, fill='x')
    
    # Parameter controls
    ttk.Label(param_frame, text="Low Cut (Hz):").grid(row=0, column=0, padx=5, pady=5)
    lowcut_var = tk.DoubleVar(value=0.1)
    ttk.Scale(param_frame, from_=0.05, to=0.5, variable=lowcut_var, 
             orient='horizontal').grid(row=0, column=1, padx=5, pady=5)
    
    ttk.Label(param_frame, text="High Cut (Hz):").grid(row=1, column=0, padx=5, pady=5)
    highcut_var = tk.DoubleVar(value=0.8)
    ttk.Scale(param_frame, from_=0.5, to=2.0, variable=highcut_var, 
             orient='horizontal').grid(row=1, column=1, padx=5, pady=5)
    
    ttk.Label(param_frame, text="OF Quality Level:").grid(row=0, column=2, padx=5, pady=5)
    quality_var = tk.DoubleVar(value=0.3)
    ttk.Scale(param_frame, from_=0.1, to=0.8, variable=quality_var, 
             orient='horizontal').grid(row=0, column=3, padx=5, pady=5)
    
    ttk.Label(param_frame, text="Max Corners:").grid(row=1, column=2, padx=5, pady=5)
    corners_var = tk.IntVar(value=100)
    ttk.Scale(param_frame, from_=50, to=200, variable=corners_var, 
             orient='horizontal').grid(row=1, column=3, padx=5, pady=5)
    
    # Method selection
    method_frame = ttk.LabelFrame(root, text="Metode Ekstraksi Sinyal")
    method_frame.pack(pady=10, padx=10, fill='x')
    
    method_var = tk.StringVar(value="combined")
    ttk.Radiobutton(method_frame, text="MediaPipe Shoulder Only", variable=method_var, 
                   value="mediapipe").pack(side=tk.LEFT, padx=10)
    ttk.Radiobutton(method_frame, text="Optical Flow Only", variable=method_var, 
                   value="optical_flow").pack(side=tk.LEFT, padx=10)
    ttk.Radiobutton(method_frame, text="Combined (Shoulder + OF)", variable=method_var, 
                   value="combined").pack(side=tk.LEFT, padx=10)
    
    # Status display
    status_var = tk.StringVar(value="Status: Siap")
    ttk.Label(root, textvariable=status_var).pack(pady=5)
    
    # Results display
    results_frame = ttk.LabelFrame(root, text="Hasil Analisis Real-time")
    results_frame.pack(pady=10, padx=10, fill='both', expand=True)
    
    results_text = tk.Text(results_frame, height=12)
    scrollbar = ttk.Scrollbar(results_frame, orient="vertical", command=results_text.yview)
    results_text.configure(yscrollcommand=scrollbar.set)
    
    results_text.pack(side="left", fill="both", expand=True)
    scrollbar.pack(side="right", fill="y")

def start_camera():
    """Mulai kamera dan processing"""
    global cap, is_running
    
    try:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            messagebox.showerror("Error", "Tidak dapat membuka kamera")
            return
        
        is_running = True
        status_var.set("Status: Kamera aktif - Shoulder Detection + Optical Flow")
        
        # Start processing thread
        processing_thread = threading.Thread(target=process_video)
        processing_thread.daemon = True
        processing_thread.start()
        
        log_result("Sistem dimulai - MediaPipe Shoulder + Lucas-Kanade Optical Flow")
        
    except Exception as e:
        messagebox.showerror("Error", f"Error starting camera: {str(e)}")

def stop_camera():
    """Stop kamera dan processing"""
    global cap, is_running
    
    is_running = False
    if cap:
        cap.release()
    cv2.destroyAllWindows()
    status_var.set("Status: Kamera dihentikan")
    log_result("Kamera dihentikan")

def reset_tracking():
    """Reset tracking points"""
    respirasi.reset_tracking()
    log_result("Tracking points direset")

def update_respirasi_parameters():
    """Update parameters di respirasi module berdasarkan GUI values"""
    respirasi.update_parameters(
        new_lowcut=lowcut_var.get(),
        new_highcut=highcut_var.get(),
        quality=quality_var.get(),
        corners=corners_var.get(),
        new_method=method_var.get()
    )

def process_video():
    """Main video processing loop"""
    global is_running, cap
    
    cv2.namedWindow('MediaPipe Shoulder + Optical Flow Respirasi')
    
    start_time = time.time()
    
    while is_running:
        ret, frame = cap.read()
        if not ret:
            break
        
        current_time = time.time() - start_time
        
        # Update respirasi parameters dari GUI
        update_respirasi_parameters()
        
        # Process frame menggunakan respirasi module
        analysis_result = respirasi.process_frame(frame, current_time)
        
        # Log hasil jika ada
        if analysis_result:
            format_and_log_result(analysis_result)
        
        # Show frame
        cv2.imshow('MediaPipe Shoulder + Optical Flow Respirasi', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cv2.destroyAllWindows()

def format_and_log_result(result):
    """Format dan log hasil analisis"""
    formatted_result = f"=== ANALISIS RESPIRASI (SHOULDER FOCUS) ===\n"
    formatted_result += f"Metode: {result['method']}\n"
    formatted_result += f"Laju Respirasi: {result['respiratory_rate_fft']:.1f} BPM (FFT)\n"
    formatted_result += f"Laju Respirasi: {result['respiratory_rate_peaks']:.1f} BPM (Peak Detection)\n"
    formatted_result += f"Frekuensi Dominan: {result['dominant_frequency']:.3f} Hz\n"
    formatted_result += f"Jumlah Peak: {result['num_peaks']}\n"
    formatted_result += f"Kualitas Sinyal: {result['signal_quality']}\n"
    
    # Add shoulder info if available
    if result['shoulder_info']:
        shoulder_info = result['shoulder_info']
        formatted_result += f"Shoulder Distance: {shoulder_info['distance']:.1f}px\n"
        formatted_result += f"Shoulder Movement: {shoulder_info['movement']:.3f}\n"
    
    formatted_result += f"Shoulder Landmarks: {result['shoulder_landmarks_count']}\n"
    formatted_result += f"OF Data Points: {result['optical_flow_points']}\n"
    formatted_result += "="*50 + "\n"
    
    log_result(formatted_result)

def log_result(message):
    """Log hasil ke text widget"""
    if results_text:
        results_text.insert(tk.END, f"{time.strftime('%H:%M:%S')} - {message}\n")
        results_text.see(tk.END)

def on_closing():
    """Handle window closing"""
    stop_camera()
    root.destroy()

def main():
    """Main function"""
    setup_gui()
    root.protocol("WM_DELETE_WINDOW", on_closing)
    root.mainloop()

if __name__ == "__main__":
    main()