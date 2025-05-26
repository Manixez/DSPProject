import tkinter as tk
from tkinter import ttk, messagebox
import threading
import main
import time
import sys
import os

class MonitoringGUI:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Real-time Monitoring rPPG & Respirasi")
        self.root.geometry("500x400")
        self.root.resizable(False, False)
        
        # Styling
        self.root.configure(bg='#f0f0f0')
        
        # Monitoring state
        self.monitoring_active = False
        self.monitoring_thread = None
        self.update_thread = None
        
        # Setup UI
        self.setup_ui()
        
        # Setup window close handler
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
    
    def setup_ui(self):
        """Setup the user interface"""
        # Title
        title_label = tk.Label(
            self.root, 
            text="Real-time Health Monitoring", 
            font=("Arial", 16, "bold"),
            bg='#f0f0f0',
            fg='#333'
        )
        title_label.pack(pady=20)
        
        # Status frame
        status_frame = tk.Frame(self.root, bg='#f0f0f0')
        status_frame.pack(pady=10)
        
        tk.Label(status_frame, text="Status:", font=("Arial", 12), bg='#f0f0f0').pack(side=tk.LEFT)
        self.status_label = tk.Label(
            status_frame, 
            text="Ready", 
            font=("Arial", 12, "bold"),
            bg='#f0f0f0',
            fg='green'
        )
        self.status_label.pack(side=tk.LEFT, padx=(10, 0))
        
        # Separator
        separator1 = ttk.Separator(self.root, orient='horizontal')
        separator1.pack(fill='x', padx=20, pady=10)
        
        # Metrics frame
        metrics_frame = tk.Frame(self.root, bg='#f0f0f0')
        metrics_frame.pack(pady=20)
        
        # Heart Rate
        hr_frame = tk.Frame(metrics_frame, bg='white', relief='raised', bd=2)
        hr_frame.pack(pady=10, padx=20, fill='x')
        
        tk.Label(
            hr_frame, 
            text="❤️ Heart Rate", 
            font=("Arial", 12, "bold"),
            bg='white',
            fg='red'
        ).pack(pady=5)
        
        self.hr_label = tk.Label(
            hr_frame, 
            text="-- bpm", 
            font=("Arial", 16, "bold"),
            bg='white',
            fg='red'
        )
        self.hr_label.pack(pady=5)
        
        # Respiratory Rate
        rr_frame = tk.Frame(metrics_frame, bg='white', relief='raised', bd=2)
        rr_frame.pack(pady=10, padx=20, fill='x')
        
        tk.Label(
            rr_frame, 
            text="🫁 Respiratory Rate", 
            font=("Arial", 12, "bold"),
            bg='white',
            fg='blue'
        ).pack(pady=5)
        
        self.rr_label = tk.Label(
            rr_frame, 
            text="-- bpm", 
            font=("Arial", 16, "bold"),
            bg='white',
            fg='blue'
        )
        self.rr_label.pack(pady=5)
        
        # Separator
        separator2 = ttk.Separator(self.root, orient='horizontal')
        separator2.pack(fill='x', padx=20, pady=10)
        
        # Control buttons frame
        button_frame = tk.Frame(self.root, bg='#f0f0f0')
        button_frame.pack(pady=15)
        
        # Start button
        self.start_button = tk.Button(
            button_frame, 
            text="🟢 Start Monitoring", 
            command=self.start_monitoring,
            bg="#4CAF50",
            fg="white",
            font=("Arial", 12, "bold"),
            width=18,
            height=2,
            relief='raised',
            bd=3
        )
        self.start_button.pack(side=tk.LEFT, padx=10)
        
        # Stop button
        self.stop_button = tk.Button(
            button_frame, 
            text="🔴 Stop Monitoring", 
            command=self.stop_monitoring,
            bg="#f44336",
            fg="white",
            font=("Arial", 12, "bold"),
            width=18,
            height=2,
            relief='raised',
            bd=3,
            state=tk.DISABLED
        )
        self.stop_button.pack(side=tk.LEFT, padx=10)
        
        # Exit button
        self.exit_button = tk.Button(
            self.root, 
            text="Exit Application", 
            command=self.on_closing,
            bg="#9E9E9E",
            fg="white",
            font=("Arial", 10),
            width=20,
            relief='raised',
            bd=2
        )
        self.exit_button.pack(pady=15)
        
        # Info label
        info_text = "Ensure camera is connected and your face/chest is clearly visible"
        info_label = tk.Label(
            self.root, 
            text=info_text,
            font=("Arial", 9),
            bg='#f0f0f0',
            fg='gray',
            wraplength=400
        )
        info_label.pack(side=tk.BOTTOM, pady=10)
    
    def update_display(self):
        """Update the display with current HR and RR values"""
        while self.monitoring_active:
            try:
                # Get current values from main module
                current_hr = getattr(main, 'hr', 0)
                current_rr = getattr(main, 'rr', 0)
                
                # Update labels using thread-safe method
                self.root.after(0, self.update_hr_label, current_hr)
                self.root.after(0, self.update_rr_label, current_rr)
                
                time.sleep(1)  # Update every second
                
            except Exception as e:
                print(f"Error updating display: {e}")
                break
    
    def update_hr_label(self, hr):
        """Update heart rate label"""
        if hr > 0:
            self.hr_label.config(text=f"{hr:.1f} bpm")
        else:
            self.hr_label.config(text="Detecting...")
    
    def update_rr_label(self, rr):
        """Update respiratory rate label"""
        if rr > 0:
            self.rr_label.config(text=f"{rr:.1f} bpm")
        else:
            self.rr_label.config(text="Detecting...")
    
    def start_monitoring(self):
        """Start the monitoring process"""
        if self.monitoring_active:
            return
        
        # Check if required files exist
        if not self.check_dependencies():
            return
        
        try:
            self.monitoring_active = True
            
            # Update UI
            self.status_label.config(text="Initializing...", fg="orange")
            self.start_button.config(state=tk.DISABLED)
            self.stop_button.config(state=tk.NORMAL)
            self.hr_label.config(text="Starting...")
            self.rr_label.config(text="Starting...")
            
            # Reset main module variables
            main.monitoring_active = True
            main.hr = 0
            main.rr = 0
            main.features = None
            main.old_gray = None
            main.resp_roi_coords = None
            
            # Clear signal buffers
            main.r_signal.clear()
            main.g_signal.clear()
            main.b_signal.clear()
            main.resp_signal.clear()
            
            # Start monitoring thread
            self.monitoring_thread = threading.Thread(target=self.run_monitoring, daemon=True)
            self.monitoring_thread.start()
            
            # Start display update thread
            self.update_thread = threading.Thread(target=self.update_display, daemon=True)
            self.update_thread.start()
            
            # Update status after a short delay
            self.root.after(2000, lambda: self.status_label.config(text="Monitoring Active", fg="green"))
            
        except Exception as e:
            self.show_error(f"Failed to start monitoring: {str(e)}")
            self.stop_monitoring()
    
    def run_monitoring(self):
        """Run the main monitoring function"""
        try:
            main.run_main()
        except Exception as e:
            print(f"Monitoring error: {e}")
            self.root.after(0, self.show_error, f"Monitoring error: {str(e)}")
            self.root.after(0, self.stop_monitoring)
    
    def stop_monitoring(self):
        """Stop the monitoring process"""
        if not self.monitoring_active:
            return
        
        try:
            # Set flags to stop monitoring
            self.monitoring_active = False
            main.monitoring_active = False
            
            # Update UI
            self.status_label.config(text="Stopping...", fg="orange")
            
            # Wait a bit for threads to finish
            time.sleep(1)
            
            # Reset UI
            self.status_label.config(text="Stopped", fg="red")
            self.start_button.config(state=tk.NORMAL)
            self.stop_button.config(state=tk.DISABLED)
            self.hr_label.config(text="-- bpm")
            self.rr_label.config(text="-- bpm")
            
            print("Monitoring stopped successfully")
            
        except Exception as e:
            self.show_error(f"Error stopping monitoring: {str(e)}")
    
    def check_dependencies(self):
        """Check if required files and dependencies exist"""
        try:
            # Check if model file exists
            model_path = "Model/pose_landmarker.task"
            if not os.path.exists(model_path):
                self.show_error(f"Model file not found: {model_path}\nPlease ensure the model file is in the correct location.")
                return False
            
            # Try to import required modules
            import cv2
            import mediapipe as mp
            import matplotlib.pyplot as plt
            from scipy.signal import butter
            
            # Test camera
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                cap.release()
                self.show_error("Camera not available or already in use.\nPlease check your camera connection.")
                return False
            cap.release()
            
            return True
            
        except ImportError as e:
            self.show_error(f"Missing required library: {str(e)}\nPlease install all required packages.")
            return False
        except Exception as e:
            self.show_error(f"Dependency check failed: {str(e)}")
            return False
    
    def show_error(self, message):
        """Show error message"""
        messagebox.showerror("Error", message)
    
    def on_closing(self):
        """Handle window close event"""
        try:
            if self.monitoring_active:
                self.stop_monitoring()
                time.sleep(1)  # Give time for cleanup
            
            self.root.quit()
            self.root.destroy()
            
        except Exception as e:
            print(f"Error during cleanup: {e}")
            self.root.destroy()