import tkinter as tk
from tkinter import ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
import numpy as np
from collections import deque
from scipy.signal import butter, filtfilt, find_peaks
from scipy.ndimage import uniform_filter1d
import threading
import time
import cv2
import main  # import dari file main.py

# --- Signal Processing Functions ---
def bandpass_filter(data, fs, lowcut, highcut, order=3):
    b, a = butter(order, [lowcut, highcut], btype='band', fs=fs)
    return filtfilt(b, a, data)

def estimate_bpm(signal, fs, lowcut, highcut):
    filtered = bandpass_filter(signal, fs, lowcut, highcut)
    filtered = uniform_filter1d(filtered, size=5)
    peaks, _ = find_peaks(filtered, distance=fs//2)
    bpm = 60 * len(peaks) / (len(filtered) / fs)
    return bpm, filtered

# --- GUI Class ---
class MedicalMonitorGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Real-time rPPG & Respirasi Monitor")
        self.root.configure(bg='black')

        # Parameters
        self.fps = 30
        self.buffer_size = 300
        self.rppg_buffer = deque([0]*self.buffer_size, maxlen=self.buffer_size)
        self.resp_buffer = deque([0]*self.buffer_size, maxlen=self.buffer_size)
        self.running = False

        # UI Layout
        self.setup_ui()

        # Handle close
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

        # Animation
        self.update_plot()

    def setup_ui(self):
        self.fig, (self.ax1, self.ax2) = plt.subplots(2, 1, figsize=(8, 4))
        self.fig.patch.set_facecolor('black')

        for ax in (self.ax1, self.ax2):
            ax.set_facecolor('black')
            ax.set_ylim(-2, 2)
            ax.grid(True, linestyle='--', alpha=0.3)
            ax.tick_params(colors='white')
            for spine in ax.spines.values():
                spine.set_color('white')

        self.line1, = self.ax1.plot([], [], color='cyan', linewidth=2)
        self.line2, = self.ax2.plot([], [], color='lime', linewidth=2)
        self.text_rr = self.ax1.text(0.95, 0.85, '', transform=self.ax1.transAxes, color='white', fontsize=14, ha='right')
        self.text_hr = self.ax2.text(0.95, 0.85, '', transform=self.ax2.transAxes, color='white', fontsize=14, ha='right')

        self.canvas = FigureCanvasTkAgg(self.fig, master=self.root)
        self.canvas.get_tk_widget().pack(padx=10, pady=10)

        control_frame = tk.Frame(self.root, bg='black')
        control_frame.pack(pady=5)

        self.start_btn = tk.Button(control_frame, text="Start Monitoring", command=self.start_monitoring, bg='green', fg='white')
        self.start_btn.pack(side=tk.LEFT, padx=10)

        self.stop_btn = tk.Button(control_frame, text="Stop", command=self.stop_monitoring, bg='red', fg='white', state=tk.DISABLED)
        self.stop_btn.pack(side=tk.LEFT, padx=10)

    def update_plot(self):
        if self.running:
            rr_bpm, rr_filtered = estimate_bpm(list(self.resp_buffer), fs=self.fps, lowcut=0.1, highcut=0.7)
            hr_bpm, hr_filtered = estimate_bpm(list(self.rppg_buffer), fs=self.fps, lowcut=0.7, highcut=3.0)

            self.line1.set_data(np.arange(len(rr_filtered)), rr_filtered)
            self.line2.set_data(np.arange(len(hr_filtered)), hr_filtered)
            self.text_rr.set_text(f'RR: {rr_bpm:.1f} bpm')
            self.text_hr.set_text(f'HR: {hr_bpm:.1f} bpm')
            self.ax1.set_xlim(0, len(rr_filtered))
            self.ax2.set_xlim(0, len(hr_filtered))
            self.canvas.draw()

        if self.root.winfo_exists():
            self.root.after(1000, self.update_plot)

    def start_monitoring(self):
        self.running = True
        self.start_btn.config(state=tk.DISABLED)
        self.stop_btn.config(state=tk.NORMAL)
        main.monitoring_active = True
        threading.Thread(target=main.run_main, daemon=True).start()
        threading.Thread(target=self.update_from_main, daemon=True).start()

    def stop_monitoring(self):
        self.running = False
        main.monitoring_active = False
        self.start_btn.config(state=tk.NORMAL)
        self.stop_btn.config(state=tk.DISABLED)

    def on_closing(self):
        self.running = False
        main.monitoring_active = False
        self.root.destroy()

    def update_from_main(self):
        while self.running:
            if len(main.g_signal) > 0:
                self.rppg_buffer.append(main.g_signal[-1])
            if len(main.resp_signal) > 0:
                self.resp_buffer.append(main.resp_signal[-1])
            time.sleep(1 / self.fps)

if __name__ == "__main__":
    root = tk.Tk()
    app = MedicalMonitorGUI(root)
    root.mainloop()