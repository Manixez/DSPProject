import sys
import cv2
import numpy as np
import threading
import time
from PyQt5.QtWidgets import QApplication, QLabel, QPushButton, QVBoxLayout, QWidget, QHBoxLayout
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import QTimer
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from signal_processing import estimate_bpm
import main

class MonitorApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Real-Time rPPG & Respirasi Monitor")
        self.setStyleSheet("background-color: black; color: white;")

        self.fps = 30
        self.buffer_size = 300
        self.rppg_data = []
        self.resp_data = []

        self.init_ui()

        self.timer = QTimer()
        self.timer.timeout.connect(self.update_gui)

    def init_ui(self):
        self.video_label = QLabel()
        self.video_label.setFixedSize(640, 480)

        self.hr_label = QLabel("HR: 0.0 bpm")
        self.rr_label = QLabel("RR: 0.0 bpm")

        self.start_btn = QPushButton("Start")
        self.start_btn.clicked.connect(self.start_monitoring)

        self.stop_btn = QPushButton("Stop")
        self.stop_btn.clicked.connect(self.stop_monitoring)
        self.stop_btn.setEnabled(False)

        self.canvas, self.axs = plt.subplots(2, 1, figsize=(5, 3))
        self.canvas_widget = FigureCanvas(self.canvas)
        self.axs[0].set_title("Respirasi")
        self.axs[1].set_title("rPPG")

        btn_layout = QHBoxLayout()
        btn_layout.addWidget(self.start_btn)
        btn_layout.addWidget(self.stop_btn)

        layout = QVBoxLayout()
        layout.addWidget(self.video_label)
        layout.addWidget(self.hr_label)
        layout.addWidget(self.rr_label)
        layout.addLayout(btn_layout)
        layout.addWidget(self.canvas_widget)

        self.setLayout(layout)

    def start_monitoring(self):
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.rppg_data.clear()
        self.resp_data.clear()
        main.monitoring_active = True
        threading.Thread(target=main.run_main, daemon=True).start()
        self.timer.start(1000 // self.fps)

    def stop_monitoring(self):
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        main.monitoring_active = False
        self.timer.stop()
        self.plot_final_results()

    def update_gui(self):
        frame = main.frame_display
        if frame is not None:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = frame_rgb.shape
            bytes_per_line = ch * w
            qimg = QImage(frame_rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
            self.video_label.setPixmap(QPixmap.fromImage(qimg))

        if len(main.g_signal) > 0:
            self.rppg_data.append(main.g_signal[-1])
        if len(main.resp_signal) > 0:
            self.resp_data.append(main.resp_signal[-1])

        # BPM estimation
        hr_bpm, _ = estimate_bpm(self.rppg_data[-self.buffer_size:], self.fps, 0.7, 3.0)
        rr_bpm, _ = estimate_bpm(self.resp_data[-self.buffer_size:], self.fps, 0.1, 0.7)

        self.hr_label.setText(f"HR: {hr_bpm:.1f} bpm")
        self.rr_label.setText(f"RR: {rr_bpm:.1f} bpm")

        # Update real-time plots
        self.axs[0].cla()
        self.axs[0].plot(self.resp_data[-self.buffer_size:], color='cyan')
        self.axs[0].set_title("Respirasi")

        self.axs[1].cla()
        self.axs[1].plot(self.rppg_data[-self.buffer_size:], color='lime')
        self.axs[1].set_title("rPPG")

        self.canvas.draw()

    def plot_final_results(self):
        plt.figure("Final Result")
        plt.subplot(2, 1, 1)
        plt.plot(self.resp_data, color='cyan')
        final_rr, _ = estimate_bpm(self.resp_data, self.fps, 0.1, 0.7)
        plt.title(f"Respiration Signal (RR: {final_rr:.1f} bpm)")

        plt.subplot(2, 1, 2)
        plt.plot(self.rppg_data, color='lime')
        final_hr, _ = estimate_bpm(self.rppg_data, self.fps, 0.7, 3.0)
        plt.title(f"rPPG Signal (HR: {final_hr:.1f} bpm)")

        plt.tight_layout()
        plt.show()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MonitorApp()
    window.show()
    sys.exit(app.exec_())