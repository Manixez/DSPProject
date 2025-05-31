import sys
import threading
import numpy as np
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QPushButton, QLabel, QHBoxLayout
from PyQt5.QtGui import QImage, QPixmap, QFont
from PyQt5.QtCore import QTimer
import cv2
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from signal_processing import estimate_bpm
import main

class RPPGApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Real-time Medical Monitor Style")
        self.setStyleSheet("background-color: black; color: white;")
        self.resize(1400, 800)

        self.fps = 30
        self.main_layout = QHBoxLayout()
        self.setLayout(self.main_layout)

        # --- Left panel: signals and labels ---
        self.left_panel = QVBoxLayout()

        self.fig, self.axs = plt.subplots(2, 1, figsize=(8, 4), dpi=100)
        for ax in self.axs:
            ax.set_facecolor('black')
            ax.tick_params(colors='white')
            for spine in ax.spines.values():
                spine.set_color('white')
            ax.grid(True, linestyle='--', alpha=0.3, color='gray')
        self.fig.patch.set_facecolor('black')
        self.canvas = FigureCanvas(self.fig)
        self.left_panel.addWidget(self.canvas)

        stat_layout = QHBoxLayout()
        self.hr_label = QLabel("HR: -- bpm")
        self.hr_label.setStyleSheet("color: lime;")
        self.hr_label.setFont(QFont("Consolas", 28))

        self.rr_label = QLabel("RR: -- bpm")
        self.rr_label.setStyleSheet("color: cyan;")
        self.rr_label.setFont(QFont("Consolas", 28))

        stat_layout.addWidget(self.hr_label)
        stat_layout.addStretch()
        stat_layout.addWidget(self.rr_label)
        self.left_panel.addLayout(stat_layout)

        control_layout = QHBoxLayout()
        self.start_btn = QPushButton("Start")
        self.stop_btn = QPushButton("Stop")
        self.stop_btn.setEnabled(False)
        control_layout.addWidget(self.start_btn)
        control_layout.addWidget(self.stop_btn)
        self.left_panel.addLayout(control_layout)

        # --- Right panel: video feed ---
        self.right_panel = QVBoxLayout()
        self.video_label = QLabel()
        self.video_label.setFixedSize(640, 480)
        self.right_panel.addWidget(self.video_label)

        # Combine both panels
        self.main_layout.addLayout(self.left_panel, 2)
        self.main_layout.addLayout(self.right_panel, 1)

        self.start_btn.clicked.connect(self.start_monitoring)
        self.stop_btn.clicked.connect(self.stop_monitoring)

        self.timer = QTimer()
        self.timer.timeout.connect(self.update_ui)

    def start_monitoring(self):
        main.monitoring_active = True
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.timer.start(1000)
        threading.Thread(target=main.run_main, daemon=True).start()

    def stop_monitoring(self):
        main.monitoring_active = False
        self.timer.stop()
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.show_final_plot()

    def update_ui(self):
        # Video frame
        if main.frame_display is not None:
            frame = main.frame_display
            rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_image.shape
            bytes_per_line = ch * w
            qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
            self.video_label.setPixmap(QPixmap.fromImage(qt_image))

        # Signal plotting
        if len(main.rgb_buffer) > 100:
            rgb_np = np.array(main.rgb_buffer)
            h_signal = main.apply_pos(rgb_np)
            hr_bpm, hr_filt = estimate_bpm(h_signal, fs=main.fps, lowcut=0.7, highcut=3.0)
            rr_bpm, rr_filt = estimate_bpm(list(main.resp_signal), fs=main.fps, lowcut=0.1, highcut=0.7)

            self.axs[0].cla()
            self.axs[0].plot(rr_filt[-300:], color='cyan')
            self.axs[0].set_title("Respiratory Signal", color='white')
            self.axs[0].set_facecolor('black')

            self.axs[1].cla()
            self.axs[1].plot(hr_filt[-300:], color='lime')
            self.axs[1].set_title("rPPG Signal", color='white')
            self.axs[1].set_facecolor('black')

            self.canvas.draw()

            self.hr_label.setText(f"HR: {hr_bpm:.1f} bpm")
            self.rr_label.setText(f"RR: {rr_bpm:.1f} bpm")

    def show_final_plot(self):
        if len(main.rgb_full) == 0 or len(main.resp_full) == 0:
            return

        rgb_np = np.array(main.rgb_full)
        h_signal = main.apply_pos(rgb_np)
        hr_bpm, hr_filt = estimate_bpm(h_signal, fs=main.fps, lowcut=0.7, highcut=3.0)
        rr_bpm, rr_filt = estimate_bpm(main.resp_full, fs=main.fps, lowcut=0.1, highcut=0.7)

        plt.figure("Final Full Signal", figsize=(10, 4))
        plt.subplot(2, 1, 1)
        plt.plot(rr_filt, color='cyan')
        plt.title(f"Respiration Signal (RR: {rr_bpm:.1f} bpm)")

        plt.subplot(2, 1, 2)
        plt.plot(hr_filt, color='lime')
        plt.title(f"rPPG Signal (HR: {hr_bpm:.1f} bpm)")

        plt.tight_layout()
        plt.show()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = RPPGApp()
    window.show()
    sys.exit(app.exec_())