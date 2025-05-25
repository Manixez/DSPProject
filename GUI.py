import tkinter as tk
import threading
import main
import time

root = tk.Tk()
root.title("Real-time Monitoring rPPG & Respirasi")

# Label tampilan
status_label = tk.Label(root, text="Status: Siap", font=("Arial", 12))
status_label.pack(pady=5)

hr_label = tk.Label(root, text="Heart Rate: -- bpm", font=("Arial", 14))
hr_label.pack()

rr_label = tk.Label(root, text="Respiratory Rate: -- bpm", font=("Arial", 14))
rr_label.pack()

def update_labels():
    while True:
        hr = main.hr
        rr = main.rr
        hr_label.config(text=f"Heart Rate: {hr:.1f} bpm")
        rr_label.config(text=f"Respiratory Rate: {rr:.1f} bpm")
        time.sleep(2)

def start_monitoring():
    status_label.config(text="Status: Monitoring aktif...")
    main.run_main()
    threading.Thread(target=update_labels, daemon=True).start()

tk.Button(root, text="Mulai Monitoring", command=start_monitoring).pack(pady=10)
tk.Button(root, text="Keluar", command=root.quit).pack(pady=10)

root.mainloop()
