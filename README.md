# 🎯 Tugas Besar Digital Signal Processing (IF3024)

**Dosen Pengampu**: Martin Clinton Tosima Manullang, S.T., M.T., Ph.D.

---

## 👨‍💻 Anggota Kelompok
| No. | Nama                              | NIM         | Username    |
|-----|-----------------------------------|-------------|-------------|
| 1   | Arkan Hariz Chandrawinata Liem   | 122140038   | ArkanHariz  |
| 2   | Bezalel Samuel Manik             | 122140140   | Manixez     |

---

## 📘 Deskripsi Proyek

Proyek akhir mata kuliah **Pengolahan Sinyal Digital (IF3024)** ini memperkenalkan program inovatif yang mampu mendeteksi dan menampilkan **sinyal respirasi** serta **remote photoplethysmography (rPPG)** secara real-time menggunakan input video dari **webcam standar**.

### 🔹 Fitur Utama:
- Deteksi **pernapasan** berbasis pergerakan bahu menggunakan **MediaPipe PoseLandmarker**.
- Ekstraksi sinyal **rPPG** (detak jantung) dari perubahan warna wajah tanpa kontak fisik, menggunakan algoritma **Plane-Orthogonal-to-Skin (POS)**.
- **Antarmuka GUI interaktif** dengan visualisasi real-time dan hasil akhir dari HR & RR.
- Dukungan **Tkinter** dan **PyQt5** GUI (versi PyQt5 direkomendasikan untuk tampilan modern).

---

## 🧪 Logbook Progress

| Tanggal       | Kegiatan                                                                 |
|---------------|--------------------------------------------------------------------------|
| 17 Mei 2025   | Membuat repository di GitHub                                             |
| 25 Mei 2025   | Membuat file `GUI.py`, `respirasi.py`, dan `rPPG.py`                    |
| 26 Mei 2025   | Mencoba program respirasi dan tracking bahu                             |
| 27 Mei 2025   | Menyimpan hasil sinyal respirasi ke PDF                                 |
| 28 Mei 2025   | Tidak ada kegiatan karena mengerjakan tugas besar lainnya               |
| 29 Mei 2025   | Menulis Bab Pendahuluan laporan                                          |
| 30 Mei 2025   | Menyicil laporan dan memfinalisasi fitur GUI & pengolahan sinyal        |
| 31 Mei 2025   | Finalisasi program dan dokumentasi                                       |

---

## 🛠 Tools yang Digunakan

| Tools              | Keterangan                                                    |
|--------------------|---------------------------------------------------------------|
| Python             | Bahasa pemrograman utama untuk logika program                 |
| Visual Studio Code | Code editor yang digunakan untuk menulis dan debugging kode   |
| GitHub             | Version control dan kolaborasi kode secara online             |

---

## 📦 Library yang Digunakan

| Library                        | Fungsi                                                                 |
|--------------------------------|------------------------------------------------------------------------|
| `PyQt5==5.15.9`               | GUI modern berbasis Qt untuk Python                                    |
| `opencv-python==4.9.0.80`     | Pemrosesan video, pengambilan frame dari webcam, deteksi objek         |
| `matplotlib==3.7.1`           | Visualisasi sinyal dalam bentuk grafik                                 |
| `numpy==1.24.3`               | Operasi numerik dan manajemen array                                    |
| `scipy==1.10.1`               | Pemrosesan sinyal (filter, peak detection, dan smooth signal)          |
| `mediapipe==0.10.14`          | Deteksi wajah dan pose tubuh secara real-time menggunakan AI           |

---

## 🧰 Langkah Instalasi

### 1. **Clone Repository**
```bash
git clone https://github.com/username/nama-repo.git
cd nama-repo
```

### 2. **Buat Virtual Environment (Direkomendasikan)**
Gunakan Python **3.9.x** agar semua dependensi, terutama MediaPipe, kompatibel:
```bash
python -m venv venv
venv\Scripts\activate         # Windows
# atau
source venv/bin/activate     # Mac/Linux
```

### 3. **Install Dependensi**
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. **Siapkan Model MediaPipe**
Pastikan direktori `Model/` berisi:
- `blaze_face_short_range.tflite`
- `pose_landmarker.task`

Unduh dari [MediaPipe GitHub](https://github.com/google/mediapipe) jika belum tersedia.

### 5. **Jalankan Program**
```bash
python GUI.py     # GUI berbasis Tkinter
# atau
python gui_qt.py  # GUI PyQt5 modern (disarankan)
```

---

## 📷 Cuplikan Program

| Video & Deteksi ROI | Visualisasi HR & RR |
|---------------------|---------------------|
| ![video](/assets/demo_frame.png) | ![signal](/assets/signal_plot.png) |

---

## 📌 Catatan Tambahan

- Pastikan webcam aktif dan tidak digunakan aplikasi lain.
- Jika sinyal tidak muncul, pastikan deteksi wajah dan bahu berjalan dengan baik.
- Program akan menampilkan **sinyal akhir lengkap** setelah tombol `Stop` ditekan.

---

## 📄 Lisensi
Proyek ini ditujukan untuk keperluan **akademik** dan **pembelajaran** dalam mata kuliah IF3024.

---

