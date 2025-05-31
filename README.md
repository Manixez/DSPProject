# Tugas Besar Digital Signal Processing (IF3024)
# Dosen Pengampu: Martin Clinton Tosima Manullang, S.T., M.T., Ph.D.

Anggota Kelompok:
<ol>
  <li>Arkan Hariz Chandrawinata Liem - 122140038 - ArkanHariz</li>
  <li>Bezalel Samuel Manik - 122140140 - Manixez</li>
</ol>
<hr>

# Deskripsi Project
Proyek akhir mata kuliah "Pengolahan Sinyal Digital IF(3024)" ini memperkenalkan sebuah program inovatif untuk memperoleh sinyal respirasi dan remote-photoplethysmography (rPPG) secara real-time menggunakan input video dari webcam standar. Program ini memanfaatkan teknologi pose-landmarker MediaPipe untuk mendeteksi pergerakan bahu pengguna, yang kemudian diolah untuk mengekstraksi sinyal respirasi. Sementara itu, sinyal rPPG (yang mengindikasikan detak jantung) juga akan didapatkan dari perubahan warna kulit yang terekam oleh webcam, tanpa memerlukan kontak fisik menggunakan algoritma Plane-Orthogonal-to-Skin (POS).
<hr>

# Logbook
| Tanggal | Progress |
|---------|----------|
| 17 Mei 2025 | Membuat repository di Github |
| 25 Mei 2025 | Membuat file GUI, respirasi, dan rPPG |
| 26 Mei 2025 | Mencoba program respirasi untuk track bahu |
| 27 Mei 2025 | Mencoba progarm respirasi kembali dan mencoba hasil disimpan dalam bentuk pdf |
| 28 Mei 2025 | Tidak ada kegiatan progress karena mengerjakan tugas besar lain |
| 29 Mei 2025 | Menyicil laporan yaitu mengerjakan pendahuluan |
| 30 Mei 2025 | Progress untuk program rPPG dan respirasi serta menyicil readme dan laporan |
| 31 Mei 2025 | Finalisasi program dan laporan |
<hr>

# Tools yang Digunakan
Berikut adalah tools yang digunakan pada pengerjaan tugas besar ini:

| Nama Tools         | Penjelasan                                                                                  |
|--------------------|---------------------------------------------------------------------------------------------|
| Python             | Python digunakan dalam menulis script untuk tugas besar kali ini sebagai bahasa pemrograman |
| Visual Studio Code | Teks editor media menulis script code sebuah program                                        |
<hr>

# Library yang Digunakan
| Library                        | Penjelasan                                                                             |
|--------------------------------|----------------------------------------------------------------------------------------|
| PyQt5 (versi 5.15.9)           | Digunakan untuk membuat antarmuka grafis (GUI) di Python                               |
| opencv-python (versi 4.9.0.80) | Digunakan untuk pemrosesan gambar dan video, termasuk deteksi wajah dan tracking       |     
| matplotlib (versi 3.7.1)       | Digunakan untuk membuat visualisasi data seperti grafik, histogram, dan plot           |
| numpy (versi 1.24.3)           | Digunakan untuk operasi matematika dan array berdimensi banyak                         |
| scipy (versi 1.10.1)           | Digunakan untuk menyediakan fungsi ilmiah dan teknik seperti integral dan sinyal       |
| mediapipe (versi 0.10.14)      | Digunakan untuk deteksi pose, wajah, tangan, dan objek menggunakan AI secara real-time |
<hr>

