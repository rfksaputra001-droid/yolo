# Test YOLO di Google Colab (GPU)

Panduan lengkap menjalankan `test_yolo_colab.ipynb` untuk menguji model deteksi dan counting kendaraan menggunakan GPU gratis dari Google Colab.

---

## Persiapan — Upload Dataset ke Google Drive

Sebelum buka Colab, upload file berikut ke Google Drive dengan struktur folder persis seperti ini:

```
MyDrive/
└── pktj/
    ├── models/
    │   └── best.pt
    └── data/
        ├── images/
        │   ├── train/
        │   └── val/
        └── labels/
            ├── train/
            └── val/
```

File yang perlu diupload:
- `best.pt` → ada di `yolo/models/best.pt`
- Folder `data/` → ada di `coba/pktj-project/backend/yolo/data/`

---

## Langkah 1 — Buka Notebook di Colab

1. Buka [colab.research.google.com](https://colab.research.google.com)
2. Klik **File → Open notebook**
3. Pilih tab **GitHub**
4. Masukkan URL repo:
   ```
   https://github.com/rfksaputra001-droid/yolo
   ```
5. Pilih file: `colab/test_yolo_colab.ipynb`
6. Klik **Open**

---

## Langkah 2 — Aktifkan GPU

1. Klik menu **Runtime** (di atas)
2. Klik **Change runtime type**
3. Pada **Hardware accelerator** pilih **T4 GPU**
4. Klik **Save**

> Tanpa GPU, notebook tetap bisa jalan tapi lebih lambat.

---

## Langkah 3 — Jalankan Cell Satu per Satu

Tekan `Shift + Enter` untuk jalankan tiap cell, atau klik **Runtime → Run all** untuk jalankan semua sekaligus.

### Cell 1 — Cek GPU & Install
Mengecek GPU tersedia dan install `ultralytics`.

Output yang diharapkan:
```
GPU tersedia: NVIDIA Tesla T4 ...
PyTorch : 2.x.x
CUDA    : True
GPU     : Tesla T4
Device yang akan digunakan: cuda
```

### Cell 2 — Mount Google Drive
Akan muncul popup izin akses Google Drive. Klik **Allow** dan ikuti instruksi.

Output yang diharapkan:
```
Model      : /content/drive/MyDrive/pktj/models/best.pt — OK
Val images : ... — OK
Train imgs : ... — OK
Jumlah val images  : 25
Jumlah train images: 142
```

> Jika muncul `TIDAK ADA!`, berarti struktur folder di Drive salah. Cek kembali langkah Persiapan.

### Cell 3 — Load Model
Model dimuat dan warmup GPU.

Output yang diharapkan:
```
Model loaded : best.pt
Device       : cuda
Classes      : {0: 'mobil', 1: 'bus', 2: 'truk'}
Warmup selesai — model siap
```

### Cell 4 — Distribusi Label Dataset
Menghitung jumlah objek per kelas di dataset train dan val. Menghasilkan grafik bar dan menyimpan `distribusi.png`.

### Cell 5 — Deteksi Val Set
Menjalankan deteksi pada 25 gambar val dan membandingkan dengan ground truth.

Output yang diharapkan (contoh):
```
Kelas    GT   Pred  Selisih
-------------------------------
mobil    84    79       -5
bus      12    10       -2
truk      8     9       +1
-------------------------------
Total   104    98
```

### Cell 6 — Visualisasi Bounding Box
Menampilkan 6 sampel gambar dengan:
- **Kotak warna** = hasil prediksi model
- **Kotak abu-abu** = ground truth label

Menyimpan `visualisasi.png`.

### Cell 7 — Simulasi Counting
Mensimulasikan penghitungan kendaraan menggunakan frame-frame train sebagai urutan video. Kendaraan dihitung saat melewati garis tengah frame (crossing line).

Output yang diharapkan:
```
Total kendaraan : 37

Kelas    Kiri   Kanan  Total
------------------------------
mobil      12      14     26
bus         3       2      5
truk        4       2      6
```

### Cell 8 — Grafik Counting
Menampilkan 3 grafik: bar per kelas, grouped bar per lajur, dan pie chart proporsi. Menyimpan `counting.png`.

### Cell 9 — Evaluasi Model (mAP)
Menjalankan evaluasi resmi menggunakan Ultralytics validator.

Output yang diharapkan (contoh):
```
Precision (mean) : 0.8231  (82.3%)
Recall (mean)    : 0.7654  (76.5%)
mAP@0.5          : 0.8102  (81.0%)
mAP@0.5:0.95     : 0.5341  (53.4%)

Verdict: BAGUS — siap produksi
```

| mAP@0.5 | Verdict |
|---------|---------|
| ≥ 0.8 | BAGUS — siap produksi |
| ≥ 0.6 | CUKUP — perlu lebih banyak data |
| ≥ 0.4 | KURANG — perlu augmentasi |
| < 0.4 | BURUK — training ulang |

### Cell 10 — Download Output
Menampilkan ringkasan semua hasil dan menawarkan download file PNG ke komputer.

---

## File Output yang Dihasilkan

| File | Isi |
|------|-----|
| `distribusi.png` | Grafik jumlah objek per kelas (train & val) |
| `visualisasi.png` | Bounding box prediksi vs ground truth |
| `counting.png` | Hasil simulasi counting per kelas & lajur |

---

## Troubleshooting

**"TIDAK ADA!" saat Cell 2:**
Struktur folder di Drive salah. Pastikan path-nya persis:
```
MyDrive/pktj/models/best.pt
MyDrive/pktj/data/images/train/
```

**GPU tidak terdeteksi:**
Ulangi langkah 2 (Change runtime type → T4 GPU), lalu **Factory reset runtime** dan jalankan ulang dari Cell 1.

**Error saat mount Drive:**
Klik ikon folder di sidebar kiri Colab → klik ikon Google Drive → authorize ulang.

**Cell 9 error `No such file`:**
Pastikan Cell 2 berhasil dan path dataset terbaca dengan benar sebelum Cell 9.

---

## Estimasi Waktu (T4 GPU)

| Cell | Estimasi |
|------|----------|
| Install (Cell 1) | 1–2 menit |
| Load model (Cell 3) | 10 detik |
| Deteksi val 25 img (Cell 5) | 5–10 detik |
| Counting 142 frame (Cell 7) | 15–30 detik |
| Evaluasi mAP (Cell 9) | 30–60 detik |
| **Total** | **~5 menit** |
