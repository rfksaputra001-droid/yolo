# Sistem Deteksi & Kinerja Ruas Jalan

Aplikasi web untuk deteksi dan penghitungan kendaraan berbasis YOLO, dengan dashboard analisis kinerja ruas jalan.

**Stack:**
- Frontend: React + Vite + TailwindCSS (port 3000)
- Backend: Node.js + Express + MongoDB (port 5000)
- YOLO API: FastAPI + Ultralytics YOLOv8 (port 8000)
- Database: MongoDB via Docker

---

## Prasyarat

| Tool | Versi Minimum |
|------|--------------|
| Node.js | 18+ |
| Python | 3.10+ |
| Docker | 20+ |
| ffmpeg | — |

Install ffmpeg jika belum ada:
```bash
sudo apt install ffmpeg
```

---

## 1. Clone & Setup

```bash
git clone git@github.com:rfksaputra001-droid/yolo.git
cd yolo
```

---

## 2. Jalankan MongoDB (Docker)

```bash
docker run -d \
  --name pktj_mongo \
  -e MONGO_INITDB_ROOT_USERNAME=admin \
  -e MONGO_INITDB_ROOT_PASSWORD=password \
  -p 27017:27017 \
  mongo:6
```

Cek apakah berjalan:
```bash
docker ps | grep mongo
```

---

## 3. Setup Backend

```bash
cd backend
npm install
```

Buat file `.env` di dalam folder `backend/`:
```env
PORT=5000
NODE_ENV=development
MONGO_URI=mongodb://admin:password@localhost:27017/pktj?authSource=admin
JWT_SECRET=pktj_jwt_secret_key_minimum_32_characters_long_2024
JWT_EXPIRE=7d
YOLO_API_URL=http://localhost:8000
CORS_ORIGIN=http://localhost:3000
```

Jalankan backend:
```bash
node src/server.js
```

Buat akun admin (jalankan sekali saja):
```bash
node src/scripts/createAdmin.js
```

> Default: `admin@admin.com` / `admin123`

---

## 4. Setup Frontend

```bash
cd frontend
npm install
```

Buat file `.env` di dalam folder `frontend/`:
```env
VITE_API_URL=
VITE_YOLO_API_URL=http://localhost:8000
VITE_APP_NAME=Kinerja Ruas Jalan
```

> `VITE_API_URL` dikosongkan agar request lewat Vite proxy ke `localhost:5000`.

Jalankan frontend:
```bash
npm run dev
```

Buka browser: **http://localhost:3000**

---

## 5. Setup YOLO API (Lokal, tanpa GPU)

Install dependensi Python:
```bash
cd yolo
pip install -r requirements.txt
```

Jalankan YOLO API:
```bash
uvicorn api_local:app --host 0.0.0.0 --port 8000
```

Test berjalan:
```bash
curl http://localhost:8000/ping
# → {"status":"ok","message":"YOLO API running (local mode)"}
```

---

## 6. Menjalankan Semua Sekaligus

Buka **3 terminal terpisah**:

**Terminal 1 — YOLO API:**
```bash
cd yolo
uvicorn api_local:app --host 0.0.0.0 --port 8000
```

**Terminal 2 — Backend:**
```bash
cd backend
node src/server.js
```

**Terminal 3 — Frontend:**
```bash
cd frontend
npm run dev
```

---

## 7. Test Model YOLO (Notebook)

Untuk menguji akurasi model deteksi dan counting tanpa menjalankan server:

```bash
cd yolo
jupyter notebook test_yolo_lokal.ipynb
```

Atau dengan JupyterLab:
```bash
jupyter lab test_yolo_lokal.ipynb
```

**Isi notebook:**

| Cell | Fungsi |
|------|--------|
| 1 | Setup path dataset & model |
| 2 | Load model YOLO |
| 3 | Analisis distribusi label dataset (ground truth) |
| 4 | Deteksi pada semua gambar val, bandingkan dengan GT |
| 5 | Visualisasi bounding box 6 sampel gambar |
| 6 | Simulasi counting kendaraan (crossing line) |
| 7 | Visualisasi hasil counting per kelas & lajur |
| 8 | Evaluasi model (mAP@0.5, Precision, Recall) |
| 9 | Ringkasan akhir semua hasil |

**Output yang dihasilkan:**
- `test_output_distribusi.png` — grafik distribusi kelas dataset
- `test_output_visualisasi.png` — visualisasi bounding box prediksi vs GT
- `test_output_counting.png` — grafik hasil counting per kelas & lajur

---

## 8. Struktur Folder

```
yolo/                          ← root repo
├── frontend/                  ← React app (port 3000)
│   ├── src/
│   │   ├── pages/             ← Dashboard, Deteksi, Histori, dll
│   │   ├── services/          ← API service calls
│   │   └── context/           ← Auth context
│   ├── .env                   ← (buat sendiri, tidak di-commit)
│   └── vite.config.js
├── backend/                   ← Node.js/Express (port 5000)
│   ├── src/
│   │   ├── routes/            ← auth, detect, dashboard, realtime, dll
│   │   ├── controllers/
│   │   └── models/
│   └── .env                   ← (buat sendiri, tidak di-commit)
├── yolo/                      ← YOLO FastAPI (port 8000)
│   ├── api_local.py           ← API lokal (tanpa Cloudinary)
│   ├── api.py                 ← API dengan Cloudinary (untuk produksi)
│   ├── models/best.pt         ← Model terlatih (6MB)
│   ├── test_yolo_lokal.ipynb  ← Notebook test lokal
│   ├── outputs/               ← Video hasil deteksi (auto-dibuat)
│   └── requirements.txt
└── colab/
    └── yolo_api_pktj.ipynb    ← Notebook untuk Google Colab
```

---

## 9. Troubleshooting

**Login gagal / 401:**
- Pastikan backend berjalan di port 5000
- Cek MongoDB Docker aktif: `docker ps`
- Cek `MONGO_URI` di `backend/.env`

**YOLO API tidak merespons:**
```bash
curl http://localhost:8000/ping
```
Jika gagal, jalankan ulang: `uvicorn api_local:app --host 0.0.0.0 --port 8000`

**Video tidak bisa diputar setelah deteksi:**
- Pastikan `ffmpeg` terinstall: `ffmpeg -version`
- Video output tersimpan di `yolo/outputs/`

**Port sudah dipakai:**
```bash
# Cari dan kill proses di port tertentu
lsof -ti:3000 | xargs kill -9
lsof -ti:5000 | xargs kill -9
lsof -ti:8000 | xargs kill -9
```

**MongoDB auth error:**
```bash
# Cek container berjalan
docker ps -a | grep mongo

# Restart jika perlu
docker start pktj_mongo
```

---

## 10. Kelas Deteksi

| ID | Nama | Warna |
|----|------|-------|
| 0  | mobil | hijau |
| 1  | bus   | biru  |
| 2  | truk  | oranye |

Dataset: 142 gambar train + 25 gambar val
