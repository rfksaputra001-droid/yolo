# api_local.py - YOLO Vehicle Detection - Local Mode (no Cloudinary)
import os
import asyncio
import tempfile
import shutil
import cv2
import gc
import json
import subprocess
import numpy as np
from fastapi import FastAPI, UploadFile, File, BackgroundTasks, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from ultralytics import YOLO

# ================= APP SETUP =================
app = FastAPI(title="YOLO Vehicle Detection API (Local)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ================= PATHS =================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "best.pt")
JOBS_DIR = os.path.join(BASE_DIR, "jobs")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")

os.makedirs(JOBS_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Serve output videos as static files
app.mount("/outputs", StaticFiles(directory=OUTPUT_DIR), name="outputs")

if not os.path.exists(MODEL_PATH):
    raise RuntimeError(f"Model not found at {MODEL_PATH}")

print(f"Loading model from: {MODEL_PATH}")
model = YOLO(MODEL_PATH)
print("Model loaded successfully")

# In-memory jobs cache
jobs = {}

# ================= JOB PERSISTENCE =================
def save_job(job_id: str):
    try:
        job_file = os.path.join(JOBS_DIR, f"{job_id}.json")
        with open(job_file, 'w') as f:
            json.dump(jobs[job_id], f)
    except Exception as e:
        print(f"Warning: failed to save job {job_id}: {e}")

def load_job(job_id: str):
    try:
        job_file = os.path.join(JOBS_DIR, f"{job_id}.json")
        if os.path.exists(job_file):
            with open(job_file, 'r') as f:
                return json.load(f)
    except Exception as e:
        print(f"Warning: failed to load job {job_id}: {e}")
    return None

# ================= HELPER =================
def to_python_type(obj):
    if isinstance(obj, np.generic):
        return obj.item()
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: to_python_type(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [to_python_type(v) for v in obj]
    return obj

# ================= TRACKING CONFIG =================
CONF_THRESH = 0.25
IOU_THRESHOLD = 0.4
MAX_DISAPPEARED = 15
MAX_DISTANCE = 120
MIN_FRAMES_TO_COUNT = 2
OFFSET = 40

class_map = {0: 'mobil', 1: 'bus', 2: 'truk'}

# ================= TRACKING FUNCTIONS =================
def get_lane(cx, frame_width):
    return 'kiri' if cx < frame_width // 2 else 'kanan'

def calculate_iou(box1, box2):
    x1_min, y1_min, x1_max, y1_max = box1
    x2_min, y2_min, x2_max, y2_max = box2
    xi_min = max(x1_min, x2_min)
    yi_min = max(y1_min, y2_min)
    xi_max = min(x1_max, x2_max)
    yi_max = min(y1_max, y2_max)
    if xi_max < xi_min or yi_max < yi_min:
        return 0.0
    intersection = (xi_max - xi_min) * (yi_max - yi_min)
    box1_area = (x1_max - x1_min) * (y1_max - y1_min)
    box2_area = (x2_max - x2_min) * (y2_max - y2_min)
    union = box1_area + box2_area - intersection
    return intersection / union if union > 0 else 0

def check_line_crossing(y_history, line_y, offset=OFFSET, is_first_detection=False, curr_y=None):
    if len(y_history) < 1:
        return False
    if is_first_detection and curr_y is not None:
        if curr_y > line_y and curr_y < (line_y + 200):
            return True
    if len(y_history) < 2:
        return False
    prev_y = y_history[-2]
    curr_y = y_history[-1]
    if prev_y <= line_y and curr_y > line_y:
        return True
    if prev_y < (line_y + offset) and curr_y >= (line_y + offset):
        return True
    if prev_y < line_y and curr_y > (line_y + offset):
        return True
    if len(y_history) >= 3 and y_history[-3] < line_y and curr_y > line_y:
        return True
    return False

def euclidean_distance(p1, p2):
    return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

def register(objects, next_id, centroid, bbox, cls_name, lane):
    objects[next_id] = {
        'centroid': centroid, 'bbox': bbox, 'disappeared': 0,
        'counted': False, 'class': cls_name, 'lane': lane,
        'y_history': [centroid[1]], 'frame_count': 1
    }
    return next_id + 1

def deregister(objects, object_id):
    if object_id in objects:
        del objects[object_id]

def update_tracking(objects, next_id, detections, frame_width, counters, total):
    if len(detections) == 0:
        for oid in list(objects.keys()):
            objects[oid]['disappeared'] += 1
            max_d = 5 if objects[oid]['counted'] else MAX_DISAPPEARED
            if objects[oid]['disappeared'] > max_d:
                deregister(objects, oid)
        return objects, next_id, total

    # Filter overlapping detections by IoU
    filtered = []
    for det in sorted(detections, key=lambda x: x['conf'], reverse=True):
        if all(calculate_iou(det['bbox'], e['bbox']) <= IOU_THRESHOLD for e in filtered):
            filtered.append(det)
    detections = filtered

    if len(objects) == 0:
        for det in detections:
            lane = get_lane(det['centroid'][0], frame_width)
            next_id = register(objects, next_id, det['centroid'], det['bbox'], det['class'], lane)
    else:
        oids = list(objects.keys())
        obj_cents = [objects[oid]['centroid'] for oid in oids]
        inp_cents = [det['centroid'] for det in detections]
        dists = np.array([[euclidean_distance(oc, ic) for ic in inp_cents] for oc in obj_cents])

        if dists.shape[0] > 0 and dists.shape[1] > 0:
            rows = dists.min(axis=1).argsort()
            cols = dists.argmin(axis=1)[rows]
            used_rows, used_cols = set(), set()

            for row, col in zip(rows, cols):
                if row in used_rows or col in used_cols:
                    continue
                if dists[row, col] > MAX_DISTANCE:
                    continue
                oid = oids[row]
                det = detections[col]
                if not objects[oid]['counted']:
                    objects[oid]['class'] = det['class']
                objects[oid].update({
                    'centroid': det['centroid'], 'bbox': det['bbox'],
                    'disappeared': 0, 'lane': get_lane(det['centroid'][0], frame_width)
                })
                objects[oid]['y_history'].append(det['centroid'][1])
                objects[oid]['frame_count'] += 1
                if len(objects[oid]['y_history']) > 15:
                    objects[oid]['y_history'].pop(0)

                if not objects[oid]['counted'] and objects[oid]['frame_count'] >= MIN_FRAMES_TO_COUNT:
                    is_first = objects[oid]['frame_count'] == MIN_FRAMES_TO_COUNT
                    if check_line_crossing(objects[oid]['y_history'], 300, OFFSET, is_first, det['centroid'][1]):
                        objects[oid]['counted'] = True
                        lane = objects[oid]['lane']
                        cls_name = objects[oid]['class']
                        counters[lane]['total'] += 1
                        counters[lane][cls_name] += 1
                        total += 1

                used_rows.add(row)
                used_cols.add(col)

            for row in set(range(dists.shape[0])).difference(used_rows):
                oid = oids[row]
                objects[oid]['disappeared'] += 1
                max_d = 5 if objects[oid]['counted'] else MAX_DISAPPEARED
                if objects[oid]['disappeared'] > max_d:
                    deregister(objects, oid)

            for col in set(range(dists.shape[1])).difference(used_cols):
                det = detections[col]
                lane = get_lane(det['centroid'][0], frame_width)
                next_id = register(objects, next_id, det['centroid'], det['bbox'], det['class'], lane)

    return objects, next_id, total

# ================= VIDEO PROCESSING =================
async def process_video(job_id: str, input_path: str):
    objects = {}
    next_id = 0
    counters = {
        'kiri': {'total': 0, 'mobil': 0, 'bus': 0, 'truk': 0},
        'kanan': {'total': 0, 'mobil': 0, 'bus': 0, 'truk': 0}
    }
    vehicle_count_total = 0
    last_detections = []

    jobs[job_id] = {"status": "processing", "progress": 0, "total": 0, "kiri": 0, "kanan": 0}
    save_job(job_id)

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        jobs[job_id].update({"status": "failed", "progress": -1, "error": "Cannot open video"})
        save_job(job_id)
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
    frame_count = 0
    PROCESS_WIDTH = 640
    counting_line_y = None

    raw_output = os.path.join(tempfile.gettempdir(), f"raw_{job_id}.mp4")
    final_output = os.path.join(OUTPUT_DIR, f"output_{job_id}.mp4")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        h, w = frame.shape[:2]

        if counting_line_y is None:
            counting_line_y = h // 2

        scale = PROCESS_WIDTH / w
        frame_small = cv2.resize(frame, (PROCESS_WIDTH, int(h * scale)))
        results = model.predict(frame_small, device="cpu", conf=CONF_THRESH, verbose=False, max_det=50)

        detections = []
        if results[0].boxes is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            cls_ids = results[0].boxes.cls.cpu().numpy().astype(int)
            confs = results[0].boxes.conf.cpu().numpy()

            for box, cls_id, conf in zip(boxes, cls_ids, confs):
                x1, y1, x2, y2 = box / scale
                cX, cY = int((x1 + x2) / 2), int((y1 + y2) / 2)
                cls_name = class_map.get(cls_id, 'mobil')
                detections.append({'bbox': [x1, y1, x2, y2], 'centroid': (cX, cY), 'class': cls_name, 'conf': conf})

                color = {'mobil': (0, 255, 0), 'bus': (255, 0, 0), 'truk': (0, 165, 255)}.get(cls_name, (255, 255, 255))
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                cv2.putText(frame, f"{cls_name} {conf:.2f}", (int(x1), int(y1) - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        last_detections = detections

        # Draw counting line
        cv2.line(frame, (0, counting_line_y), (w, counting_line_y), (0, 0, 255), 2)
        cv2.putText(frame, f"Total: {vehicle_count_total}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        if writer is None:
            writer = cv2.VideoWriter(raw_output, fourcc, 20.0, (w, h))
        writer.write(frame)

        del frame, frame_small, results

        objects, next_id, vehicle_count_total = update_tracking(objects, next_id, detections, w, counters, vehicle_count_total)

        progress = int((frame_count / total_frames) * 100)
        jobs[job_id].update({
            "progress": progress,
            "total": vehicle_count_total,
            "kiri": counters['kiri']['total'],
            "kanan": counters['kanan']['total'],
        })

        if progress % 10 == 0:
            save_job(job_id)
            gc.collect()

        await asyncio.sleep(0)

    cap.release()
    if writer:
        writer.release()

    # Re-encode for browser compatibility
    ffmpeg_ok = False
    try:
        subprocess.run(
            ["ffmpeg", "-y", "-i", raw_output, "-c:v", "libx264", "-movflags", "+faststart", final_output],
            check=True, capture_output=True
        )
        ffmpeg_ok = True
    except Exception as e:
        print(f"ffmpeg re-encode failed: {e}, using raw output")
        shutil.move(raw_output, final_output)

    if ffmpeg_ok and os.path.exists(raw_output):
        os.remove(raw_output)

    # Delete input temp file
    if os.path.exists(input_path) and input_path.startswith(tempfile.gettempdir()):
        os.remove(input_path)

    gc.collect()

    output_url = f"http://localhost:8000/outputs/output_{job_id}.mp4"

    jobs[job_id].update({
        "status": "completed",
        "progress": 100,
        "completed": True,
        "outputVideoUrl": output_url,
        "vehicle_count": vehicle_count_total,
        "frames_processed": frame_count,
        "lane": {
            "kiri": {**counters['kiri']},
            "kanan": {**counters['kanan']}
        },
        "detections": to_python_type(last_detections),
    })
    save_job(job_id)

    print(f"Done: {vehicle_count_total} vehicles in {frame_count} frames")
    print(f"Output: {output_url}")

# ================= ROUTES =================
@app.get("/ping")
def ping():
    return {"status": "ok", "message": "YOLO API running (local mode)"}

@app.post("/detect")
async def detect_video(file: UploadFile = File(...), background_tasks: BackgroundTasks = None):
    suffix = os.path.splitext(file.filename or ".mp4")[1] or ".mp4"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name

    job_id = str(int(asyncio.get_event_loop().time() * 1000))
    background_tasks.add_task(process_video, job_id, tmp_path)
    return {"job_id": job_id, "message": "Processing started"}

@app.get("/progress/{job_id}")
async def progress(job_id: str):
    async def stream():
        while True:
            job = jobs.get(job_id)
            if not job:
                yield f"data: 0\n\n"
                break
            yield f"data: {job['progress']}\n\n"
            if job["progress"] >= 100 or job["progress"] < 0:
                break
            await asyncio.sleep(0.5)
    return StreamingResponse(stream(), media_type="text/event-stream")

@app.get("/result/{job_id}")
async def result(job_id: str):
    job = jobs.get(job_id) or load_job(job_id)
    if job:
        jobs[job_id] = job

    if not job:
        return JSONResponse({
            "job_id": job_id, "status": "pending", "progress": 0,
            "vehicle_count": 0, "frames_processed": 0,
            "lane": {"kiri": {"total": 0, "mobil": 0, "bus": 0, "truk": 0},
                     "kanan": {"total": 0, "mobil": 0, "bus": 0, "truk": 0}},
            "detections": [], "outputVideoUrl": None
        })

    response = {
        "job_id": job_id,
        "status": "completed" if job.get("completed") else "processing",
        "progress": job.get("progress", 0),
        "vehicle_count": job.get("vehicle_count", 0),
        "frames_processed": job.get("frames_processed", 0),
        "lane": job.get("lane", {"kiri": {"total": 0, "mobil": 0, "bus": 0, "truk": 0},
                                  "kanan": {"total": 0, "mobil": 0, "bus": 0, "truk": 0}}),
        "detections": job.get("detections", []),
        "outputVideoUrl": job.get("outputVideoUrl"),
    }
    return JSONResponse(to_python_type(response))
