# encoding_face.py
# Encode khuôn mặt bằng dlib 20 từ ảnh RAW
from pathlib import Path
import sys, pickle
import numpy as np
import cv2, dlib

# ===== CONFIG =====
ROOT    = Path(r"E:\BTL_Python")           # chỉnh nếu bạn để dự án chỗ khác
DATASET = ROOT / "dataset"                   # dataset/<ten>/raw/*.jpg
OUT_PKL = ROOT / "encodings" / "encodings_dlib20.pkl"

# Models nằm trong thư mục models/
MODELS          = ROOT / "models"
PREDICTOR_PATH  = MODELS / "shape_predictor_5_face_landmarks.dat"
RECOG_MODEL_PATH= MODELS / "dlib_face_recognition_resnet_model_v1.dat"
CNN_PATH        = MODELS / "mmod_human_face_detector.dat"   # tùy chọn

def require_file(p: Path, hint: str):
    if not p.exists():
        print(f"❌ Missing: {p}\n👉 {hint}")
        sys.exit(1)

require_file(PREDICTOR_PATH,  "Đặt shape_predictor_5_face_landmarks.dat vào E:\\BTL_Python\\models\\")
require_file(RECOG_MODEL_PATH,"Đặt dlib_face_recognition_resnet_model_v1.dat vào E:\\BTL_Python\\models\\")

# ===== INIT DLIB =====
USE_CNN = CNN_PATH.exists()
if USE_CNN:
    _cnn = dlib.cnn_face_detection_model_v1(str(CNN_PATH))
    def detect_rects(rgb): return [d.rect for d in _cnn(rgb, 1)]
else:
    _hog = dlib.get_frontal_face_detector()
    def detect_rects(rgb): return _hog(rgb, 1)

PRED = dlib.shape_predictor(str(PREDICTOR_PATH))
REC  = dlib.face_recognition_model_v1(str(RECOG_MODEL_PATH))

def _largest(rects):
    return max(rects, key=lambda r: r.width()*r.height()) if rects else None

def encode_one(bgr):
    if bgr is None or bgr.size == 0: return None
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    r = _largest(detect_rects(rgb))
    if r is None: return None
    shape = PRED(rgb, r)
    chip  = dlib.get_face_chip(rgb, shape, size=150)
    vec   = REC.compute_face_descriptor(chip)
    return np.asarray(vec, dtype=np.float32)

def build():
    OUT_PKL.parent.mkdir(parents=True, exist_ok=True)
    if not DATASET.exists():
        print(f"❌ Không thấy thư mục dataset: {DATASET}"); sys.exit(1)

    exts = {".jpg",".jpeg",".png",".bmp",".tif",".tiff",".webp"}
    names, vecs = [], []
    total, used = 0, 0

    for person_dir in sorted(p for p in DATASET.iterdir() if p.is_dir()):
        raw_dir = person_dir / "raw"
        if not raw_dir.exists():
            print(f"⚠️  Bỏ qua {person_dir.name} (không có 'raw')"); continue
        files = [p for p in sorted(raw_dir.iterdir()) if p.suffix.lower() in exts]
        if not files:
            print(f"⚠️  Bỏ qua {person_dir.name} (không có ảnh)"); continue

        ok = 0
        for img_path in files:
            total += 1
            img = cv2.imread(str(img_path))
            vec = encode_one(img)
            if vec is None: continue
            names.append(person_dir.name); vecs.append(vec); ok += 1
        used += ok
        print(f"[OK] {person_dir.name}: {ok}/{len(files)} ảnh dùng được")

    if not vecs:
        print("❌ Không thu được embedding nào."); sys.exit(1)

    arr = np.vstack(vecs).astype(np.float32)
    with open(OUT_PKL, "wb") as f:
        pickle.dump({"names": names, "embeddings": arr}, f)

    print("\n✅ DONE")
    print(f"🖼️  Tổng ảnh duyệt: {total} | Ảnh dùng được: {used}")
    print(f"👥  Tổng vector: {len(names)} | Shape: {arr.shape}")
    print(f"💾 Saved: {OUT_PKL}")

if __name__ == "__main__":
    build()
