# encode_sync.py
from pathlib import Path
import sys, pickle
import numpy as np
import cv2, dlib, os

# ===== CONFIG =====
ROOT    = Path(r"E:\BTL_Python")
DATASET = ROOT / "dataset"
OUT_PKL = ROOT / "encodings" / "encodings_dlib20.pkl"

# Model paths
MODELS          = ROOT / "models"
PREDICTOR_PATH  = MODELS / "shape_predictor_5_face_landmarks.dat"
RECOG_MODEL_PATH= MODELS / "dlib_face_recognition_resnet_model_v1.dat"
CNN_PATH        = MODELS / "mmod_human_face_detector.dat"

# ===== UTILS =====
def require_file(p: Path, hint: str):
    if not p.exists():
        print(f"❌ Missing: {p}\n👉 {hint}")
        sys.exit(1)

require_file(PREDICTOR_PATH,  "Đặt shape_predictor_5_face_landmarks.dat vào models/")
require_file(RECOG_MODEL_PATH,"Đặt dlib_face_recognition_resnet_model_v1.dat vào models/")

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

# ===== MAIN SYNC =====
def sync_encodings():
    OUT_PKL.parent.mkdir(parents=True, exist_ok=True)

    # Load existing data if any
    if OUT_PKL.exists():
        data = pickle.load(open(OUT_PKL, "rb"))
        old_names = list(data["names"])
        old_embeds = np.array(data["embeddings"], dtype=np.float32)
        print(f"📦 Đã tải file cũ ({len(old_names)} vector, {len(set(old_names))} người).")
    else:
        old_names, old_embeds = [], np.empty((0,128), np.float32)
        print("📁 Không có file cũ — sẽ tạo mới.")

    existing_people = set(old_names)
    dataset_people = {p.name for p in DATASET.iterdir() if p.is_dir()}

    # Determine changes
    removed_people = existing_people - dataset_people
    new_people     = dataset_people - existing_people
    common_people  = existing_people & dataset_people

    print("\n🔍 Kết quả so sánh dataset và file:")
    print(f"🆕 Người mới: {list(new_people) or 'Không có'}")
    print(f"🗑️ Người bị xóa: {list(removed_people) or 'Không có'}")
    print(f"📂 Giữ nguyên: {list(common_people) or 'Không có'}\n")

    new_names, new_embeds = [], []

    # Keep people who remain
    for n, e in zip(old_names, old_embeds):
        if n not in removed_people:
            new_names.append(n)
            new_embeds.append(e)

    # Encode new or updated people
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
    added = 0
    for person_dir in sorted(DATASET.iterdir()):
        if not person_dir.is_dir(): continue
        if person_dir.name not in dataset_people: continue

        raw_dir = person_dir / "raw"
        if not raw_dir.exists():
            print(f"⚠️  Bỏ qua {person_dir.name} (không có 'raw')")
            continue

        files = [p for p in sorted(raw_dir.iterdir()) if p.suffix.lower() in exts]
        if not files:
            print(f"⚠️  Bỏ qua {person_dir.name} (không có ảnh)")
            continue

        # Nếu là người mới, hoặc ảnh trong raw thay đổi thì encode lại
        if person_dir.name in new_people or _is_folder_updated(person_dir):
            print(f"⏳ Encode lại cho {person_dir.name} ...")
            ok = 0
            for img_path in files:
                img = cv2.imread(str(img_path))
                vec = encode_one(img)
                if vec is None: continue
                new_names.append(person_dir.name)
                new_embeds.append(vec)
                ok += 1; added += 1
            print(f"[OK] {person_dir.name}: {ok}/{len(files)} ảnh dùng được")

    # Save back
    arr = np.vstack(new_embeds).astype(np.float32)
    pickle.dump({"names": new_names, "embeddings": arr}, open(OUT_PKL, "wb"))

    print("\n✅ Đã đồng bộ xong!")
    print(f"👥 Tổng vector: {len(new_names)} | Người mới thêm: {len(new_people)} | Người xoá: {len(removed_people)}")
    print(f"💾 Lưu vào: {OUT_PKL}")

def _is_folder_updated(folder):
    """Kiểm tra xem thư mục có thay đổi gần đây (ảnh mới/chỉnh sửa)"""
    latest_mod = max((f.stat().st_mtime for f in folder.glob("raw/*") if f.is_file()), default=0)
    # Nếu mới sửa trong vòng 2 phút -> xem như thay đổi
    import time
    return (time.time() - latest_mod) < 120

if __name__ == "__main__":
    sync_encodings()