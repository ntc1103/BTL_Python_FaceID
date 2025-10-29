# recognize_checkin_checkout_fast.py  — tối ưu giảm lag (Dlib-only, HOG)
# Các cải tiến chính:
# - Đọc camera bằng DirectShow + BUFFERSIZE=1 để giảm độ trễ.
# - Threaded camera reader để UI mượt, xử lý tách luồng.
# - Phát hiện mặt trên ảnh thu nhỏ (DETECT_SCALE), upsample=0.
# - Chỉ mã hóa/so khớp mỗi N khung (PROCESS_EVERY) thay vì mọi frame.
# - Dùng compute_face_descriptor trực tiếp trên ảnh gốc (không get_face_chip).
# - Tối ưu so khớp bằng khoảng cách L2 bình phương (không cần sqrt).
# - Giữ nguyên logic check-in/out tổng thời gian.

import os
import csv
import cv2
import pickle
import numpy as np
import dlib
import threading
import time
from datetime import datetime
from collections import defaultdict, deque
from pathlib import Path
from encoding_loaded import load_all_encodings

# ===================== CẤU HÌNH =====================
ENCODING_DIR = "encodings"                    # nơi lưu *.pkl (không dùng trực tiếp ở file này)
ATTENDANCE_CSV = "logs/attendance.csv"        # file chấm công
TOLERANCE = 0.5                                # ngưỡng khớp (càng nhỏ càng chặt)
TOLERANCE2 = TOLERANCE * TOLERANCE             # dùng so sánh bình phương khoảng cách
DEBOUNCE_SECONDS = 3                           # chống bấm phím liên tục

# Tối ưu hiển thị/nhận diện
FRAME_WIDTH = 640
FRAME_HEIGHT = 360
TARGET_FPS = 30
PROCESS_EVERY = 3        # chỉ nhận diện mỗi 2 khung (giảm tải ~50%)
DETECT_SCALE = 0.4       # phát hiện mặt trên ảnh thu nhỏ (0.5 = 1/2 cạnh ~ 1/4 số điểm ảnh)
USE_CNN = False          # ép dùng HOG cho nhanh, dù có CNN model

# Model paths (đặt theo cấu trúc trước đó)
ROOT = Path(__file__).resolve().parent
MODELS_DIR = ROOT / "models"
PREDICTOR_PATH = MODELS_DIR / "shape_predictor_5_face_landmarks.dat"
RECOG_MODEL_PATH = MODELS_DIR / "dlib_face_recognition_resnet_model_v1.dat"
CNN_PATH = MODELS_DIR / "mmod_human_face_detector.dat"  # tùy chọn

os.makedirs(os.path.dirname(ATTENDANCE_CSV), exist_ok=True)

CSV_FIELDS = ["date", "id", "name", "check_in", "check_out", "total_seconds", "total_hhmm"]

# ===================== DLIB INIT (không dùng face_recognition) =====================
def _require(p: Path, hint: str):
    if not p.exists():
        raise FileNotFoundError(f"Missing model: {p}\nHint: {hint}")

_require(PREDICTOR_PATH,  "Đặt shape_predictor_5_face_landmarks.dat vào thư mục models/")
_require(RECOG_MODEL_PATH,"Đặt dlib_face_recognition_resnet_model_v1.dat vào thư mục models/")

# Detector: ưu tiên HOG để nhanh; CNN có thể rất nặng
_hog = dlib.get_frontal_face_detector()
if USE_CNN and CNN_PATH.exists():
    _cnn = dlib.cnn_face_detection_model_v1(str(CNN_PATH))
else:
    _cnn = None

PRED = dlib.shape_predictor(str(PREDICTOR_PATH))
REC  = dlib.face_recognition_model_v1(str(RECOG_MODEL_PATH))


def _detect_rects_small(rgb):
    """Phát hiện mặt trên ảnh thu nhỏ -> trả về list dlib.rectangle theo tọa độ của ảnh GỐC."""
    h, w = rgb.shape[:2]
    if DETECT_SCALE != 1.0:
        small = cv2.resize(rgb, (int(w * DETECT_SCALE), int(h * DETECT_SCALE)))
    else:
        small = rgb

    if _cnn is not None:
        # CNN có thể chạy chậm; vẫn để upsample=0 cho nhanh nhất
        dets = _cnn(small, 0)
        rects_small = [d.rect for d in dets]
    else:
        # HOG nhanh nhất, upsample=0
        rects_small = _hog(small, 0)

    # Scale back lên ảnh gốc
    if DETECT_SCALE != 1.0:
        inv = 1.0 / DETECT_SCALE
        rects = []
        for r in rects_small:
            l = int(r.left() * inv)
            t = int(r.top() * inv)
            rgt = int(r.right() * inv)
            btm = int(r.bottom() * inv)
            rects.append(dlib.rectangle(l, t, rgt, btm))
        return rects
    else:
        return list(rects_small)


def _encode_one_rgb(rgb, rect):
    """Encode 1 khuôn mặt (RGB + dlib.rectangle) -> vector 128D (np.float32)
    Dùng trực tiếp ảnh gốc + landmarks để nhanh hơn (không get_face_chip).
    """
    shape = PRED(rgb, rect)
    vec = REC.compute_face_descriptor(rgb, shape, num_jitters=0)  # nhanh hơn
    return np.asarray(vec, dtype=np.float32)


# ===================== Tách tên và ID =====================
def split_name_id(full_name: str):
    parts = full_name.rsplit("_", 1)
    if len(parts) == 2:
        return parts[0].strip(), parts[1].strip()
    return full_name.strip(), ""

# ===================== TIỆN ÍCH THỜI GIAN/CSV =====================
def _today_str():
    return datetime.now().strftime("%Y-%m-%d")


def _time_str():
    return datetime.now().strftime("%H:%M:%S")


def _ensure_csv_headers(path):
    if not os.path.exists(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
            w.writeheader()


def _load_attendance_all(path):
    _ensure_csv_headers(path)
    rows = []
    with open(path, "r", newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            rows.append(row)
    return rows


def _save_attendance_all(path, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        w.writeheader()
        for row in rows:
            w.writerow(row)


def _parse_hms(s: str):
    if not s:
        return None
    try:
        return datetime.strptime(s, "%H:%M:%S").time()
    except Exception:
        return None


def _sec_to_hhmm(sec: int) -> str:
    if sec is None or sec < 0:
        return ""
    m, _ = divmod(sec, 60)
    h, m = divmod(m, 60)
    return f"{h:02d}:{m:02d}"


def _compute_total_seconds(check_in: str, check_out: str) -> int | None:
    tin = _parse_hms(check_in)
    tout = _parse_hms(check_out)
    if not tin or not tout:
        return None
    today = datetime.now().date()
    dt_in = datetime.combine(today, tin)
    dt_out = datetime.combine(today, tout)
    delta = int((dt_out - dt_in).total_seconds())
    return max(delta, 0)

# ===================== CHECK-IN / CHECK-OUT =====================
def check_in(full_name: str) -> str:
    name_display, id_code = split_name_id(full_name)
    rows = _load_attendance_all(ATTENDANCE_CSV)
    today = _today_str()
    now = _time_str()

    idx = next((i for i, r in enumerate(rows)
                if r["date"] == today and r.get("id") == id_code), None)

    if idx is None:
        rows.append({
            "date": today,
            "id": id_code,
            "name": name_display,
            "check_in": now,
            "check_out": "",
            "total_seconds": "",
            "total_hhmm": ""
        })
        _save_attendance_all(ATTENDANCE_CSV, rows)
        return f"✅ CHECK-IN: {name_display} ({id_code}) @ {now}"
    else:
        if not rows[idx]["check_in"]:
            rows[idx]["check_in"] = now
            rows[idx]["total_seconds"] = ""
            rows[idx]["total_hhmm"] = ""
            _save_attendance_all(ATTENDANCE_CSV, rows)
            return f"✅ CHECK-IN (update): {name_display} ({id_code}) @ {now}"
        else:
            return f"ℹ️ You have already checked in before: {name_display} ({id_code}) @ {rows[idx]['check_in']}"


def check_out(full_name: str) -> str:
    name_display, id_code = split_name_id(full_name)
    rows = _load_attendance_all(ATTENDANCE_CSV)
    today = _today_str()
    now = _time_str()

    idx = next((i for i, r in enumerate(rows)
                if r["date"] == today and r.get("id") == id_code), None)

    if idx is None:
        rows.append({
            "date": today,
            "id": id_code,
            "name": name_display,
            "check_in": "",
            "check_out": now,
            "total_seconds": "",
            "total_hhmm": ""
        })
        _save_attendance_all(ATTENDANCE_CSV, rows)
        return f"✅ CHECK-OUT (new row): {name_display} ({id_code}) @ {now} (chưa có check-in, không tính tổng)"
    else:
        if not rows[idx]["check_out"]:
            rows[idx]["check_out"] = now
            total_sec = _compute_total_seconds(rows[idx]["check_in"], rows[idx]["check_out"]) if rows[idx]["check_in"] else None
            rows[idx]["total_seconds"] = "" if total_sec is None else str(total_sec)
            rows[idx]["total_hhmm"] = "" if total_sec is None else _sec_to_hhmm(total_sec)
            _save_attendance_all(ATTENDANCE_CSV, rows)
            if total_sec is None:
                return f"✅ CHECK-OUT: {name_display} ({id_code}) @ {now} (⚠️ thiếu check-in)"
            else:
                return f"✅ CHECK-OUT: {name_display} ({id_code}) @ {now} | Tổng: {_sec_to_hhmm(total_sec)}"
        else:
            return f"ℹ️ You have already checked out before: {name_display} ({id_code}) @ {rows[idx]['check_out']}"


def today_total_for(full_name: str) -> tuple[int | None, str]:
    name_display, id_code = split_name_id(full_name)
    rows = _load_attendance_all(ATTENDANCE_CSV)
    today = _today_str()
    row = next((r for r in rows if r["date"] == today and r.get("id") == id_code), None)
    if not row:
        return None, ""
    if row.get("total_seconds"):
        sec = int(row["total_seconds"])
        return sec, _sec_to_hhmm(sec)
    return None, ""

# ===================== Camera Reader (Thread) =====================
class CameraReader:
    def __init__(self, src=0):
        self.cap = cv2.VideoCapture(src, cv2.CAP_DSHOW)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
        self.cap.set(cv2.CAP_PROP_FPS, TARGET_FPS)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.q = deque(maxlen=1)
        self.stopped = False
        self.t = threading.Thread(target=self._update, daemon=True)
        self.t.start()

    def _update(self):
        while not self.stopped:
            ok, frame = self.cap.read()
            if ok:
                self.q.append(frame)

    def read(self):
        return self.q[-1] if self.q else None

    def release(self):
        self.stopped = True
        time.sleep(0.05)
        self.cap.release()


# ===================== VÒNG LẶP CAMERA (TỰ ĐỘNG CHẤM CÔNG) =====================
def main():
    # Load known encodings
    known_encodings, known_names = load_all_encodings()
    if not isinstance(known_encodings, np.ndarray):
        known_encodings = np.array(known_encodings, dtype=np.float32)
    else:
        known_encodings = known_encodings.astype(np.float32, copy=False)
    known_names = np.array(known_names)

    if known_encodings.size == 0:
        print("❌ Không có dữ liệu encodings. Hãy encode trước.")
        return

    cam = CameraReader(0)
    time.sleep(0.15)  # chờ ổn định buffer

    print("🎥 Camera sẵn sàng. Hệ thống chấm công tự động (1 phút/lần). Nhấn [q] để thoát.")
    last_action_time = defaultdict(lambda: datetime.min)
    i = 0
    boxes_cache = []

    while True:
        frame = cam.read()
        if frame is None:
            continue

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        if i % PROCESS_EVERY == 0:
            rects = _detect_rects_small(rgb)
            boxes_cache = []
            current_name = "Unknown"

            for rect in rects:
                l, t, rgt, btm = rect.left(), rect.top(), rect.right(), rect.bottom()
                if l < 0 or t < 0 or rgt <= l or btm <= t:
                    continue

                vec = _encode_one_rgb(rgb, rect)

                diff = known_encodings - vec
                d2 = np.einsum('ij,ij->i', diff, diff)
                min_idx = int(np.argmin(d2)) if d2.size else None

                name = "Unknown"
                color = (0, 0, 255)

                if min_idx is not None and d2[min_idx] <= TOLERANCE2:
                    name = str(known_names[min_idx])
                    color = (0, 255, 0)
                    current_name = name

                    # ======== AUTO CHECK-IN / CHECK-OUT ========
                    now = datetime.now()
                    dt_last = last_action_time[name]
                    if (now - dt_last).total_seconds() >= 60:  # cách nhau >= 1 phút
                        rows = _load_attendance_all(ATTENDANCE_CSV)
                        today = _today_str()
                        idx = next((i for i, r in enumerate(rows)
                                    if r["date"] == today and r.get("id") == split_name_id(name)[1]), None)

                        if idx is None or not rows[idx]["check_in"]:
                            msg = check_in(name)
                        elif not rows[idx]["check_out"]:
                            msg = check_out(name)
                        else:
                            msg = f"ℹ️ {name} đã check-in & check-out hôm nay."
                        print(msg)
                        last_action_time[name] = now

                boxes_cache.append((rect, name, color))

        # Vẽ khung khuôn mặt
        for rect, name, color in boxes_cache:
            l, t, rgt, btm = rect.left(), rect.top(), rect.right(), rect.bottom()
            cv2.rectangle(frame, (l, t), (rgt, btm), color, 2)
            disp, pid = split_name_id(name)
            label = disp if not pid else f"{disp} ({pid})"
            cv2.putText(frame, label, (l, t - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        # Hiển thị trạng thái footer
        h, w = frame.shape[:2]
        cv2.putText(frame, "AUTO MODE (1p delay) - press [q] to quit",
                    (10, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (50, 220, 50), 2)

        cv2.imshow("Attendance - Auto Mode (FAST)", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:
            break

        i += 1

    cam.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
