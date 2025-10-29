import os
import pickle
import numpy as np
from collections import defaultdict

def _euclid(a, b):
    return np.linalg.norm(a - b)

def _dedup_person_vectors(vecs: np.ndarray, dup_tol: float = 1e-2) -> np.ndarray:
    """
    Khử trùng lặp vector cho 1 người.
    Giữ lại một tập đại diện sao cho mọi cặp còn lại cách nhau > dup_tol (Euclid).
    dup_tol gợi ý: 1e-2 (0.01). Có thể tăng lên 0.02 nếu dữ liệu quá trùng.
    """
    if len(vecs) <= 1:
        return vecs
    kept = []
    for v in vecs:
        if not kept:
            kept.append(v)
            continue
        dmin = min(_euclid(v, u) for u in kept)
        if dmin > dup_tol:
            kept.append(v)
    return np.vstack(kept).astype(np.float32)

def load_all_encodings(
    encoding_dir: str = "encodings",
    dedup: bool = True,
    dup_tol: float = 1e-2,
    ensure_dir: bool = True
):
    """
    Trả về: (known_encodings: np.ndarray[N,128], known_names: List[str])
    - Đọc mọi file .pkl trong encoding_dir
    - Hỗ trợ 2 format:
        + dlib20: {"names": [...], "embeddings": np.ndarray (N,128)}
        + cũ:     {"name": "...", "encodings": List[128D]}
    - Thống kê đếm theo người
    - Khử trùng lặp vector trong từng người (nếu dedup=True)
    - Tự tạo thư mục encoding_dir nếu thiếu (ensure_dir=True). Không tạo file rỗng.
    """
    if ensure_dir and not os.path.exists(encoding_dir):
        os.makedirs(encoding_dir, exist_ok=True)

    files = [f for f in os.listdir(encoding_dir) if f.endswith(".pkl")]
    if not files:
        print(f"⚠️ Không thấy file .pkl trong '{encoding_dir}'. Hãy encode trước.")
        return np.array([]), []

    # Tạm gom thô
    raw_names = []
    raw_vecs = []

    for f in files:
        path = os.path.join(encoding_dir, f)
        try:
            with open(path, "rb") as fh:
                data = pickle.load(fh)

            if "embeddings" in data and "names" in data:
                encs = data["embeddings"]
                names = data["names"]
            elif "encodings" in data and "name" in data:
                encs = data["encodings"]
                names = [data["name"]] * len(encs)
            else:
                print(f"⚠️ {f} không đúng cấu trúc, bỏ qua.")
                continue

            if isinstance(encs, list):
                encs = np.array(encs, dtype=np.float32)
            else:
                encs = encs.astype(np.float32)

            raw_vecs.append(encs)
            raw_names.extend(names)
            print(f"✅ Loaded {len(names)} vectors từ {f}")

        except Exception as e:
            print(f"❌ Lỗi đọc {f}: {e}")

    if not raw_vecs:
        print("⚠️ Không có vector nào hợp lệ được nạp.")
        return np.array([]), []

    all_vecs = np.vstack(raw_vecs).astype(np.float32)
    all_names = np.array(raw_names)

    # Thống kê ban đầu
    unique, counts = np.unique(all_names, return_counts=True)
    print("\n📊 Thống kê ban đầu (chưa khử trùng lặp):")
    for u, c in zip(unique, counts):
        print(f"   - {u}: {c} vector")
    print(f"📦 Tổng: {len(all_names)} vector, {len(unique)} người")

    # Khử trùng lặp theo từng người (tùy chọn)
    if dedup:
        per_person = defaultdict(list)
        for name, v in zip(all_names, all_vecs):
            per_person[name].append(v)

        kept_names = []
        kept_vecs = []

        removed_total = 0
        for name, vec_list in per_person.items():
            vec_arr = np.vstack(vec_list).astype(np.float32)
            before = len(vec_arr)
            vec_dedup = _dedup_person_vectors(vec_arr, dup_tol=dup_tol)
            after = len(vec_dedup)
            removed = before - after
            removed_total += removed
            kept_vecs.append(vec_dedup)
            kept_names.extend([name] * after)
            if removed > 0:
                print(f"🧹 Dedup '{name}': {before} → {after} (loại {removed})")

        if kept_vecs:
            all_vecs = np.vstack(kept_vecs).astype(np.float32)
            all_names = np.array(kept_names)
            print(f"\n🧾 Tổng số vector loại bỏ do trùng: {removed_total}")
            unique2, counts2 = np.unique(all_names, return_counts=True)
            print("📊 Thống kê sau khử trùng lặp:")
            for u, c in zip(unique2, counts2):
                print(f"   - {u}: {c} vector")
            print(f"📦 Tổng: {len(all_names)} vector, {len(unique2)} người")

    return all_vecs, list(all_names)
