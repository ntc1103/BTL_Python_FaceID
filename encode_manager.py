# encode_manager.py
import os, sys, pickle, time
from pathlib import Path
import numpy as np

# ===== CONFIG CHUNG =====
ROOT = Path(r"E:\Python_chung")
ENCODINGS_PKL = ROOT / "encodings" / "encodings_dlib20.pkl"

def clear():
    os.system("cls" if os.name == "nt" else "clear")

def pause():
    input("\nNhấn Enter để quay lại menu...")

# ====== CÁC CHỨC NĂNG CHÍNH ======

def encode_full():
    """Encode lại toàn bộ dataset từ đầu (sử dụng encoding_face.py)."""
    print("⚙️ Đang chạy encode toàn bộ dataset (mất vài phút)...\n")
    os.system(f'python "{ROOT / "encoding_face.py"}"')
    pause()

def encode_sync():
    """Đồng bộ thông minh (tự thêm / xóa / cập nhật)."""
    print("🔄 Đang chạy encode đồng bộ thông minh...\n")
    os.system(f'python "{ROOT / "encode_sync.py"}"')
    pause()

def remove_person():
    """Xóa 1 người cụ thể khỏi file encodings."""
    if not ENCODINGS_PKL.exists():
        print("❌ Không tìm thấy file encodings_dlib20.pkl")
        pause()
        return
    name = input("👤 Nhập tên người cần xóa: ").strip()
    if not name:
        print("⚠️ Tên không hợp lệ.")
        pause()
        return
    data = pickle.load(open(ENCODINGS_PKL, "rb"))
    names = data["names"]
    embeddings = data["embeddings"]

    indices = [i for i, n in enumerate(names) if n.lower() != name.lower()]
    removed = len(names) - len(indices)

    if removed == 0:
        print(f"⚠️ Không tìm thấy '{name}' trong file.")
    else:
        names = [names[i] for i in indices]
        embeddings = embeddings[indices]
        pickle.dump({"names": names, "embeddings": embeddings}, open(ENCODINGS_PKL, "wb"))
        print(f"✅ Đã xóa {removed} vector của '{name}' khỏi file.")
    pause()

def info_file():
    """In thông tin chi tiết file encodings."""
    if not ENCODINGS_PKL.exists():
        print("❌ Không tìm thấy file encodings_dlib20.pkl")
        pause()
        return
    data = pickle.load(open(ENCODINGS_PKL, "rb"))
    names = np.array(data["names"])
    embeddings = np.array(data["embeddings"])
    unique, counts = np.unique(names, return_counts=True)

    print("📊 THÔNG TIN FILE ENCODINGS:")
    print(f"📁 Đường dẫn: {ENCODINGS_PKL}")
    print(f"👥 Tổng vector: {len(names)}")
    print(f"📏 Kích thước embedding: {embeddings.shape}")
    print("\n🧩 Danh sách người:")
    for u, c in zip(unique, counts):
        print(f"   - {u}: {c} vector")
    pause()

# ====== MENU CHÍNH ======
def main_menu():
    while True:
        clear()
        print("===============================")
        print("👤 FACE ENCODE MANAGER")
        print("===============================")
        print("[1] Encode toàn bộ từ đầu")
        print("[2] Đồng bộ thông minh (thêm / xóa / cập nhật)")
        print("[3] Xóa 1 người khỏi file encodings")
        print("[4] Kiểm tra thông tin file .pkl")
        print("[0] Thoát")
        print("===============================")
        choice = input("Chọn thao tác: ").strip()

        if choice == "1": encode_full()
        elif choice == "2": encode_sync()
        elif choice == "3": remove_person()
        elif choice == "4": info_file()
        elif choice == "0":
            print("👋 Thoát chương trình.")
            time.sleep(1)
            break
        else:
            print("⚠️ Lựa chọn không hợp lệ.")
            time.sleep(1)

if __name__ == "__main__":
    main_menu()
