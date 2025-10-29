# remove_person.py
import pickle
from pathlib import Path

ROOT = Path(r"E:\Python_chung")
OUT_PKL = ROOT / "encodings" / "encodings_dlib20.pkl"

def remove_person(person_name):
    if not OUT_PKL.exists():
        print("❌ Không tìm thấy file encodings.")
        return

    data = pickle.load(open(OUT_PKL, "rb"))
    names = data["names"]
    embeddings = data["embeddings"]

    indices = [i for i, n in enumerate(names) if n.lower() != person_name.lower()]
    removed = len(names) - len(indices)

    if removed == 0:
        print(f"⚠️ Không tìm thấy '{person_name}' trong file encodings.")
        return

    # Cập nhật dữ liệu sau khi xoá
    names = [names[i] for i in indices]
    embeddings = embeddings[indices]

    pickle.dump({"names": names, "embeddings": embeddings}, open(OUT_PKL, "wb"))
    print(f"✅ Đã xoá {removed} vector của '{person_name}' khỏi encodings_dlib20.pkl")

if __name__ == "__main__":
    person = input("👤 Nhập tên người cần xoá: ").strip()
    if person:
        remove_person(person)