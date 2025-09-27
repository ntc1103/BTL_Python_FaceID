import cv2, os, sys

print("== Diagnostics ==")
print("cv2 version  :", cv2.__version__)
print("cv2 path     :", cv2.__file__)
haar_dir = cv2.data.haarcascades
xml_name = "haarcascade_frontalface_default.xml"
xml_path = os.path.join(haar_dir, xml_name)
print("haar dir     :", haar_dir)
print("xml exists?  :", os.path.exists(xml_path), "-", xml_path)

# 1) Load cascade và kiểm tra .empty()
face_cascade = cv2.CascadeClassifier(xml_path)
if face_cascade.empty():
    print("\n[ERROR] Không load được CascadeClassifier.")
    print("=> Thử cài lại gói hoặc kiểm tra quyền đọc file:")
    print("   python -m pip install --force-reinstall opencv-contrib-python")
    sys.exit(1)
else:
    print("[OK] CascadeClassifier load thành công.")

# 2) Mở webcam (nếu không có webcam, thoát với thông báo rõ ràng)
cap = cv2.VideoCapture(0)  # thay 1 nếu bạn dùng camera phụ
if not cap.isOpened():
    print("\n[ERROR] Không mở được webcam (VideoCapture(0) fail).")
    print("=> Kiểm tra quyền camera / chọn đúng index / thử file ảnh thay thế.")
    sys.exit(1)

print("\nNhấn 'q' để thoát.")
while True:
    ok, frame = cap.read()
    if not ok:
        print("[ERROR] Không đọc được khung hình từ webcam.")
        break

    # 3) Convert grayscale và detect
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,      # tăng lên 1.2 nếu nhiều nhiễu
        minNeighbors=5,       # giảm xuống 3 nếu ít phát hiện
        minSize=(60, 60)      # thử (30,30) nếu ngồi xa
    )

    # 4) Vẽ bbox
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0,255,0), 2)

    # 5) Hiển thị
    cv2.imshow("Face Detection (Haar)", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

