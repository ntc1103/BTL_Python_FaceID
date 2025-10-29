import cv2
import os
import numpy as np
import time
import json
from datetime import datetime
from pathlib import Path

class FaceCollector:
    def __init__(
        self,
        max_images=50,
        min_area=20000,
        max_area=100000,
        radius=120,
        blur_threshold=60,
        capture_delay=1.0,
        preview_size=150,      # đổi về 150 cho khớp dlib.get_face_chip
        save_root="dataset"
    ):
        self.MAX_IMAGES = max_images
        self.MIN_AREA = min_area
        self.MAX_AREA = max_area
        self.RADIUS = radius
        self.DELAY = capture_delay
        self.BLUR_THRESHOLD = blur_threshold
        self.PREVIEW_SIZE = preview_size
        self.SAVE_ROOT = Path(save_root)

        # Haar cascade dùng để hỗ trợ thu thập (encode sẽ dùng dlib)
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        self.last_capture = 0

    # ===== quality helpers =====
    def blur_score(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
        return cv2.Laplacian(gray, cv2.CV_64F).var()

    def enhance_preview(self, img):
        # chỉ phục vụ hiển thị cho dễ canh; KHÔNG dùng ảnh đã xử lý để encode
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        l = self.clahe.apply(l)
        enhanced = cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)
        enhanced = cv2.convertScaleAbs(enhanced, alpha=1.6, beta=50)
        gamma = 0.8
        look_up = np.array([((i / 255.0) ** (1.0 / gamma)) * 255 for i in np.arange(256)]).astype("uint8")
        enhanced = cv2.LUT(enhanced, look_up)
        return enhanced

    def quality_score(self, face_img):
        blur = min(self.blur_score(face_img) / self.BLUR_THRESHOLD, 1.0)
        gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
        brightness = 1.0 - abs(np.mean(gray) - 127) / 127
        contrast = np.std(gray) / 128
        return float((blur + brightness + contrast) / 3)

    # ===== UI =====
    def draw_ui(self, frame, msg, count, quality=None):
        h, w = frame.shape[:2]
        center = (w // 2, h // 2)

        # Guide circle + crosshair
        cv2.circle(frame, center, self.RADIUS, (0, 255, 0), 2)
        cv2.line(frame, (center[0] - 20, center[1]), (center[0] + 20, center[1]), (0, 255, 0), 1)
        cv2.line(frame, (center[0], center[1] - 20), (center[0], center[1] + 20), (0, 255, 0), 1)

        # Progress bar
        progress = count / self.MAX_IMAGES
        cv2.rectangle(frame, (50, h - 60), (w - 50, h - 40), (90, 90, 90), -1)
        cv2.rectangle(frame, (50, h - 60), (50 + int((w - 100) * progress), h - 40), (0, 255, 0), -1)

        # Text
        cv2.putText(frame, msg, (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        cv2.putText(frame, f"{count}/{self.MAX_IMAGES}", (30, h - 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        if quality is not None:
            color = (0, 255, 0) if quality > 0.7 else (0, 255, 255) if quality > 0.5 else (0, 0, 255)
            cv2.putText(frame, f"Quality: {quality:.2f}", (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # ===== main collect =====
    def collect(self, name):
        name = name.strip()
        if not name:
            print("❌ Name required")
            return False

        # Create directories
        person_root = self.SAVE_ROOT / name
        raw_dir = person_root / "raw"
        proc_dir = person_root / "processed"
        raw_dir.mkdir(parents=True, exist_ok=True)
        proc_dir.mkdir(parents=True, exist_ok=True)

        # Open camera (set properties khi được driver hỗ trợ)
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        cap.set(cv2.CAP_PROP_FPS, 30)
        # Một số driver không chấp nhận các option dưới, không sao:
        cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)
        cap.set(cv2.CAP_PROP_EXPOSURE, -4)

        count = 0
        stats = {"saved": 0, "rejected": 0}

        print(f"📸 Collecting faces for: {name}")
        print("Controls: Q-quit, SPACE-pause, R-reset")

        paused = False
        while count < self.MAX_IMAGES:
            if not paused:
                ret, frame = cap.read()
                if not ret:
                    print("❌ Cannot read from camera")
                    break

                frame = cv2.flip(frame, 1)
                preview = self.enhance_preview(frame)  # chỉ để nhìn
                gray = cv2.cvtColor(preview, cv2.COLOR_BGR2GRAY)

                h, w = frame.shape[:2]
                center = (w // 2, h // 2)
                faces = self.face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(80, 80))

                msg = "📍 Align face in circle"
                quality = None

                if len(faces) == 1:
                    x, y, fw, fh = faces[0]
                    face_center = (x + fw // 2, y + fh // 2)
                    area = fw * fh

                    dx = face_center[0] - center[0]
                    dy = face_center[1] - center[1]
                    in_circle = dx * dx + dy * dy <= self.RADIUS * self.RADIUS

                    if in_circle:
                        if area < self.MIN_AREA:
                            msg = "📏 Move closer"
                        elif area > self.MAX_AREA:
                            msg = "📏 Move farther"
                        else:
                            # Quality trên ảnh CẮT từ khung gốc (không dùng preview)
                            face_img = frame[y : y + fh, x : x + fw]
                            quality = self.quality_score(face_img)

                            if time.time() - self.last_capture >= self.DELAY and quality > 0.35:
                                count += 1
                                stats["saved"] += 1

                                # Lưu RAW (cho encode)
                                raw_path = raw_dir / f"{count:04d}.jpg"
                                cv2.imwrite(str(raw_path), face_img)

                                # Lưu bản xem trước 150x150 (để QA, không dùng để encode)
                                proc = cv2.resize(face_img, (self.PREVIEW_SIZE, self.PREVIEW_SIZE))
                                proc_path = proc_dir / f"{count:04d}.jpg"
                                cv2.imwrite(str(proc_path), proc)

                                self.last_capture = time.time()
                                msg = f"✅ Captured {count}"
                                print(f"Saved {count:04d} (Q: {quality:.2f})")
                            else:
                                msg = f"⏳ Ready... Q:{quality:.2f}" if quality is not None else "⏳ Ready..."
                                if quality is not None and quality <= 0.35:
                                    stats["rejected"] += 1
                    else:
                        msg = "🎯 Center face in circle"

                    color = (0, 255, 0) if (quality and quality > 0.7) else (0, 255, 255) if (quality and quality > 0.5) else (0, 0, 255)
                    cv2.rectangle(frame, (x, y), (x + fw, y + fh), color, 2)

                elif len(faces) > 1:
                    msg = "👥 Only one face"

                self.draw_ui(frame, msg, count, quality)
                cv2.imshow("Face Collector", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord(" "):
                paused = not paused
                print("⏸️ Paused" if paused else "▶️ Resumed")
            elif key == ord("r"):
                count = 0
                stats = {"saved": 0, "rejected": 0}
                print("🔄 Reset")

        cap.release()
        cv2.destroyAllWindows()

        metadata = {
            "name": name,
            "date": datetime.now().isoformat(),
            "images": count,
            "stats": stats,
        }
        with open(person_root / "metadata.json", "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)

        print(f"\n✅ Collected {count} images for {name}")
        print(f"📊 Saved: {stats['saved']}, Rejected: {stats['rejected']}")
        return count > 0


if __name__ == "__main__":
    name = input("👤 Enter name: ").strip()
    if name:
        FaceCollector().collect(name)
    else:
        print("❌ Name required")
