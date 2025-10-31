# app/ui_update.py
import tkinter as tk
from tkinter import ttk, messagebox, StringVar, filedialog, simpledialog
from datetime import datetime, date
import subprocess, sys
from pathlib import Path
import shutil

import bcrypt
import pandas as pd
import face_recognition
import numpy as np

# App modules
from app.db import DB
from app.capture_faces import FaceCollector
from app.attendance_cam import run_manual_attendance
from app.config import ROOT
from app.encoding_loaded import load_all_encodings

# -------------------- Colors/Styles --------------------
PRIMARY = "#3b82f6"
PRIMARY_DARK = "#2563eb"
BG = "#f6f9ff"
CARD = "#ffffff"
SUBTEXT = "#50616a"

# -------------------- Login Window --------------------
class LoginWindow(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Đăng nhập — Ứng dụng Chấm công FaceID")
        self.geometry("460x400")
        self.configure(bg=BG)
        self.resizable(False, False)
        self._build_ui()

    def _build_ui(self):
        frm = ttk.Frame(self, padding=18, style="Card.TFrame")
        frm.place(relx=0.5, rely=0.5, anchor="center")

        ttk.Label(frm, text="Ứng dụng Chấm công nhận diện khuôn mặt",
                  font=("Segoe UI Semibold", 12), background=CARD).pack(pady=(0,10))

        ttk.Label(frm, text="Tên đăng nhập:", background=CARD).pack(anchor="w", pady=(6,0))
        self.username = ttk.Entry(frm, width=36)
        self.username.pack(pady=(0,6))

        ttk.Label(frm, text="Mật khẩu:", background=CARD).pack(anchor="w", pady=(6,0))
        self.password = ttk.Entry(frm, show="*", width=36)
        self.password.pack(pady=(0,6))

        ttk.Label(frm, text="Vai trò:", background=CARD).pack(anchor="w", pady=(6,0))
        self.role_var = StringVar(value="Admin")
        role_cb = ttk.Combobox(frm, textvariable=self.role_var, values=["Admin", "Staff"],
                               state="readonly", width=34)
        role_cb.pack(pady=(0,10))

        hint = ("Tài khoản: chỉ Admin đăng nhập app.\n- Staff dùng nút 'Mở chấm công (Staff)'.")
        ttk.Label(frm, text=hint, font=("Segoe UI", 8),
                  foreground=SUBTEXT, background=CARD).pack(anchor="w", pady=(4,8))

        btn_bar = ttk.Frame(frm, style="Card.TFrame")
        btn_bar.pack(fill="x", pady=(6,0))

        ttk.Button(btn_bar, text="Đăng nhập", command=self._on_login,
                   style="Accent.TButton").pack(fill="x", pady=(0,6))
        ttk.Button(btn_bar, text="Mở chấm công (Staff)",
                   command=self._open_staff_attendance).pack(fill="x")
        ttk.Button(btn_bar, text="Đăng ký Admin", command=self._open_register).pack(fill="x", pady=(6,0))

        style = ttk.Style(self)
        try: style.theme_use("clam")
        except Exception: pass
        style.configure("Card.TFrame", background=CARD)
        style.configure("Accent.TButton", background=PRIMARY, foreground="white", padding=8)
        style.map("Accent.TButton", background=[("active", PRIMARY_DARK)])

    def _on_login(self):
        user = self.username.get().strip()
        pwd = self.password.get().strip()
        role = self.role_var.get()

        if role != "Admin":
            messagebox.showwarning("Quyền", "Chỉ Admin được đăng nhập. Staff dùng nút 'Mở chấm công (Staff)'.")
            return

        try:
            acc = DB().get_admin_by_username(user)
        except Exception as e:
            messagebox.showerror("DB", f"Lỗi kết nối DB: {e}")
            return

        if not acc:
            messagebox.showerror("Sai", "Tài khoản không tồn tại")
            return

        ok = False
        try:
            ok = bcrypt.checkpw(pwd.encode(), acc["password_hash"].encode())
        except Exception:
            pass

        if ok:
            Dashboard(self, role="Admin", username=user)
            self.withdraw()
        else:
            messagebox.showerror("Sai", "Mật khẩu không đúng")

    def _open_staff_attendance(self):
        run_manual_attendance(0)

    def _open_register(self):
        RegisterDialog(self)

# -------------------- Register Admin Dialog (bắt buộc kiểm tra mặt) --------------------
class RegisterDialog(tk.Toplevel):
    def __init__(self, master):
        super().__init__(master)
        self.title("Đăng ký tài khoản Admin")
        self.geometry("520x520")
        self.configure(bg=BG)
        self.grab_set()
        self.temp_ok = False
        self.temp_path = None
        self._build()

    def _build(self):
        frm = ttk.Frame(self, padding=14, style="Card.TFrame")
        frm.pack(fill="both", expand=True)

        self.entries = {}
        fields = [
            "Tên thật",
            "Ngày sinh (dd/mm/yyyy) — có thể bỏ trống",
            "Phòng ban (có thể bỏ trống)",
            "Username",
            "Mật khẩu",
            "Nhập lại mật khẩu"
        ]
        for label in fields:
            ttk.Label(frm, text=label, background=CARD).pack(anchor="w", pady=(8,0))
            show = "*" if "Mật" in label else None
            e = ttk.Entry(frm, width=40, show=show)
            e.pack()
            self.entries[label] = e

        ttk.Button(frm, text="Chụp ảnh khuôn mặt (kiểm tra trùng) — BẮT BUỘC", style="Accent.TButton",
                   command=self._capture_temp).pack(pady=(10, 4))

        ttk.Button(frm, text="Tạo tài khoản Admin", style="Accent.TButton",
                   command=self._do_register).pack(pady=12)

    def _capture_temp(self):
        try:
            self.temp_path = FaceCollector().collect_one_temp()
            img = face_recognition.load_image_file(self.temp_path)
            boxes = face_recognition.face_locations(img, model="hog", number_of_times_to_upsample=0)
            if not boxes:
                self.temp_ok = False
                messagebox.showerror("Ảnh", "Không tìm thấy khuôn mặt trong ảnh tạm.")
                return
            enc = face_recognition.face_encodings(img, boxes, num_jitters=1)
            if not enc:
                self.temp_ok = False
                messagebox.showerror("Ảnh", "Không encode được khuôn mặt.")
                return

            known_encs, labels = load_all_encodings()
            if known_encs is None or len(known_encs) == 0:
                self.temp_ok = True
                messagebox.showinfo("OK", "Ảnh hợp lệ. Không thấy dữ liệu cũ để so trùng.")
                return

            enc_vec = enc[0]
            dists = np.linalg.norm(known_encs - enc_vec, axis=1)
            idx = int(np.argmin(dists))
            DUP_TOL = 0.43
            if dists[idx] <= DUP_TOL:
                self.temp_ok = False
                messagebox.showerror("Trùng mặt",
                    f"Khuôn mặt này trùng với: {labels[idx]}\nVui lòng dùng tài khoản đã có hoặc chụp người khác.")
            else:
                self.temp_ok = True
                messagebox.showinfo("OK", "Ảnh hợp lệ, không trùng với dữ liệu hiện có.")
        except Exception as e:
            self.temp_ok = False
            messagebox.showerror("Ảnh", str(e))

    def _do_register(self):
        ten_that = self.entries["Tên thật"].get().strip()
        bday_txt = self.entries["Ngày sinh (dd/mm/yyyy) — có thể bỏ trống"].get().strip()
        phongban = self.entries["Phòng ban (có thể bỏ trống)"].get().strip()
        username = self.entries["Username"].get().strip()
        pw1      = self.entries["Mật khẩu"].get().strip()
        pw2      = self.entries["Nhập lại mật khẩu"].get().strip()

        if not ten_that or not username or not pw1 or not pw2:
            messagebox.showwarning("Thiếu", "Nhập đầy đủ Tên thật, Username, Mật khẩu.")
            return
        if pw1 != pw2:
            messagebox.showerror("Lỗi", "Mật khẩu nhập lại không khớp")
            return
        if not self.temp_ok:
            messagebox.showerror("Bắt buộc", "Bạn phải chụp ảnh kiểm tra trùng mặt thành công trước khi tạo tài khoản.")
            return

        # Sinh mã NV tự tăng
        try:
            row = DB().q("SELECT MAX(ma_nv) AS m FROM nhanvien")
            max_code = (row[0]["m"] if row and row[0]["m"] else None)
            n = int(max_code[2:]) + 1 if (max_code and max_code.upper().startswith("NV")) else 1
            ma_nv = f"NV{n:03d}"
        except Exception as e:
            messagebox.showerror("DB", f"Lỗi sinh mã NV: {e}")
            return

        # Parse ngày sinh
        ngaysinh = None
        if bday_txt:
            try:
                ngaysinh = datetime.strptime(bday_txt, "%d/%m/%Y").date()
            except Exception:
                messagebox.showwarning("Ngày sinh", "Định dạng ngày sinh không hợp lệ (dd/mm/yyyy).")
                return

        # Thêm NV (vai trò admin)
        try:
            DB().add_employee(ma_nv, ten_that, ngaysinh, (phongban or None), "admin")
        except Exception as e:
            messagebox.showerror("DB", f"Không thêm được nhân viên: {e}")
            return

        # Tạo tài khoản admin
        hashed = bcrypt.hashpw(pw1.encode(), bcrypt.gensalt()).decode()
        try:
            DB().create_admin_account(ten_that, ma_nv, username, hashed)
        except Exception as e:
            messagebox.showerror("DB", f"Không tạo được tài khoản admin: {e}")
            return

        # Thu 30 ảnh cho mã NV
        try:
            label = f"{ten_that}_{ma_nv}"
            FaceCollector(max_images=30).collect(label)
        except Exception as e:
            messagebox.showwarning("Capture", f"Lỗi quét mặt: {e}")

        # Encode lại
        try:
            encode_script = Path(__file__).resolve().parent / "encode_sync.py"
            subprocess.run([sys.executable, str(encode_script)], check=True)
        except Exception as e:
            messagebox.showwarning("Encode", f"Lỗi encode_sync: {e}")

        messagebox.showinfo("OK", f"Đã tạo Admin {ten_that} ({ma_nv}).")
        self.destroy()

# -------------------- Dashboard --------------------
class Dashboard(tk.Toplevel):
    def __init__(self, master, role="Admin", username="admin"):
        super().__init__(master)
        self.title("FaceID Attendance — Dashboard")
        self.geometry("1120x720")
        self.configure(bg=BG)
        self.role = role
        self.username = username

        self.employee_list_cache = []
        self.tree_emp = None

        self._build_style()
        self._build_ui()
        self.show_home()
        self.protocol("WM_DELETE_WINDOW", self._on_close)

    def _build_style(self):
        style = ttk.Style(self)
        try: style.theme_use("clam")
        except Exception: pass
        style.configure("Header.TFrame", background=PRIMARY)
        style.configure("Header.TLabel", background=PRIMARY, foreground="white", font=("Segoe UI Semibold", 14))
        style.configure("Card.TFrame", background=CARD, relief="flat")
        style.configure("Title.TLabel", background=CARD, font=("Segoe UI Semibold", 14))
        style.configure("Sub.TLabel", background=CARD, foreground=SUBTEXT)
        style.configure("AccentSmall.TButton", background=PRIMARY, foreground="white", padding=6)
        style.map("AccentSmall.TButton", background=[("active", PRIMARY_DARK)])
        style.configure("Treeview", background="white", fieldbackground="white", rowheight=28, font=("Segoe UI", 10))
        style.configure("Treeview.Heading", font=("Segoe UI Semibold", 10))
        style.map("Treeview", background=[("selected", "#dbeafe")], foreground=[("selected", "black")])

    def _build_ui(self):
        header = ttk.Frame(self, style="Header.TFrame", padding=10)
        header.pack(side="top", fill="x")
        ttk.Label(header, text="Ứng dụng Chấm công FaceID", style="Header.TLabel").pack(side="left", padx=12)
        ttk.Label(header, text=f"{self.role}: {self.username}", background=PRIMARY, foreground="white"
                  ).pack(side="right", padx=12)

        menu_bar = ttk.Frame(self, padding=8, style="Card.TFrame")
        menu_bar.pack(side="top", fill="x", padx=12, pady=(12,6))

        items = [("Trang chủ", self.show_home),
                 ("Nhân viên", self.show_employees),
                 ("Chấm công", self.show_attendance),
                 ("Xuất Excel", self.show_export)]
        if self.role != "Admin":
            items = [("Chấm công", self.show_attendance)]
        for (label, cmd) in items:
            ttk.Button(menu_bar, text=label, command=cmd, style="AccentSmall.TButton").pack(side="left", padx=6)

        self.content = ttk.Frame(self, style="Card.TFrame", padding=12)
        self.content.pack(fill="both", expand=True, padx=12, pady=(0,12))

    def clear_content(self):
        for w in self.content.winfo_children():
            w.destroy()

    # ---------- Trang chủ ----------
    def show_home(self):
        self.clear_content()
        card = ttk.Frame(self.content, style="Card.TFrame", padding=16)
        card.pack(fill="both", expand=True)
        ttk.Label(card, text="🏠 Trang chủ", style="Title.TLabel").pack(anchor="w")
        ttk.Label(card, text="Thống kê nhanh hôm nay (theo DB)", style="Sub.TLabel").pack(anchor="w", pady=(6,12))

        stats_frame = ttk.Frame(card, style="Card.TFrame")
        stats_frame.pack(fill="x")
        total_emp, checked_today, not_checked = self._fetch_today_stats()

        for title, val in [
            ("Tổng số nhân viên", str(total_emp)),
            ("Đã chấm công hôm nay", str(checked_today)),
            ("Chưa chấm công hôm nay", str(not_checked)),
        ]:
            box = ttk.Frame(stats_frame, style="Card.TFrame", padding=14)
            box.pack(side="left", padx=8)
            ttk.Label(box, text=title, font=("Segoe UI", 10), background=CARD).pack(anchor="w")
            ttk.Label(box, text=val, font=("Segoe UI Semibold", 14), background=CARD).pack(anchor="w")

        ttk.Label(card, text="• Vào Nhân viên để thêm người và quét mặt.\n"
                             "• Vào Chấm công để mở camera (tự động IN/OUT, ESC để thoát).",
                  style="Sub.TLabel").pack(anchor="w", pady=(12,0))

    def _fetch_today_stats(self):
        try:
            db = DB()
            total_emp = db.q("SELECT COUNT(*) AS c FROM nhanvien")[0]["c"]
            checked_today = db.q(
                "SELECT COUNT(DISTINCT ma_nv) AS c FROM chamcong WHERE ngay = CURDATE() AND check_in IS NOT NULL"
            )[0]["c"]
            not_checked = max(0, total_emp - checked_today)
            return total_emp, checked_today, not_checked
        except Exception as e:
            messagebox.showerror("DB", f"Lỗi lấy thống kê: {e}")
            return 0, 0, 0

    # ---------- Nhân viên ----------
    def show_employees(self):
        self.clear_content()
        card = ttk.Frame(self.content, style="Card.TFrame", padding=12)
        card.pack(fill="both", expand=True)
        ttk.Label(card, text="👨‍💼 Quản lý nhân viên", style="Title.TLabel").pack(anchor="w")

        topbar = ttk.Frame(card, style="Card.TFrame")
        topbar.pack(fill="x", pady=(6,8))

        ttk.Label(topbar, text="Mã NV:", background=CARD).pack(side="left", padx=(0,6))
        self.search_code_var = StringVar()
        e_code = ttk.Entry(topbar, textvariable=self.search_code_var, width=18)
        e_code.pack(side="left")
        ttk.Button(topbar, text="Tìm", command=self._filter_by_code).pack(side="left", padx=(6,12))
        ttk.Button(topbar, text="Hiện tất cả", command=self._reload_employees).pack(side="left")

        ttk.Button(topbar, text="🗑 Xóa nhân viên", style="AccentSmall.TButton",
                   command=self._open_delete_dialog).pack(side="right", padx=6)
        ttk.Button(topbar, text="➕ Thêm nhân viên", style="AccentSmall.TButton",
                   command=self.add_employee).pack(side="right", padx=6)
        ttk.Button(topbar, text="↻ Tải lại", command=self._reload_employees).pack(side="right")

        cols = ("Mã NV", "Họ tên", "Ngày sinh", "Phòng ban", "Chức vụ")
        self.tree_emp = ttk.Treeview(card, columns=cols, show="headings")
        for c in cols:
            self.tree_emp.heading(c, text=c)
            self.tree_emp.column(c, width=180 if c == "Họ tên" else 120, anchor="w")
        self.tree_emp.pack(fill="both", expand=True, pady=(6,0))

        self._reload_employees()

    def _reload_employees(self):
        try:
            rows = DB().list_employees()
        except Exception as e:
            messagebox.showerror("DB", f"Lỗi tải nhân viên: {e}")
            return
        self.employee_list_cache = rows
        if self.tree_emp:
            self.tree_emp.delete(*self.tree_emp.get_children())
            for r in rows:
                ns = r["ngaysinh"].strftime("%d/%m/%Y") if r["ngaysinh"] else ""
                self.tree_emp.insert("", "end",
                                     values=(r["ma_nv"], r["ten"], ns, r["phongban"] or "", r["chucvu"]))

    def _filter_by_code(self):
        code = (self.search_code_var.get() or "").strip().upper()
        if not code:
            self._reload_employees()
            return
        filtered = [r for r in self.employee_list_cache if r["ma_nv"].upper() == code]
        if self.tree_emp:
            self.tree_emp.delete(*self.tree_emp.get_children())
            for r in filtered:
                ns = r["ngaysinh"].strftime("%d/%m/%Y") if r["ngaysinh"] else ""
                self.tree_emp.insert("", "end",
                                     values=(r["ma_nv"], r["ten"], ns, r["phongban"] or "", r["chucvu"]))

    def _open_delete_dialog(self):
        ma_nv = simpledialog.askstring("Xóa nhân viên", "Nhập Mã nhân viên (ví dụ: NV001):", parent=self)
        if not ma_nv:
            return
        ten = simpledialog.askstring("Xóa nhân viên", "Nhập HỌ TÊN nhân viên chính xác:", parent=self)
        if not ten:
            return
        ma_nv = ma_nv.strip().upper()
        ten_clean = " ".join(ten.strip().split())
        if not ma_nv or not ten_clean:
            return

        if not messagebox.askyesno("Xác nhận", f"Bạn chắc chắn muốn xóa {ten_clean} ({ma_nv})?\nHành động không thể hoàn tác."):
            return

        try:
            affected = DB().delete_employee(ma_nv, ten_clean)
            if affected == 0:
                messagebox.showwarning("Không tìm thấy", "Không có nhân viên khớp Mã NV và Họ tên.")
                return

            dataset_dir = (ROOT / "dataset")
            removed_any = False
            if dataset_dir.exists():
                for p in dataset_dir.iterdir():
                    if p.is_dir() and p.name.endswith(f"_{ma_nv}"):
                        try:
                            shutil.rmtree(p, ignore_errors=True)
                            removed_any = True
                        except Exception:
                            pass

            try:
                encode_script = Path(__file__).resolve().parent / "encode_sync.py"
                subprocess.run([sys.executable, str(encode_script)], check=True)
            except Exception as e:
                messagebox.showwarning("Encode", f"Đã xóa ảnh. Lỗi encode lại: {e}")

            msg = f"Đã xóa {ten_clean} ({ma_nv})."
            msg += "\nẢnh dataset đã xóa & encodings đã cập nhật." if removed_any else "\nKhông thấy thư mục ảnh tương ứng."
            messagebox.showinfo("Đã xóa", msg)
            self._reload_employees()

        except Exception as e:
            messagebox.showerror("DB", f"Lỗi xóa nhân viên: {e}")

    def add_employee(self):
        AddEmployeeDialog(self, on_done=self._reload_employees)

        # ---------- Chấm công ----------
    def show_attendance(self):
        self.clear_content()
        card = ttk.Frame(self.content, style="Card.TFrame", padding=12)
        card.pack(fill="both", expand=True)
        ttk.Label(card, text="🕒 Chấm công (tự động)", style="Title.TLabel").pack(anchor="w")

        # Topbar: tìm theo mã + mở camera + tải lại
        topbar = ttk.Frame(card, style="Card.TFrame")
        topbar.pack(fill="x", pady=(6,8))

        ttk.Button(topbar, text="⟳ Tải lại hôm nay", command=lambda: self._load_today_attendance()).pack(side="left", padx=(0,8))

        ttk.Label(topbar, text="Tìm theo mã NV (hôm nay):", background=CARD).pack(side="left", padx=(0,6))
        self.att_code_var = StringVar()
        e_code = ttk.Entry(topbar, textvariable=self.att_code_var, width=18)
        e_code.pack(side="left")
        ttk.Button(topbar, text="Tìm", command=lambda: self._load_today_attendance(self.att_code_var.get().strip().upper())
                   ).pack(side="left", padx=(6,12))
        ttk.Button(topbar, text="Xóa lọc", command=lambda: (self.att_code_var.set(""), self._load_today_attendance())
                   ).pack(side="left")

        ttk.Button(topbar, text="Mở camera (Auto IN/OUT)", style="AccentSmall.TButton",
                   command=lambda: run_manual_attendance(0, on_success=self._on_scan_success)
                   ).pack(side="right", padx=6)

        # Panel thông tin nhân viên vừa chấm
        info = ttk.LabelFrame(card, text="Thông tin nhân viên vừa chấm", padding=10)
        info.pack(fill="x", pady=(6, 6))
        self.att_lbl_name = ttk.Label(info, text="Họ tên: ", background=CARD)
        self.att_lbl_code = ttk.Label(info, text="Mã NV: ", background=CARD)
        self.att_lbl_dept = ttk.Label(info, text="Phòng ban: ", background=CARD)
        self.att_lbl_role = ttk.Label(info, text="Chức vụ: ", background=CARD)
        for w in (self.att_lbl_name, self.att_lbl_code, self.att_lbl_dept, self.att_lbl_role):
            w.pack(anchor="w")

        # Bảng danh sách chấm công hôm nay
        cols = ("Ngày", "Mã NV", "Họ tên", "Check-in", "Check-out", "Ghi chú")
        self.att_tree = ttk.Treeview(card, columns=cols, show="headings")
        for c in cols:
            self.att_tree.heading(c, text=c)
            self.att_tree.column(c, width=140 if c in ("Họ tên","Ghi chú") else 110, anchor="w")
        self.att_tree.pack(fill="both", expand=True, pady=(6,0))

        ttk.Label(card, text="• Mở camera, đứng trước ống kính. Hệ thống tự nhận diện: lần 1 = Check-in, lần 2 = Check-out.\n"
                             "• Khi thành công sẽ hiện 'Done' 4s trên cửa sổ camera và bảng dưới sẽ cập nhật ngay.",
                  style="Sub.TLabel").pack(anchor="w", pady=(8,0))

        # tải dữ liệu hôm nay ban đầu
        self._load_today_attendance()

    def _load_today_attendance(self, code_filter: str = ""):
        """Nạp bảng chấm công hôm nay (có thể lọc theo mã)."""
        try:
            if code_filter:
                rows = DB().q(
                    "SELECT ma_nv, ten_nv, ngay, check_in, check_out, note "
                    "FROM chamcong WHERE ngay = CURDATE() AND ma_nv = %s ORDER BY ma_nv", (code_filter,))
            else:
                rows = DB().q(
                    "SELECT ma_nv, ten_nv, ngay, check_in, check_out, note "
                    "FROM chamcong WHERE ngay = CURDATE() ORDER BY ma_nv")
        except Exception as e:
            messagebox.showerror("DB", f"Lỗi tải chấm công: {e}")
            return

        self.att_tree.delete(*self.att_tree.get_children())
        for r in rows:
            d = r["ngay"].strftime("%d/%m/%Y") if r["ngay"] else ""
            self.att_tree.insert("", "end", values=(d, r["ma_nv"], r["ten_nv"], r["check_in"] or "",
                                                    r["check_out"] or "", r.get("note","") or ""))

    def _on_scan_success(self, rec: dict):
        """
        Được gọi từ attendance_cam.run_manual_attendance khi chấm công thành công.
        rec: {ma_nv, ten_nv, ngay, check_in, check_out, total_seconds, note}
        """
        try:
            # cập nhật panel thông tin
            emp = DB().get_employee(rec["ma_nv"]) or {}
            self.att_lbl_name.config(text=f"Họ tên: {rec.get('ten_nv','')}")
            self.att_lbl_code.config(text=f"Mã NV: {rec.get('ma_nv','')}")
            self.att_lbl_dept.config(text=f"Phòng ban: {emp.get('phongban','') or ''}")
            self.att_lbl_role.config(text=f"Chức vụ: {emp.get('chucvu','') or ''}")
        except Exception:
            pass

        # refresh bảng hôm nay (giữ filter hiện tại nếu có)
        current_filter = (self.att_code_var.get() or "").strip().upper() if hasattr(self, "att_code_var") else ""
        self._load_today_attendance(current_filter)


    def _att_lookup(self):
        code = (self.att_code_var.get() or "").strip().upper()
        if not code:
            return
        rec = DB().get_employee(code)
        if not rec:
            messagebox.showinfo("Tìm kiếm", "Không tìm thấy nhân viên.")
            return
        self.att_lbl_name.config(text=f"Họ tên: {rec.get('ten','')}")
        self.att_lbl_code.config(text=f"Mã NV: {rec.get('ma_nv','')}")
        self.att_lbl_dept.config(text=f"Phòng ban: {rec.get('phongban','') or ''}")
        self.att_lbl_role.config(text=f"Chức vụ: {rec.get('chucvu','') or ''}")

    # ---------- Xuất Excel ----------
    def show_export(self):
        self.clear_content()
        card = ttk.Frame(self.content, style="Card.TFrame", padding=16)
        card.pack(fill="both", expand=True)
        ttk.Label(card, text="📊 Xuất Excel", style="Title.TLabel").pack(anchor="w")

        one_day = ttk.LabelFrame(card, text="Xuất 1 ngày (00:00 - 23:59)", padding=12)
        one_day.pack(anchor="w", fill="x", pady=(8,10))
        ttk.Label(one_day, text="Ngày (YYYY-MM-DD):", background=CARD).grid(row=0, column=0, sticky="w", padx=(0,8))
        e_day = ttk.Entry(one_day, width=20); e_day.grid(row=0, column=1, sticky="w")

        def _sec_to_hhmm(sec: int) -> str:
            m, _ = divmod(int(sec), 60); h, m = divmod(m, 60); return f"{h:02d}:{m:02d}"

        def do_export_one_day():
            d = e_day.get().strip()
            if not d:
                messagebox.showwarning("Thiếu", "Nhập ngày cần xuất (YYYY-MM-DD)"); return
            try:
                df = pd.DataFrame(DB().q(
                    "SELECT ma_nv, ten_nv, ngay, check_in, check_out, total_seconds, note "
                    "FROM chamcong WHERE ngay = %s ORDER BY ma_nv", (d,)
                ))
                if df.empty:
                    messagebox.showinfo("Trống", f"Không có bản ghi trong ngày {d}."); return
                if "ngay" in df.columns:
                    df["ngay"] = pd.to_datetime(df["ngay"]).dt.strftime("%d/%m/%Y")
                if "total_seconds" in df.columns:
                    df["total_hhmm"] = df["total_seconds"].fillna(0).astype(int).apply(_sec_to_hhmm)

                save_path = filedialog.asksaveasfilename(
                    defaultextension=".xlsx",
                    initialfile=f"chamcong_{d}.xlsx",
                    filetypes=[("Excel", "*.xlsx")]
                )
                if not save_path: return
                df.to_excel(save_path, index=False)
                messagebox.showinfo("OK", f"Đã xuất: {save_path}")
            except Exception as e:
                messagebox.showerror("Export", str(e))

        ttk.Button(one_day, text="Xuất 1 ngày", style="AccentSmall.TButton",
                   command=do_export_one_day).grid(row=0, column=2, padx=12)

        rng = ttk.LabelFrame(card, text="Xuất khoảng ngày (bao gồm 2 đầu)", padding=12)
        rng.pack(anchor="w", fill="x", pady=(4,0))

        ttk.Label(rng, text="Từ ngày (YYYY-MM-DD):", background=CARD).grid(row=0, column=0, sticky="w", padx=(0,8))
        e_from = ttk.Entry(rng, width=20); e_from.grid(row=0, column=1, sticky="w")
        ttk.Label(rng, text="Đến ngày (YYYY-MM-DD):", background=CARD).grid(row=1, column=0, sticky="w", padx=(0,8))
        e_to = ttk.Entry(rng, width=20); e_to.grid(row=1, column=1, sticky="w")

        def do_export_range():
            d1 = e_from.get().strip()
            d2 = e_to.get().strip()
            if not d1 or not d2:
                messagebox.showwarning("Thiếu", "Nhập đủ khoảng thời gian (YYYY-MM-DD)"); return
            try:
                df = pd.DataFrame(DB().q(
                    "SELECT ma_nv, ten_nv, ngay, check_in, check_out, total_seconds, note "
                    "FROM chamcong WHERE ngay BETWEEN %s AND %s ORDER BY ngay, ma_nv",
                    (d1, d2)
                ))
                if df.empty:
                    messagebox.showinfo("Trống", "Không có bản ghi trong khoảng ngày đã chọn.")
                    return
                if "ngay" in df.columns:
                    df["ngay"] = pd.to_datetime(df["ngay"]).dt.strftime("%d/%m/%Y")
                if "total_seconds" in df.columns:
                    df["total_hhmm"] = df["total_seconds"].fillna(0).astype(int).apply(_sec_to_hhmm)

                save_path = filedialog.asksaveasfilename(
                    defaultextension=".xlsx",
                    initialfile=f"chamcong_{d1}_to_{d2}.xlsx",
                    filetypes=[("Excel", "*.xlsx")]
                )
                if not save_path: return
                df.to_excel(save_path, index=False)
                messagebox.showinfo("OK", f"Đã xuất: {save_path}")
            except Exception as e:
                messagebox.showerror("Export", str(e))

        ttk.Button(rng, text="Xuất khoảng ngày", style="AccentSmall.TButton",
                   command=do_export_range).grid(row=0, column=2, rowspan=2, padx=12)

    def _on_close(self):
        self.master.deiconify()
        self.destroy()

# -------------------- Dialog: Add Employee (bắt buộc kiểm tra mặt) --------------------
class AddEmployeeDialog(tk.Toplevel):
    def __init__(self, master, on_done=None):
        super().__init__(master)
        self.title("Thêm nhân viên mới")
        self.geometry("520x560")
        self.configure(bg=BG)
        self.grab_set()
        self.on_done = on_done
        self.temp_ok = False
        self.temp_path = None
        self._build()

    def _build(self):
        frm = ttk.Frame(self, padding=14, style="Card.TFrame")
        frm.pack(fill="both", expand=True)

        self.ename = ttk.Entry(frm, width=30)
        self.ebirth = ttk.Entry(frm, width=30)
        self.edept = ttk.Entry(frm, width=30)
        self.erole = ttk.Combobox(frm, values=["nhanvien", "admin"], state="readonly", width=28)
        self.erole.set("nhanvien")

        for text, widget in [
            ("Họ tên", self.ename),
            ("Ngày sinh (dd/mm/yyyy) — có thể bỏ trống", self.ebirth),
            ("Phòng ban", self.edept),
            ("Chức vụ", self.erole),
        ]:
            ttk.Label(frm, text=text, background=CARD).pack(anchor="w", pady=(8,0))
            widget.pack(anchor="w")

        ttk.Button(frm, text="Chụp ảnh kiểm tra trùng mặt — BẮT BUỘC", style="Accent.TButton",
                   command=self._capture_temp).pack(pady=(10,4))

        ttk.Button(frm, text="Thêm & Quét mặt (30 ảnh)", style="Accent.TButton",
                   command=self._submit).pack(pady=12)

    def _capture_temp(self):
        try:
            self.temp_path = FaceCollector().collect_one_temp()
            img = face_recognition.load_image_file(self.temp_path)
            boxes = face_recognition.face_locations(img, model="hog", number_of_times_to_upsample=0)
            if not boxes:
                self.temp_ok = False
                messagebox.showerror("Ảnh", "Không tìm thấy khuôn mặt trong ảnh tạm.")
                return
            enc = face_recognition.face_encodings(img, boxes, num_jitters=1)
            if not enc:
                self.temp_ok = False
                messagebox.showerror("Ảnh", "Không encode được khuôn mặt.")
                return

            known_encs, labels = load_all_encodings()
            if known_encs is None or len(known_encs) == 0:
                self.temp_ok = True
                messagebox.showinfo("OK", "Ảnh hợp lệ. Không có dữ liệu cũ để so trùng.")
                return

            enc_vec = enc[0]
            dists = np.linalg.norm(known_encs - enc_vec, axis=1)
            idx = int(np.argmin(dists))
            DUP_TOL = 0.43
            if dists[idx] <= DUP_TOL:
                self.temp_ok = False
                messagebox.showerror("Trùng mặt",
                                     f"Khuôn mặt này trùng với: {labels[idx]}\nVui lòng kiểm tra lại.")
            else:
                self.temp_ok = True
                messagebox.showinfo("OK", "Ảnh hợp lệ, không trùng với dữ liệu hiện có.")
        except Exception as e:
            self.temp_ok = False
            messagebox.showerror("Ảnh", str(e))

    def _submit(self):
        if not self.temp_ok:
            messagebox.showerror("Bắt buộc", "Bạn phải chụp ảnh kiểm tra trùng mặt thành công trước khi thêm nhân viên.")
            return

        ten = self.ename.get().strip() or "Unknown"
        bday = self.ebirth.get().strip()
        dept = self.edept.get().strip() or "Khác"
        role = self.erole.get().strip().lower()

        # Sinh mã NV tự tăng
        try:
            row = DB().q("SELECT MAX(ma_nv) AS m FROM nhanvien")
            max_code = (row[0]["m"] if row and row[0]["m"] else None)
            n = int(max_code[2:]) + 1 if (max_code and max_code.upper().startswith("NV")) else 1
            ma = f"NV{n:03d}"
        except Exception as e:
            messagebox.showerror("DB", f"Lỗi sinh mã NV: {e}")
            return

        ngaysinh = None
        if bday:
            try:
                ngaysinh = datetime.strptime(bday, "%d/%m/%Y").date()
            except Exception:
                messagebox.showwarning("Ngày sinh", "Định dạng ngày sinh không hợp lệ (dd/mm/yyyy).")
                return

        # Lưu DB
        try:
            DB().add_employee(ma, ten, ngaysinh, dept, role)
        except Exception as e:
            messagebox.showerror("DB", f"Không thêm được nhân viên: {e}")
            return

        # Thu 30 ảnh
        try:
            label = f"{ten}_{ma}"
            FaceCollector(max_images=30).collect(label)
        except Exception as e:
            messagebox.showwarning("Capture", f"Lỗi quét mặt: {e}")

        # Encode sync
        try:
            encode_script = Path(__file__).resolve().parent / "encode_sync.py"
            subprocess.run([sys.executable, str(encode_script)], check=True)
        except Exception as e:
            messagebox.showwarning("Encode", f"Lỗi encode_sync: {e}")

        messagebox.showinfo("OK", f"Đã thêm {ten} ({ma}).")
        if callable(self.on_done):
            self.on_done()
        self.destroy()

# -------------------- Run --------------------
if __name__ == "__main__":
    app = LoginWindow()
    app.mainloop()
