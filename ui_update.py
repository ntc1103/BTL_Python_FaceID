import tkinter as tk
from tkinter import ttk, messagebox, simpledialog, StringVar
from datetime import datetime
import random
import cv2
import os

# --------------------
# Config (soft blue)
# --------------------
PRIMARY = "#3b82f6"
PRIMARY_DARK = "#2563eb"
BG = "#f6f9ff"
CARD = "#ffffff"
SUBTEXT = "#50616a"

# ---------- App ----------
class LoginWindow(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Đăng nhập — Ứng dụng Chấm công nhận diện khuôn mặt")
        self.geometry("420x340")
        self.configure(bg=BG)
        self.resizable(False, False)
        self._build_ui()

    def _build_ui(self):
        frm = ttk.Frame(self, padding=18, style="Card.TFrame")
        frm.place(relx=0.5, rely=0.5, anchor="center")

        ttk.Label(frm, text="Ứng dụng Chấm công nhận diện khuôn mặt",
                  font=("Segoe UI Semibold", 12)).pack(pady=(0,10))

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

        hint = ("Tài khoản mẫu:\nAdmin: admin / 123\nStaff: staff / 123")
        ttk.Label(frm, text=hint, font=("Segoe UI", 8),
                  foreground=SUBTEXT, background=CARD).pack(anchor="w", pady=(4,8))

        btn_frame = ttk.Frame(frm)
        btn_frame.pack(fill="x", pady=(6,0))
        login_btn = ttk.Button(btn_frame, text="Đăng nhập", command=self._on_login, style="Accent.TButton")
        login_btn.pack(side="left", expand=True, fill="x", padx=(0,6))
        quit_btn = ttk.Button(btn_frame, text="Thoát", command=self.quit)
        quit_btn.pack(side="left", fill="x", padx=(6,0))

        style = ttk.Style(self)
        try:
            style.theme_use("clam")
        except Exception:
            pass
        style.configure("Card.TFrame", background=CARD)
        style.configure("Accent.TButton", background=PRIMARY, foreground="white", padding=8)
        style.map("Accent.TButton", background=[("active", PRIMARY_DARK)])

    def _on_login(self):
        user = self.username.get().strip()
        pwd = self.password.get().strip()
        role = self.role_var.get()

        if role == "Admin" and user == "admin" and pwd == "123":
            Dashboard(self, role="Admin", username=user)
            self.withdraw()
        elif role == "Staff" and user == "staff" and pwd == "123":
            Dashboard(self, role="Staff", username=user)
            self.withdraw()
        else:
            messagebox.showerror("Lỗi đăng nhập", "Tên đăng nhập, mật khẩu hoặc vai trò không đúng (demo).")


class Dashboard(tk.Toplevel):
    def __init__(self, master, role="Staff", username="user"):
        super().__init__(master)
        self.title("Ứng dụng Chấm công nhận diện khuôn mặt")
        self.geometry("1100x700")
        self.configure(bg=BG)
        self.role = role
        self.username = username
        self.employee_list = []
        self.tree_emp = None
        self._build_style()
        self._build_ui()
        self._load_demo_data()
        self.protocol("WM_DELETE_WINDOW", self._on_close)

    def _build_style(self):
        style = ttk.Style(self)
        try:
            style.theme_use("clam")
        except Exception:
            pass
        style.configure("Header.TFrame", background=PRIMARY)
        style.configure("Header.TLabel", background=PRIMARY, foreground="white", font=("Segoe UI Semibold", 14))
        style.configure("TopMenu.TButton", background=PRIMARY, foreground="white", padding=8)
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
        ttk.Label(header, text="Ứng dụng Chấm công nhận diện khuôn mặt", style="Header.TLabel").pack(side="left", padx=12)
        ttk.Label(header, text=f"{self.role}: {self.username}", background=PRIMARY, foreground="white").pack(side="right", padx=12)

        menu_bar = ttk.Frame(self, padding=8, style="Card.TFrame")
        menu_bar.pack(side="top", fill="x", padx=12, pady=(12,6))

        self.menu_buttons = {}
        menu_items_admin = [("Trang chủ", self.show_home),
                            ("Nhân viên", self.show_employees),
                            ("Chấm công", self.show_attendance),
                            ("Báo cáo", self.show_reports),
                            ("Cấu hình", self.show_settings)]
        menu_items_staff = [("Chấm công", self.show_attendance),
                            ("Xin nghỉ phép", self.show_leave_request)]
        items = menu_items_admin if self.role == "Admin" else menu_items_staff

        for (label, cmd) in items:
            b = ttk.Button(menu_bar, text=label, command=cmd, style="AccentSmall.TButton")
            b.pack(side="left", padx=6)
            self.menu_buttons[label] = b

        right_frame = ttk.Frame(menu_bar, style="Card.TFrame")
        right_frame.pack(side="right")
        self.search_var = StringVar()
        self.search_entry = ttk.Entry(right_frame, textvariable=self.search_var, width=30)
        self.search_entry.pack(side="left", padx=(0,6))
        self.search_entry.insert(0, "Tìm nhân viên…")
        self.search_entry.bind("<FocusIn>", lambda e: self._clear_placeholder())
        self.search_entry.bind("<KeyRelease>", lambda e: self.on_search())
        clear_btn = ttk.Button(right_frame, text="✖", command=self.clear_search)
        clear_btn.pack(side="left")

        self.content = ttk.Frame(self, style="Card.TFrame", padding=12)
        self.content.pack(fill="both", expand=True, padx=12, pady=(0,12))
        self.show_home()

    def _clear_placeholder(self):
        if self.search_var.get().strip().lower() == "tìm nhân viên…":
            self.search_var.set("")

    def _load_demo_data(self):
        # demo employee list used by Staff/Admin
        self.employee_list = [
            {"id": "NV001", "name": "Nguyen Van A", "dept": "Kỹ thuật", "role": "Nhân viên"},
            {"id": "NV002", "name": "Tran Thi B", "dept": "Hành chính", "role": "Quản lý"},
            {"id": "NV003", "name": "Le Van C", "dept": "Kế toán", "role": "Nhân viên"},
            {"id": "NV004", "name": "Pham Thi D", "dept": "Kỹ thuật", "role": "Leader"},
            {"id": "NV005", "name": "Nguyen Thi E", "dept": "Marketing", "role": "Thực tập"},
            {"id": "NV006", "name": "Pham Van F", "dept": "Kinh doanh", "role": "Nhân viên"},
        ]

    def clear_content(self):
        for w in self.content.winfo_children():
            w.destroy()

    # ------- Pages -------
    def show_home(self):
        self.clear_content()
        card = ttk.Frame(self.content, style="Card.TFrame", padding=16)
        card.pack(fill="both", expand=True)
        ttk.Label(card, text="🏠 Trang chủ", style="Title.TLabel").pack(anchor="w")
        ttk.Label(card, text=f"Chào mừng {self.username} ({self.role})!", style="Sub.TLabel").pack(anchor="w", pady=(6,12))
        stats_frame = ttk.Frame(card, style="Card.TFrame")
        stats_frame.pack(fill="x", pady=6)
        for title, val in [("Nhân viên", str(len(self.employee_list))), ("Chấm công hôm nay", "9"), ("Vắng", "3")]:
            box = ttk.Frame(stats_frame, style="Card.TFrame", padding=12)
            box.pack(side="left", padx=8)
            ttk.Label(box, text=title, font=("Segoe UI", 10)).pack(anchor="w")
            ttk.Label(box, text=val, font=("Segoe UI Semibold", 12)).pack(anchor="w")

    def show_employees(self):
        self.clear_content()
        card = ttk.Frame(self.content, style="Card.TFrame", padding=12)
        card.pack(fill="both", expand=True)
        ttk.Label(card, text="👨‍💼 Quản lý nhân viên", style="Title.TLabel").pack(anchor="w")
        # add button only for Admin
        topbar = ttk.Frame(card, style="Card.TFrame")
        topbar.pack(fill="x", pady=(6,8))
        if self.role == "Admin":
            ttk.Button(topbar, text="➕ Thêm nhân viên", command=self.add_employee).pack(side="right", padx=6)


        cols = ("Mã NV", "Họ tên", "Phòng ban", "Chức vụ")
        self.tree_emp = ttk.Treeview(card, columns=cols, show="headings")
        for c in cols:
            self.tree_emp.heading(c, text=c)
            self.tree_emp.column(c, width=200 if c == "Họ tên" else 140, anchor="w")
        self.tree_emp.pack(fill="both", expand=True, pady=(6,0))

        # load demo employees
        self.tree_emp.delete(*self.tree_emp.get_children())
        for emp in self.employee_list:
            self.tree_emp.insert("", "end", values=(emp["id"], emp["name"], emp["dept"], emp["role"]))

    def show_attendance(self):
        self.clear_content()
        card = ttk.Frame(self.content, style="Card.TFrame", padding=12)
        card.pack(fill="both", expand=True)
        ttk.Label(card, text="🕒 Chấm công", style="Title.TLabel").pack(anchor="w")
        topbar = ttk.Frame(card, style="Card.TFrame")
        topbar.pack(fill="x", pady=(6,8))
        # Only FaceID check-in (button kept)
        ttk.Button(topbar, text="Check-in FaceID", command=self.fake_checkin).pack(side="right", padx=6)

        cols = ("Thời gian vào", "Mã NV", "Họ tên", "Trạng thái", "Ghi chú")
        tv = ttk.Treeview(card, columns=cols, show="headings")
        for c in cols:
            tv.heading(c, text=c)
            tv.column(c, width=170 if c == "Họ tên" else 140, anchor="w")
        tv.pack(fill="both", expand=True, pady=(6,0))

        rows = [
            (datetime.now().strftime("%Y-%m-%d 08:00"), "NV001", "Nguyen Van A", "valid", "OK"),
            (datetime.now().strftime("%Y-%m-%d 08:15"), "NV002", "Tran Thi B", "late", "Vào muộn 15p"),
        ]
        for r in rows:
            tv.insert("", "end", values=r)

    def show_reports(self):
        self.clear_content()
        card = ttk.Frame(self.content, style="Card.TFrame", padding=16)
        card.pack(fill="both", expand=True)
        ttk.Label(card, text="📊 Báo cáo (demo)", style="Title.TLabel").pack(anchor="w")
        ttk.Label(card, text="Tính năng xuất báo cáo sẽ do backend thực hiện; hiện đây là khung demo.", style="Sub.TLabel").pack(anchor="w", pady=8)
        ttk.Button(card, text="Xuất Excel (demo)", command=self.export_excel).pack(pady=8)

    def show_settings(self):
        self.clear_content()
        card = ttk.Frame(self.content, style="Card.TFrame", padding=16)
        card.pack(fill="both", expand=True)
        ttk.Label(card, text="⚙️ Cấu hình hệ thống (demo)", style="Title.TLabel").pack(anchor="w")
        ttk.Label(card, text="Các thiết lập (tạm) sẽ do backend quản lý.", style="Sub.TLabel").pack(anchor="w", pady=8)

    def show_leave_request(self):
        self.clear_content()
        card = ttk.Frame(self.content, style="Card.TFrame", padding=16)
        card.pack(fill="both", expand=True)
        ttk.Label(card, text="📝 Xin nghỉ phép", style="Title.TLabel").pack(anchor="w")
        ttk.Label(card, text="Lý do:", style="Sub.TLabel").pack(anchor="w", pady=(6,0))
        reason = ttk.Entry(card, width=60); reason.pack(anchor="w", pady=4)
        ttk.Label(card, text="Từ ngày (YYYY-MM-DD):", style="Sub.TLabel").pack(anchor="w", pady=(6,0))
        start = ttk.Entry(card, width=20); start.pack(anchor="w", pady=4)
        ttk.Label(card, text="Đến ngày (YYYY-MM-DD):", style="Sub.TLabel").pack(anchor="w", pady=(6,0))
        end = ttk.Entry(card, width=20); end.pack(anchor="w", pady=4)
        def send_request():
            if not reason.get().strip() or not start.get().strip() or not end.get().strip():
                messagebox.showwarning("Thiếu thông tin", "Vui lòng nhập đầy đủ thông tin!")
                return
            messagebox.showinfo("Đã gửi", "Yêu cầu nghỉ phép đã được gửi (demo).")
        ttk.Button(card, text="Gửi yêu cầu", command=send_request).pack(pady=10)

    # ------- small features -------
    # -------------------------------------
    # ✅ THÊM NHÂN VIÊN + QUÉT KHUÔN MẶT
    # -------------------------------------
    def add_employee(self):
        dlg = tk.Toplevel(self)
        dlg.title("Thêm nhân viên mới")
        dlg.geometry("420x350")
        dlg.grab_set()

        frm = ttk.Frame(dlg, padding=12, style="Card.TFrame")
        frm.pack(fill="both", expand=True)

        ttk.Label(frm, text="Thêm nhân viên mới", style="Title.TLabel").pack(pady=6)

        ttk.Label(frm, text="Mã nhân viên").pack(anchor="w")
        eid = ttk.Entry(frm); eid.pack(fill="x")

        ttk.Label(frm, text="Họ tên").pack(anchor="w")
        name = ttk.Entry(frm); name.pack(fill="x")

        ttk.Label(frm, text="Ngày sinh (dd/mm/yyyy)").pack(anchor="w")
        birthday = ttk.Entry(frm); birthday.pack(fill="x")

        ttk.Label(frm, text="Phòng ban").pack(anchor="w")
        dept = ttk.Entry(frm); dept.pack(fill="x")

        ttk.Label(frm, text="Chức vụ").pack(anchor="w")
        role = ttk.Entry(frm); role.pack(fill="x")

        def submit():
            new_emp = {
                "id": eid.get().strip() or f"NV{random.randint(100,999)}",
                "name": name.get().strip() or "Unknown",
                "birthday": birthday.get().strip() or "Không rõ",
                "dept": dept.get().strip() or "Khác",
                "role": role.get().strip() or "Nhân viên"
            }

            # ✅ Lưu vào danh sách UI
            self.employee_list.append(new_emp)

            # ✅ Hiển thị ngay lên bảng
            if self.tree_emp:
                self.tree_emp.insert("", "end",
                    values=(new_emp["id"], new_emp["name"], new_emp["dept"], new_emp["role"])
                )

            dlg.destroy()

            # ✅ Gọi chức năng quét mặt ngay sau khi thêm
            self.capture_faces_and_add_employee(new_emp["id"], new_emp["name"])

        ttk.Button(frm, text="Thêm & Quét mặt", command=submit).pack(pady=10)


    def capture_faces_and_add_employee(self, emp_id=None, emp_name=None):
        print(">>> [UI] Đã gọi hàm quét mặt — chờ backend xử lý...")
        print(f">>> Employee ID: {emp_id}, Name: {emp_name}")

        # 📌 Tổng kết:
        #  - UI đã gọi đúng hàm
        #  - Backend sau này sẽ đưa code capture vào đây
        messagebox.showinfo(
            "Quét mặt",
            f"📷 Đang quét khuôn mặt cho nhân viên:\n\n"
            f"➡️ {emp_name} ({emp_id})\n\n"
            f"(UI đã gọi hàm — Backend xử lý tiếp)"
        )



    def fake_checkin(self):
        # simple checkin demo (no anti-spoofing here)
        messagebox.showinfo("Check-in", "Nhận diện khuôn mặt thành công (demo). Dữ liệu đã gửi lên server (demo).")

    def export_excel(self):
        try:
            from openpyxl import Workbook
        except Exception:
            messagebox.showwarning("Thiếu thư viện", "Cài openpyxl để xuất Excel (pip install openpyxl).")
            return
        wb = Workbook(); ws = wb.active
        ws.append(["Thời gian", "Mã NV", "Họ tên", "Trạng thái", "Ghi chú"])
        ws.append([datetime.now().strftime("%Y-%m-%d %H:%M"), "NV001", "Nguyen Van A", "valid", "OK"])
        fname = "bao_cao_demo.xlsx"; wb.save(fname)
        messagebox.showinfo("Xuất Excel", f"Đã xuất: {fname}")

    def on_search(self):
        kw = self.search_var.get().strip().lower()
        if not self.tree_emp:
            return
        # if empty show all
        if not kw or kw == "tìm nhân viên…":
            # reset to the demo list
            self.tree_emp.delete(*self.tree_emp.get_children())
            for emp in self.employee_list:
                self.tree_emp.insert("", "end", values=(emp["id"], emp["name"], emp["dept"], emp["role"]))
            return
        # filter
        filtered = [e for e in self.employee_list if kw in e["id"].lower() or kw in e["name"].lower()]
        self.tree_emp.delete(*self.tree_emp.get_children())
        for emp in filtered:
            self.tree_emp.insert("", "end", values=(emp["id"], emp["name"], emp["dept"], emp["role"]))

    def clear_search(self):
        self.search_var.set("")
        self.on_search()

    def _on_close(self):
        # when dashboard closed, show login window back
        self.master.deiconify()
        self.destroy()

# ---------- run ----------
if __name__ == "__main__":
    app = LoginWindow()
    app.mainloop()
