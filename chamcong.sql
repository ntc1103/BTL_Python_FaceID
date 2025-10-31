CREATE TABLE chamcong (
  id INT AUTO_INCREMENT PRIMARY KEY,
  ma_nv VARCHAR(20),
  ngay DATE,
  checkin_time DATETIME,
  checkout_time DATETIME,
  ghi_chu VARCHAR(255),
  FOREIGN KEY (ma_nv) REFERENCES nhanvien(ma_nv)
);

