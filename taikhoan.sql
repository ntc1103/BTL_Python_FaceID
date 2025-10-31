CREATE TABLE taikhoan (
  id INT AUTO_INCREMENT PRIMARY KEY,
  ma_nv VARCHAR(20),
  username VARCHAR(50) UNIQUE NOT NULL,
  password VARCHAR(100) NOT NULL,
  role ENUM('admin', 'user') DEFAULT 'user',
  FOREIGN KEY (ma_nv) REFERENCES nhanvien(ma_nv)
);
