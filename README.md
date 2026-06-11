# 🛡️ Ứng dụng Web phát hiện giao dịch gian lận tài chính

Ứng dụng này được chuyển đổi tự động từ mã nguồn xử lý và huấn luyện mô hình học máy trong Notebook thành một giao dịch Web UI tương tác trực quan sử dụng thư viện **Streamlit**. 

Hệ thống cho phép người dùng quản trị, thay đổi linh hoạt các siêu tham số của mô hình học máy **Random Forest** trực tiếp trên màn hình, đánh giá các chỉ số kiểm định chất lượng (`Accuracy`, `Precision`, `Recall`, `F1-Score`), đồng thời thực hiện dự đoán phân loại nhanh rủi ro cho các giao dịch mới theo cơ chế thời gian thực hoặc xử lý danh sách hàng loạt.

---

## 🛠️ Hướng dẫn cài đặt và vận hành ứng dụng

### Bước 1: Chuẩn bị môi trường máy tính
Đảm bảo bạn máy tính của bạn đã cài đặt Python (Khuyến nghị phiên bản `>= 3.9` đến `3.12`).

### Bước 2: Cài đặt các gói thư viện phụ thuộc
Mở terminal hoặc cửa sổ dòng lệnh tại thư mục chứa mã nguồn ứng dụng và chạy câu lệnh sau:
```bash
pip install -r requirements.txt
