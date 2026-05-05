# Baby Sleep Tracker – AI + Flutter Frontend

Đây là workspace tổng hợp gồm 2 phần chính:

- [ai/README.md](ai/README.md) – mô hình AI nhận diện tiếng khóc
- [fe/README.md](fe/README.md) – ứng dụng Flutter cho phụ huynh theo dõi bé

Mục tiêu của hệ thống là giám sát giấc ngủ và môi trường của bé theo thời gian thực, từ đó phát hiện các tình huống bất thường như bé khóc, tư thế ngủ không phù hợp hoặc nhiệt độ/độ ẩm vượt ngưỡng.

---

## Kiến trúc tổng quan

1. Thiết bị hoặc nguồn dữ liệu tạo ra trạng thái giấc ngủ và môi trường
2. Phần AI xử lý âm thanh để phát hiện tiếng khóc
3. Frontend Flutter hiển thị dashboard, biểu đồ, cảnh báo và cài đặt
4. Firebase đóng vai trò backend đồng bộ dữ liệu, xác thực và thông báo

---

## Thành phần

### AI

- Phân loại âm thanh thành `cry` và `not_crying`
- Hỗ trợ train, evaluate, benchmark và xuất `TFLite`
- Phù hợp cho triển khai mobile hoặc edge

### Frontend

- Ứng dụng Flutter đa nền tảng
- Đăng nhập bằng số điện thoại OTP
- Liên kết thiết bị bằng QR code
- Dashboard theo dõi dữ liệu thời gian thực
- Cảnh báo cục bộ và push notification
- Chạy nền để giám sát liên tục

---

## Hướng dẫn nhanh

### AI

Xem chi tiết tại [ai/README.md](ai/README.md).

### Frontend

Xem chi tiết tại [fe/README.md](fe/README.md).
