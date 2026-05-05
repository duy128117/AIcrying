# Baby Sleep Tracker

Baby Sleep Tracker là dự án kết hợp giữa **phần AI nhận diện tiếng khóc** và **ứng dụng Flutter giám sát bé**. Hệ thống có khả năng **thu âm trực tiếp** từ môi trường và phát hiện tiếng khóc real-time, không chỉ xử lý những file âm thanh được gửi đến.

Mục tiêu của dự án là hỗ trợ phụ huynh theo dõi tình trạng của bé 24/7, bao gồm:

- **phát hiện tiếng khóc ngay lập tức** từ audio trực tiếp
- theo dõi nhiệt độ, độ ẩm và tư thế ngủ
- gửi cảnh báo khi có dấu hiệu bất thường
- lưu lại lịch sử dữ liệu để có thể xem lại muộn

---

## Tổng quan hệ thống

Hệ thống được xây theo luồng khá đơn giản:

1. **Dữ liệu âm thanh và cảm biến** được thu thập từ môi trường theo dõi bé
2. **Phần AI** xử lý âm thanh, chuyển sang spectrogram và phân loại tiếng khóc
3. **Phần Flutter** hiển thị dữ liệu realtime, biểu đồ và cảnh báo
4. **Firebase** dùng để lưu trữ, đồng bộ dữ liệu và hỗ trợ xác thực

---

## Về phía AI

- **thu âm trực tiếp** từ thiết bị hoặc sử dụng file `.wav` đã có sẵn
- xử lý audio real-time để phát hiện tiếng khóc ngay khi có
- hoạt động liên tục trong background

### Quá trình xử lý

- audio được chia thành các đoạn 2 giây
- mỗi đoạn được chuyển sang Mel-spectrogram
- ảnh spectrogram được resize về `128 x 128 x 3`

### Đầu ra

- `cry`: có tiếng khóc
- `not_crying`: không có tiếng khóc

### Cách xử lý chính

1. Chuẩn bị dữ liệu âm thanh
2. Chia đoạn audio
3. Tạo Mel-spectrogram
4. Huấn luyện mô hình
5. Đánh giá mô hình
6. Chuyển sang TFLite
7. Benchmark tốc độ, dung lượng và độ chính xác

### Điểm nổi bật

- dùng `MobileNetV2` để tận dụng transfer learning
- có thêm nhiễu nền để mô hình thực tế hơn
- xuất được sang `TensorFlow Lite`
- phù hợp cho triển khai mobile hoặc thiết bị edge

---

## Về phía Flutter

Phần Flutter là giao diện theo dõi dành cho phụ huynh hoặc người chăm sóc. Ứng dụng hiển thị trạng thái bé, nhận cảnh báo và cho phép cấu hình một số thông số cơ bản.

### Chức năng chính

- đăng nhập bằng số điện thoại và OTP
- quét QR để liên kết thiết bị
- xem dashboard thời gian thực
- theo dõi các chỉ số liên quan đến bé
- cấu hình ngưỡng cảnh báo
- xem biểu đồ lịch sử
- chạy nền để tiếp tục giám sát khi tắt màn hình

### Cảnh báo có thể hiển thị

- bé đang khóc
- tư thế ngủ bất thường
- nhiệt độ cơ thể vượt ngưỡng
- nhiệt độ phòng vượt ngưỡng
- độ ẩm phòng vượt ngưỡng

---

## Tech stack

### AI & Data Science

**Core**
- Python
- TensorFlow / Keras

**Audio Processing**
- librosa
- sounddevice
- soundfile

**Math & Tools**
- NumPy
- Scikit-learn
- Matplotlib
- Pillow

### Mobile & Backend

**Framework**
- Flutter / Dart

**Cloud**
- Firebase Authentication
- Firebase Realtime Database
- Firebase Cloud Messaging

**Hardware Integration & UI**
- flutter_background_service
- flutter_local_notifications
- qr_flutter
- mobile_scanner (QR scanner và liên kết thiết bị)
- fl_chart
- shared_preferences
- connectivity_plus

---

## Cấu trúc dự án

```text
PBL4/
├── ai/        # phần AI phân loại tiếng khóc
├── fe/        # phần Flutter frontend
├── doc/       # tài liệu nghiên cứu và báo cáo
└── README.md  # mô tả tổng quan dự án
```

---

## Cài đặt nhanh

### Phần AI

```bash
pip install -r ai/requirements.txt
```

### Phần Flutter

```bash
cd fe
flutter pub get
```
