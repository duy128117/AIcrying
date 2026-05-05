# Baby Sleep Tracker

Baby Sleep Tracker là dự án kết hợp giữa **phần AI nhận diện tiếng khóc** và **ứng dụng Flutter giám sát bé**. Trong toàn bộ hệ thống, phần AI được phát triển kỹ hơn và xuất hiện nhiều hơn ở luồng xử lý, còn Flutter đóng vai trò giao diện để hiển thị dữ liệu, cảnh báo và hỗ trợ người dùng thao tác.

Mục tiêu của dự án là hỗ trợ phụ huynh theo dõi tình trạng của bé theo thời gian thực, bao gồm:

- phát hiện tiếng khóc từ âm thanh
- theo dõi nhiệt độ, độ ẩm và tư thế ngủ
- hiển thị cảnh báo khi có dấu hiệu bất thường
- xem lại lịch sử dữ liệu để tiện đánh giá

---

## Tổng quan hệ thống

Hệ thống được xây theo luồng khá đơn giản:

1. **Dữ liệu âm thanh và cảm biến** được thu thập từ môi trường theo dõi bé
2. **Phần AI** xử lý âm thanh, chuyển sang spectrogram và phân loại tiếng khóc
3. **Phần Flutter** hiển thị dữ liệu realtime, biểu đồ và cảnh báo
4. **Firebase** dùng để lưu trữ, đồng bộ dữ liệu và hỗ trợ xác thực

---

## Phần AI

Phần AI là phần được đầu tư nhiều hơn trong dự án vì nó xử lý trực tiếp bài toán nhận diện tiếng khóc.

### Đầu vào

- file âm thanh `.wav`
- âm thanh được chia thành các đoạn 2 giây
- mỗi đoạn được chuyển sang Mel-spectrogram
- ảnh đầu ra được resize về `128 x 128 x 3`

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

## Phần Flutter

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

### AI

- Python
- TensorFlow / Keras
- librosa
- scikit-learn
- NumPy
- Pillow
- matplotlib
- soundfile, sounddevice

### Flutter

- Flutter / Dart
- Firebase Authentication
- Firebase Realtime Database
- Firebase Cloud Messaging
- flutter_background_service
- flutter_local_notifications
- qr_flutter
- mobile_scanner
- fl_chart
- shared_preferences

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

## Ghi chú ngắn cho người xem

- Phần AI là phần có khối lượng xử lý nhiều hơn và là phần nổi bật của dự án.
- Phần Flutter chủ yếu phục vụ hiển thị, cảnh báo và tương tác người dùng.
- Hai phần được tách riêng để dễ phát triển và bảo trì.

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
