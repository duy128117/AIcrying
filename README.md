# Baby Sleep Tracker – AI trọng tâm, Flutter hỗ trợ giám sát bé

Đây là dự án mà **phần AI là cốt lõi quan trọng nhất**. Ứng dụng Flutter chỉ đóng vai trò giao diện giám sát, hiển thị dữ liệu và cảnh báo cho phụ huynh. Nói ngắn gọn: **AI phát hiện tiếng khóc, Flutter giúp người dùng theo dõi và nhận cảnh báo**.

Mục tiêu của hệ thống là hỗ trợ phụ huynh giám sát bé một cách chủ động bằng cách:

- phát hiện tiếng khóc từ âm thanh
- theo dõi nhiệt độ, độ ẩm, tư thế ngủ và trạng thái cảnh báo
- gửi thông báo ngay khi có dấu hiệu bất thường
- cho phép xem lại lịch sử dữ liệu để đánh giá tình trạng của bé

---

## Tổng quan hệ thống

Hệ thống gồm 4 phần chính:

1. **Nguồn dữ liệu âm thanh và cảm biến**
	- âm thanh từ môi trường xung quanh bé
	- dữ liệu trạng thái như nhiệt độ, độ ẩm, tư thế ngủ và trạng thái khóc

2. **Phần AI**
	- nhận dữ liệu audio
	- chuyển audio thành Mel-spectrogram
	- phân loại thành 2 nhóm: tiếng khóc và không phải tiếng khóc

3. **Phần frontend Flutter**
	- hiển thị dashboard theo thời gian thực
	- cấu hình cảnh báo
	- hiển thị biểu đồ lịch sử
	- nhận cảnh báo cục bộ và push notification

4. **Firebase**
	- lưu và đồng bộ dữ liệu
	- xác thực bằng OTP
	- hỗ trợ kết nối thiết bị bằng QR code
	- phục vụ realtime database và messaging

---

## Phần AI làm gì?  
**Đây là phần quan trọng nhất của dự án.**

Phần AI được xây dựng để xử lý bài toán phân loại âm thanh.

### Đầu vào

- file âm thanh `.wav`
- audio được chia thành từng đoạn 2 giây
- mỗi đoạn được chuyển thành Mel-spectrogram
- ảnh spectrogram được resize về `128 x 128 x 3`

### Đầu ra

- `cry`: có tiếng khóc
- `not_crying`: không có tiếng khóc

### Điểm mạnh của phần AI

- dùng `MobileNetV2` để tận dụng transfer learning
- hỗ trợ dữ liệu nhiễu nền để mô hình thực tế hơn
- có thể xuất sang `TensorFlow Lite`
- phù hợp để triển khai trên mobile hoặc thiết bị edge

### Các bước xử lý chính

1. Chuẩn bị dữ liệu âm thanh
2. Chia đoạn audio
3. Tạo Mel-spectrogram
4. Huấn luyện mô hình
5. Đánh giá mô hình
6. Chuyển sang TFLite
7. Benchmark tốc độ, dung lượng và độ chính xác

---

## Phần Flutter làm gì?

Flutter là phần giao diện đi kèm để hiển thị kết quả từ AI và dữ liệu cảm biến theo thời gian thực. Phần này giúp người dùng xem trạng thái bé, nhận cảnh báo và cấu hình hệ thống.

Phần frontend là ứng dụng Flutter dành cho phụ huynh hoặc người chăm sóc.

### Chức năng chính

- đăng nhập bằng số điện thoại và OTP
- quét QR để liên kết thiết bị
- xem dashboard thời gian thực
- hiển thị trạng thái bé theo từng chỉ số
- cấu hình ngưỡng cảnh báo
- xem biểu đồ lịch sử
- chạy nền để tiếp tục giám sát khi tắt màn hình

### Các loại cảnh báo

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

### Frontend

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

> Ghi chú: Repository chỉ giữ **1 README tổng quan ở root** để nhà tuyển dụng đọc nhanh, không cần mở thêm file con.

---

## Các điểm nhà tuyển dụng có thể quan tâm

### Kỹ thuật

- xử lý tín hiệu âm thanh
- chuyển audio sang ảnh spectrogram
- transfer learning với MobileNetV2
- chuyển mô hình sang TFLite
- triển khai realtime với Firebase và Flutter

### Kỹ năng hệ thống

- tách rõ backend dữ liệu, AI và frontend
- có luồng xử lý end-to-end
- có phần huấn luyện, đánh giá và triển khai
- có cơ chế cảnh báo realtime và background service

### Ứng dụng thực tế

Dự án mô phỏng đúng bài toán giám sát bé trong môi trường gia đình, nên dễ trình bày khi phỏng vấn vì có giá trị thực tế rõ ràng.

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

---

## Ghi chú

- Phần AI và phần Flutter được tách riêng để dễ bảo trì.
- Dữ liệu huấn luyện và model sinh ra có thể rất lớn, nên thường không đẩy toàn bộ lên GitHub công khai.
- Nếu cần demo, nên ưu tiên file TFLite đã lượng tử hóa vì nhẹ hơn và phù hợp chạy trên thiết bị di động.
