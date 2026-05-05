# Baby Sleep Tracker

![Flutter Version](https://img.shields.io/badge/Flutter-%3E%3D3.0.0-blue?logo=flutter)
![Dart Version](https://img.shields.io/badge/Dart-%3E%3D2.17.0-blue?logo=dart)
![Firebase](https://img.shields.io/badge/Firebase-Realtime%20Database-orange?logo=firebase)
![License](https://img.shields.io/badge/License-MIT-green)

Ứng dụng Flutter dùng để giám sát giấc ngủ và môi trường của bé theo thời gian thực. Ứng dụng kết nối Firebase để đồng bộ dữ liệu, xác thực người dùng, nhận cảnh báo và theo dõi lịch sử hoạt động của bé một cách trực quan.

---

## Mục tiêu

- Theo dõi dữ liệu bé theo thời gian thực
- Cảnh báo sớm khi có dấu hiệu bất thường
- Kết nối cha mẹ/người chăm sóc qua mã QR và Firebase
- Hỗ trợ chạy nền để không bỏ lỡ cảnh báo quan trọng

---

## Tính năng chính

- **Xác thực bằng OTP** qua số điện thoại
- **Liên kết thiết bị bằng QR code**
- **Dashboard thời gian thực** hiển thị trạng thái của bé
- **Cảnh báo thông minh** cho:
    - bé đang khóc
    - tư thế ngủ bất thường
    - nhiệt độ bé vượt ngưỡng
    - nhiệt độ/độ ẩm môi trường vượt ngưỡng
- **Thông báo đẩy và thông báo cục bộ**
- **Background service** để giám sát liên tục
- **Biểu đồ lịch sử** cho dữ liệu giấc ngủ và môi trường
- **Cài đặt cảnh báo** theo ngưỡng riêng của người dùng

---

## Tech stack

### Core

- Flutter / Dart
- Firebase Authentication
- Firebase Realtime Database
- Firebase Cloud Messaging

### Package chính

- `flutter_background_service`
- `flutter_local_notifications`
- `mobile_scanner`
- `qr_flutter`
- `fl_chart`
- `shared_preferences`
- `connectivity_plus`

---

## Kiến trúc ứng dụng

- `main.dart` xử lý khởi tạo Firebase, notification và background service
- `AuthScreen` quản lý đăng nhập OTP và liên kết thiết bị
- `DashboardScreen` hiển thị dữ liệu realtime, cảnh báo và lịch sử
- `DataService` đọc dữ liệu từ Firebase Realtime Database
- `NotificationService` và `background_service.dart` xử lý thông báo khi app chạy nền

---

## Cấu trúc thư mục

```text
fe/
├── android/
├── ios/
├── lib/
│   ├── models/
│   ├── screens/
│   ├── services/
│   └── widgets/
├── web/
├── pubspec.yaml
└── README.md
```

---

## Yêu cầu cài đặt

- Flutter SDK
- Android Studio / Xcode tùy nền tảng
- Firebase project đã cấu hình
- File cấu hình Firebase:
    - `android/app/google-services.json`
    - `ios/Runner/GoogleService-Info.plist`
    - `lib/firebase_options.dart`

---

## Chạy dự án

```bash
flutter pub get
flutter analyze
flutter run
```

---

## Luồng hoạt động

1. Người dùng đăng nhập bằng số điện thoại
2. Ứng dụng xác thực OTP qua Firebase
3. Người dùng quét QR để liên kết thiết bị
4. Dashboard đọc dữ liệu realtime từ Firebase
5. Nếu có bất thường, app hiển thị cảnh báo và gửi notification

---

## Ghi chú cho nhà tuyển dụng

Dự án tập trung vào 3 điểm chính:

- xử lý realtime
- đồng bộ dữ liệu với Firebase
- trải nghiệm giám sát ổn định trên mobile với background service

