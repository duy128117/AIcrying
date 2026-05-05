# AI – Nhận diện tiếng khóc của bé từ âm thanh

Đây là phần AI của hệ thống **Baby Sleep Tracker**. Mục tiêu của dự án là phân loại âm thanh thành 2 nhóm:

- `cry`
- `not_crying`

Pipeline được thiết kế theo hướng nhẹ, dễ triển khai và phù hợp cho môi trường edge/mobile.

---

## Tổng quan

Luồng xử lý chính của phần AI:

1. Đọc dữ liệu âm thanh `.wav`
2. Chia audio thành các đoạn cố định 2 giây
3. Chuyển từng đoạn sang Mel-spectrogram
4. Resize ảnh về `128 x 128` và nhân 3 kênh để tạo đầu vào cho CNN
5. Huấn luyện mô hình phân loại bằng `MobileNetV2`
6. Đánh giá, benchmark và chuyển đổi sang `TFLite`
7. Suy luận từ file WAV hoặc micro theo thời gian thực

---

## Điểm nổi bật

- Xử lý âm thanh bằng `librosa`
- Tạo Mel-spectrogram và chuẩn hóa dữ liệu đầu vào
- Tăng cường dữ liệu bằng nhiễu nền cho lớp `cry`
- Huấn luyện bằng transfer learning với `MobileNetV2`
- Lưu mô hình Keras dạng `.keras`
- Chuyển sang `TensorFlow Lite`
- Hỗ trợ lượng tử hóa `INT8`
- Có script benchmark tốc độ, dung lượng và độ chính xác

---

## Tech stack

- Python 3.10+
- TensorFlow / Keras
- NumPy
- librosa
- scikit-learn
- Pillow
- matplotlib
- soundfile / sounddevice

---

## Dữ liệu

Dữ liệu được tổ chức theo cấu trúc:

```text
dataset/
├── raw/
│   ├── cry/
│   └── not_crying/
├── background_noises/
└── features.npz
```

Thư mục `not_crying/` gồm nhiều lớp âm thanh môi trường như:

- siren
- car_horn
- breathing
- coughing
- dog
- rain
- thunderstorm
- whisper

---

## Cấu trúc thư mục

```text
ai/
├── evaluate.py
├── requirements.txt
├── dataset/
├── model/
├── scripts/
│   ├── prepare_dataset.py
│   ├── train.py
│   ├── convert_to_tflite.py
│   ├── benchmark.py
│   └── live_from_wav.py
└── test_audio/
```

---

## Các script chính

### 1. Chuẩn bị dữ liệu

Script: `scripts/prepare_dataset.py`

Chức năng:

- đọc file `.wav`
- chia đoạn theo thời lượng
- tạo Mel-spectrogram
- lưu ra file `features.npz`

### 2. Huấn luyện

Script: `scripts/train.py`

Chức năng:

- train mô hình `MobileNetV2`
- dùng early stopping và reduce learning rate
- lưu `best_model.keras`, `final_model.keras`, `classes.npy`

### 3. Đánh giá

Script: `evaluate.py`

Chức năng:

- tái tạo lại tập test
- load mô hình đã train
- in ra loss và accuracy

### 4. Chuyển sang TFLite

Script: `scripts/convert_to_tflite.py`

Chức năng:

- export mô hình sang `.tflite`
- hỗ trợ quantization `INT8`

### 5. Benchmark

Script: `scripts/benchmark.py`

Chức năng:

- so sánh Keras và TFLite
- đo kích thước model
- đo tốc độ suy luận
- đo độ chính xác trên tập test

### 6. Suy luận realtime

Script: `scripts/live_from_wav.py`

Chức năng:

- ghi âm từ micro
- chia audio thành các cửa sổ 2 giây
- dự đoán nhãn cuối cùng bằng majority vote và trung bình xác suất

---

## Mô hình

- Backbone: `MobileNetV2`
- Input: `128 x 128 x 3`
- Loss: categorical cross-entropy
- Metric: accuracy

---

## Cài đặt

```bash
pip install -r requirements.txt
```

---

## Cách chạy nhanh

### Tạo features

```bash
python scripts/prepare_dataset.py --dataset_dir ../dataset/raw --out ../dataset/features.npz
```

### Huấn luyện

```bash
python scripts/train.py --features ../dataset/features.npz --checkpoint_dir ../model
```

### Đánh giá

```bash
python evaluate.py --features ../dataset/features.npz --model ../model/best_model.keras
```

### Chuyển sang TFLite

```bash
python scripts/convert_to_tflite.py --keras_model ../model/best_model.keras --out ../model/model.tflite
```

### Benchmark

```bash
python scripts/benchmark.py --keras_model ../model/best_model.keras --tflite ../model/model_quant_int8.tflite --features ../dataset/features.npz
```

---

## Ghi chú triển khai

- Dữ liệu audio và model có thể rất nặng, nên thường tách riêng khỏi source code khi đẩy lên GitHub.
- Nếu cần demo thực tế, nên ưu tiên `model_quant_int8.tflite` vì nhẹ và phù hợp cho thiết bị di động.
