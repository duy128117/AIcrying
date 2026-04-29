# Nhận diện tiếng khóc từ âm thanh bằng Mel-spectrogram

Đây là một dự án AI end-to-end dùng để phân loại âm thanh thành 2 nhóm: **cry** và **not_crying**. Dự án chuyển tín hiệu âm thanh sang ảnh Mel-spectrogram và sử dụng mô hình **MobileNetV2** để huấn luyện, đánh giá và triển khai.

Dự án bao gồm toàn bộ quy trình:

- chuẩn bị dữ liệu âm thanh
- trích xuất đặc trưng dạng spectrogram
- huấn luyện và đánh giá mô hình
- chuyển đổi sang TensorFlow Lite
- benchmark tốc độ, kích thước và độ chính xác
- suy luận realtime từ micro hoặc file WAV

---

## Mục tiêu dự án

Mục tiêu của dự án là xây dựng một hệ thống nhận diện tiếng khóc có tính thực tiễn cao. Thay vì huấn luyện trực tiếp trên sóng âm thô, âm thanh được chuyển thành ảnh Mel-spectrogram 2 chiều và phân loại bằng mạng CNN nhẹ.

Hướng tiếp cận này phù hợp cho:

- thiết bị edge
- triển khai trên mobile
- thử nghiệm AI nhúng
- giám sát âm thanh theo thời gian thực

---

## Tính năng nổi bật

- **Tiền xử lý âm thanh** bằng `librosa`
- **Tạo Mel-spectrogram** từ các đoạn âm thanh 2 giây
- **Tăng cường dữ liệu** cho lớp `cry` bằng nhiễu nền
- **Transfer learning** với `MobileNetV2`
- **Chia tập train/validation/test** với random seed cố định để tái lập kết quả
- **Lưu mô hình Keras** dưới dạng `.keras`
- **Chuyển sang TFLite** và hỗ trợ **INT8 quantization**
- **Benchmark** kích thước, tốc độ suy luận và độ chính xác
- **Suy luận realtime** từ micro hoặc file WAV

---

## Cấu trúc thư mục

```text
ai-git/
├── requirements.txt
├── scripts/
│   ├── prepare_dataset.py
│   ├── train.py
│   ├── evaluate.py
│   ├── convert_to_tflite.py
│   ├── benchmark.py
│   ├── live_from_wav.py
│   └── infer_tflite.py
└── test_audio/
```

> Lưu ý: tùy cách bạn triển khai, thư mục `dataset/` và `model/` có thể được đặt bên ngoài repo GitHub nếu dung lượng quá lớn.

---

## Tổng quan dữ liệu

Dữ liệu được tổ chức theo dạng các thư mục lớp trong `dataset/raw/`.

Cấu trúc điển hình:

```text
dataset/raw/
├── cry/
└── not_crying/
    ├── airplane/
    ├── breathing/
    ├── brushing_teeth/
    ├── can_opening/
    ├── car_horn/
    ├── cat/
    ├── chirping_birds/
    ├── church_bells/
    ├── clapping/
    ├── clock_alarm/
    ├── clock_tick/
    ├── common voice/
    ├── coughing/
    ├── dog/
    ├── fan/
    ├── footsteps/
    ├── helicopter/
    ├── insects/
    ├── keyboard_typing/
    ├── laughing/
    ├── mouse_click/
    ├── rain/
    ├── rooster/
    ├── sneezing/
    ├── snoring/
    ├── thunderstorm/
    ├── train/
    ├── vacuum_cleaner/
    ├── voice-newlive/
    ├── washing_machine/
    └── whisper/
```

Script tiền xử lý cũng hỗ trợ thư mục `background_noises/` để trộn nhiễu vào mẫu tiếng khóc khi tạo đặc trưng.

---

## Quy trình xử lý mô hình

1. Đọc các file `.wav`
2. Chia audio thành các đoạn cố định
3. Chuyển mỗi đoạn thành ảnh Mel-spectrogram
4. Resize về `128 x 128`
5. Nhân 3 kênh để tạo đầu vào `128 x 128 x 3`
6. Huấn luyện bộ phân loại dựa trên `MobileNetV2`
7. Lưu mô hình Keras và danh sách lớp
8. Chuyển sang TFLite để triển khai

---

## Yêu cầu cài đặt

Các thư viện chính:

- Python 3.10+
- TensorFlow 2.20+
- Keras 3+
- librosa
- numpy
- scikit-learn
- matplotlib
- soundfile
- sounddevice
- Pillow

Cài đặt dependencies:

```bash
pip install -r requirements.txt
```

---

## Hướng dẫn chạy

### 1) Chuẩn bị features

Chuyển dữ liệu âm thanh thô thành file đặc trưng nén:

```bash
python scripts/prepare_dataset.py --dataset_dir ../dataset/raw --out ../dataset/features.npz
```

Script này sẽ:

- đọc các file `.wav`
- chia âm thanh thành các cửa sổ 2 giây
- có thể thêm nhiễu nền vào mẫu của lớp `cry`
- chuyển từng đoạn thành ảnh spectrogram đã chuẩn hóa
- lưu `X`, `y` và `classes` vào `features.npz`

---

### 2) Huấn luyện mô hình

```bash
python scripts/train.py --features ../dataset/features.npz --checkpoint_dir ../model
```

Kết quả tạo ra:

- `best_model.keras`
- `final_model.keras`
- `classes.npy`
- `training_history.png`

Trong quá trình huấn luyện, dự án sử dụng:

- backbone `MobileNetV2`
- optimizer Adam
- early stopping
- giảm learning rate khi plateau
- chia validation/test với seed cố định

---

### 3) Đánh giá mô hình đã huấn luyện

```bash
python scripts/evaluate.py --features ../dataset/features.npz --model ../model/best_model.keras
```

Script này tái tạo đúng tập test đã dùng trong huấn luyện và in ra loss, accuracy.

---

### 4) Chuyển sang TFLite

```bash
python scripts/convert_to_tflite.py --keras_model ../model/best_model.keras --out ../model/model.tflite
```

Với INT8 quantization:

```bash
python scripts/convert_to_tflite.py --keras_model ../model/best_model.keras --out ../model/model_quant_int8.tflite --features ../dataset/features.npz --quantize
```

---

### 5) Benchmark Keras và TFLite

```bash
python scripts/benchmark.py --keras_model ../model/best_model.keras --tflite ../model/model_quant_int8.tflite --features ../dataset/features.npz
```

Lệnh này so sánh:

- kích thước mô hình
- tốc độ suy luận
- độ chính xác trên tập test

---

### 6) Suy luận trên file WAV

```bash
python scripts/infer_tflite.py --tflite ../model/model_quant_int8.tflite --wav test_audio/sample.wav --classes ../model/classes.npy
```

Script này sẽ:

- chia file WAV thành các đoạn
- dự đoán từng đoạn
- tổng hợp kết quả bằng majority vote và trung bình xác suất

---

### 7) Suy luận realtime từ micro

```bash
python scripts/live_from_wav.py --tflite ../model/model_quant_int8.tflite --classes ../model/classes.npy
```

Script sẽ ghi âm từ micro, chuyển thành spectrogram và in ra lớp dự đoán cuối cùng.

---

## Thông số kỹ thuật

- **Sample rate đầu vào:** 16 kHz
- **Độ dài mỗi đoạn:** 2 giây
- **Kích thước spectrogram:** `128 x 128`
- **Số kênh:** 3 kênh
- **Mô hình nền:** `MobileNetV2`
- **Hàm loss:** categorical cross-entropy
- **Metric:** accuracy

---

## Vì sao chọn cách tiếp cận này?

Việc dùng ảnh spectrogram giúp mô hình học được đặc trưng thời gian - tần số của âm thanh, rất phù hợp để phân biệt tiếng khóc với nhiều loại âm thanh nền khác nhau.

Ưu điểm:

- dễ huấn luyện hơn so với mô hình trực tiếp trên waveform thô
- dễ trực quan hóa đặc trưng âm thanh
- tương thích tốt với kiến trúc CNN cho ảnh
- thuận lợi khi triển khai lên mobile/edge thông qua TFLite

---

## Điểm nổi bật của dự án

Nếu đưa lên GitHub để ứng tuyển, bạn có thể nhấn mạnh các điểm sau:

- xây dựng pipeline AI âm thanh hoàn chỉnh từ tiền xử lý đến triển khai
- xử lý tăng cường dữ liệu bằng nhiễu nền thực tế
- thiết kế quy trình train và evaluate có thể tái lập
- xuất mô hình sang TFLite để suy luận nhẹ và nhanh
- bổ sung script cho các kịch bản offline, benchmark và realtime

---

## Kết quả

> Bạn nên thay phần này bằng kết quả thật của mình trước khi đăng GitHub.

Mẫu trình bày đề xuất:

| Chỉ số | Keras | TFLite INT8 |
|---|---:|---:|
| Test Accuracy | xx.xx% | xx.xx% |
| Kích thước mô hình | xx MB | xx MB |
| Thời gian suy luận | xx ms | xx ms |

---

## Hướng phát triển tiếp theo

- xây dựng giao diện web hoặc mobile demo
- thử nghiệm thêm các kiến trúc audio khác
- xử lý mất cân bằng dữ liệu tốt hơn
- thêm confusion matrix và classification report
- gom toàn bộ pipeline thành một CLI duy nhất
- triển khai trên Raspberry Pi hoặc Android

---

## Giấy phép

Nếu bạn dự định public dự án trên GitHub, nên thêm một file license phù hợp.

---

## Liên hệ

Nếu dùng cho portfolio, bạn có thể bổ sung:

- họ tên
- email
- LinkedIn
- GitHub profile

---

## Tóm tắt ngắn gọn

Dự án này xây dựng hệ thống nhận diện tiếng khóc từ âm thanh bằng cách chuyển audio sang Mel-spectrogram và phân loại bằng MobileNetV2. Ngoài mô hình Keras, dự án còn hỗ trợ xuất sang TFLite, benchmark, và suy luận realtime từ micro hoặc file WAV.