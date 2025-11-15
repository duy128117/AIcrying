import numpy as np
import tensorflow as tf
import keras
import time
import argparse
import os
import librosa
from PIL import Image
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical

# ===================================================================
# CÁC HÀM TIỆN ÍCH (SAO CHÉP TỪ CÁC TỆP KHÁC ĐỂ ĐẢM BẢO NHẤT QUÁN)
# ===================================================================

def load_features(path):
    """
    Tải dữ liệu đặc trưng đã được xử lý trước từ tệp .npz.
    (Giống như trong train.py)
    """
    data = np.load(path, allow_pickle=True)
    X = data['X']
    y = data['y']
    classes = data['classes']
    return X, y, classes

def audio_to_spectrogram_image(audio_segment, sr=16000, target_shape=(128, 128)):
    """
    Chuyển đổi một đoạn âm thanh thành ảnh phổ mel (128x128x3)
    đã được chuẩn hóa. (Giống như trong prepare_dataset.py đã sửa lỗi)
    """
    mel_spectrogram = librosa.feature.melspectrogram(y=audio_segment, sr=sr, n_mels=128)
    log_mel_spectrogram = librosa.power_to_db(mel_spectrogram, ref=np.max)
    
    spec_min = log_mel_spectrogram.min()
    spec_max = log_mel_spectrogram.max()

    if spec_max == spec_min:
        img_normalized = np.zeros(log_mel_spectrogram.shape, dtype=np.float32)
    else:
        img_normalized = (log_mel_spectrogram - spec_min) / (spec_max - spec_min)
    
    img_uint8 = (img_normalized * 255).astype(np.uint8)
    
    pil_img = Image.fromarray(img_uint8)
    pil_img = pil_img.resize(target_shape, Image.Resampling.LANCZOS)
    img_resized = np.array(pil_img)
    
    img_rgb = np.stack([img_resized] * 3, axis=-1)
    img_rgb = img_rgb.astype(np.float32) / 255.0
    
    return img_rgb

def load_tflite_model(path):
    """
    Tải mô hình TFLite và các chi tiết I/O.
    (Giống như trong infer_tflite.py đã sửa lỗi)
    """
    inter = tf.lite.Interpreter(model_path=path)
    inter.allocate_tensors()
    inp_details = inter.get_input_details()
    out_details = inter.get_output_details()
    return inter, inp_details, out_details

def get_model_size(file_path):
    """Lấy kích thước tệp tính bằng Megabyte (MB)."""
    if os.path.exists(file_path):
        size_bytes = os.path.getsize(file_path)
        return size_bytes / (1024 * 1024)
    return 0

# ===================================================================
# CHỨC NĂNG BENCHMARK CHÍNH
# ===================================================================

def main(args):
    # --- 1. Tải và chuẩn bị dữ liệu ---
    print(f"Đang tải dữ liệu từ: {args.features}")
    X, y, classes = load_features(args.features)
    num_classes = len(classes)
    
    # Tái tạo lại chính xác bộ test/train split như trong train.py
    # Rất quan trọng: sử dụng cùng một 'random_state' (42)
    idx = np.arange(len(y))
    np.random.shuffle(idx) # Mặc dù train_test_split có shuffle, làm vậy cho giống file train
    X = X[idx]; y = y[idx]
    y = y.astype('int32')

    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=42, stratify=y
    )
    # Chúng ta không cần tập validation ở đây, chỉ cần tập test
    y_test_cat = to_categorical(y_test, num_classes)

    print(f"Đã tải {len(X_test)} mẫu test để đánh giá.")
    
    # Lấy một mẫu để đo tốc độ
    sample_input_fp32 = X_test[0:1].astype(np.float32)
    num_runs = 500 # Chạy 500 lần để đo tốc độ trung bình

    # --- 2. Phân tích mô hình Keras (.keras) ---
    print("\n--- Bắt đầu phân tích mô hình Keras ---")
    model = keras.models.load_model(args.keras_model)
    
    # Đo kích thước
    keras_size_mb = get_model_size(args.keras_model)
    print(f"Kích thước tệp Keras: {keras_size_mb:.2f} MB")
    
    # Đo tốc độ
    model.predict(sample_input_fp32, verbose=0) # Khởi động
    
    print(f"Đang đo tốc độ (chạy {num_runs} lần)...")
    start_time_keras = time.perf_counter()
    for _ in range(num_runs):
        model.predict(sample_input_fp32, verbose=0)
    end_time_keras = time.perf_counter()
    keras_time_ms = ((end_time_keras - start_time_keras) / num_runs) * 1000
    
    # Đo độ chính xác
    print("Đang đo độ chính xác trên tập test...")
    keras_loss, keras_acc = model.evaluate(X_test, y_test_cat, verbose=0)
    print("Hoàn thành phân tích Keras.")

    # --- 3. Phân tích mô hình TFLite (.tflite) ---
    print("\n--- Bắt đầu phân tích mô hình TFLite ---")
    interpreter, input_details, output_details = load_tflite_model(args.tflite)

    # Đo kích thước
    tflite_size_mb = get_model_size(args.tflite)
    print(f"Kích thước tệp TFLite: {tflite_size_mb:.2f} MB")

    # Kiểm tra kiểu dữ liệu (Float32 hay INT8)
    input_dtype = input_details[0]['dtype']
    output_dtype = output_details[0]['dtype']
    is_quantized = input_dtype == np.int8
    
    if is_quantized:
        print("Mô hình TFLite đã được lượng tử hóa (INT8).")
        in_scale, in_zero_point = input_details[0]['quantization']
        out_scale, out_zero_point = output_details[0]['quantization']
    else:
        print("Mô hình TFLite là Float32.")

    # Chuẩn bị mẫu cho TFLite
    if is_quantized:
        sample_input_tflite = (sample_input_fp32 / in_scale + in_zero_point).astype(input_dtype)
    else:
        sample_input_tflite = sample_input_fp32

    # Đo tốc độ
    interpreter.set_tensor(input_details[0]['index'], sample_input_tflite)
    interpreter.invoke() # Khởi động
    
    print(f"Đang đo tốc độ (chạy {num_runs} lần)...")
    start_time_tflite = time.perf_counter()
    for _ in range(num_runs):
        interpreter.set_tensor(input_details[0]['index'], sample_input_tflite)
        interpreter.invoke()
        _ = interpreter.get_tensor(output_details[0]['index'])
    end_time_tflite = time.perf_counter()
    tflite_time_ms = ((end_time_tflite - start_time_tflite) / num_runs) * 1000
    
    # Đo độ chính xác
    print("Đang đo độ chính xác trên tập test...")
    tflite_correct_predictions = 0
    
    for i in range(len(X_test)):
        # Lấy mẫu
        input_sample = X_test[i:i+1]
        
        # Lượng tử hóa đầu vào nếu cần
        if is_quantized:
            input_sample = (input_sample / in_scale + in_zero_point).astype(input_dtype)
            
        interpreter.set_tensor(input_details[0]['index'], input_sample)
        interpreter.invoke()
        output_data = interpreter.get_tensor(output_details[0]['index'])
        
        # Giải lượng tử hóa đầu ra nếu cần
        if is_quantized:
            output_data = (output_data.astype(np.float32) - out_zero_point) * out_scale
            
        # Lấy kết quả
        prob = tf.nn.softmax(output_data[0]).numpy()
        pred_class = np.argmax(prob)
        true_class = np.argmax(y_test_cat[i])
        
        if pred_class == true_class:
            tflite_correct_predictions += 1
            
    tflite_acc = tflite_correct_predictions / len(X_test)
    print("Hoàn thành phân tích TFLite.")

    # --- 4. In kết quả tổng hợp ---
    print("\n\n" + "="*30)
    print("     KẾT QUẢ SO SÁNH MÔ HÌNH")
    print("="*30)
    print(f"{'Metric':<20} | {'Keras (.keras)':<20} | {'TFLite (.tflite)':<20}")
    print("-"*64)
    print(f"{'Model Size (MB)':<20} | {keras_size_mb:<20.2f} | {tflite_size_mb:<20.2f}")
    print(f"{'Test Accuracy':<20} | {keras_acc:<20.4f} | {tflite_acc:<20.4f}")
    print(f"{'Inference Time (ms)':<20} | {keras_time_ms:<20.4f} | {tflite_time_ms:<20.4f}")
    print("-"*64)
    
    print("\nPHÂN TÍCH:")
    print(f"* Kích thước: TFLite nhỏ hơn {keras_size_mb/tflite_size_mb:.1f} lần.")
    print(f"* Tốc độ: TFLite nhanh hơn {keras_time_ms/tflite_time_ms:.2f} lần.")
    print(f"* Độ chính xác: Thay đổi { (tflite_acc - keras_acc) * 100 :.2f}%.")
    print("Lưu ý: Thời gian suy luận được đo trên máy tính này. Kết quả có thể khác trên thiết bị di động (ARM).")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="So sánh tốc độ, kích thước và độ chính xác giữa Keras và TFLite.")
    
    parser.add_argument("--keras_model", type=str, default="../model/best_model.keras", 
                        help="Đường dẫn đến tệp .keras đã huấn luyện (best_model.keras)")
    parser.add_argument("--tflite", type=str, default="../model/model_quant_int8.tflite", 
                        help="Đường dẫn đến tệp .tflite đã chuyển đổi")
    parser.add_argument("--features", type=str, default="../dataset/features.npz", 
                        help="Đường dẫn đến tệp features.npz để lấy dữ liệu test")
    
    # Thêm các tham số này để đảm bảo việc chia dữ liệu giống hệt file train.py
    parser.add_argument("--test_size", type=float, default=0.1)
    parser.add_argument("--val_size", type=float, default=0.1)
    
    args = parser.parse_args()
    
    # Kiểm tra tệp
    if not os.path.exists(args.keras_model):
        print(f"Lỗi: Không tìm thấy tệp Keras tại: {args.keras_model}")
    elif not os.path.exists(args.tflite):
        print(f"Lỗi: Không tìm thấy tệp TFLite tại: {args.tflite}")
    elif not os.path.exists(args.features):
        print(f"Lỗi: Không tìm thấy tệp features tại: {args.features}")
    else:
        main(args)