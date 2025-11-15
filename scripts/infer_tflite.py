import numpy as np
import argparse
import librosa
import tensorflow as tf
from PIL import Image # Cần import thư viện Image

# --- Bắt đầu: Hàm xử lý âm thanh ĐÚNG ---
# (Sao chép từ prepare_dataset.py / live_from_wav.py để đảm bảo nhất quán)
def audio_to_spectrogram_image(audio_segment, sr=16000, target_shape=(128, 128)):
    mel_spectrogram = librosa.feature.melspectrogram(y=audio_segment, sr=sr, n_mels=128)
    log_mel_spectrogram = librosa.power_to_db(mel_spectrogram, ref=np.max)
    
    # Chuẩn hóa về 0-255
    img_normalized = (log_mel_spectrogram - log_mel_spectrogram.min()) / (log_mel_spectrogram.max() - log_mel_spectrogram.min())
    img_uint8 = (img_normalized * 255).astype(np.uint8)
    
    # Dùng Pillow để resize
    pil_img = Image.fromarray(img_uint8)
    pil_img = pil_img.resize(target_shape, Image.Resampling.LANCZOS)
    img_resized = np.array(pil_img)
    
    # Dùng Numpy để chuyển thành ảnh 3 kênh
    img_rgb = np.stack([img_resized] * 3, axis=-1)
    
    # Chuẩn hóa ảnh cho mô hình
    img_rgb = img_rgb.astype(np.float32) / 255.0
    
    return img_rgb
# --- Kết thúc: Hàm xử lý âm thanh ĐÚNG ---


def load_tflite_model(path):
    inter = tf.lite.Interpreter(model_path=path)
    inter.allocate_tensors()
    inp_details = inter.get_input_details()
    out_details = inter.get_output_details()
    return inter, inp_details, out_details

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tflite", required=True, help="Path to TFLite model")
    parser.add_argument("--wav", required=True, help="Path to wav file to infer") # Đổi tên cho rõ ràng
    parser.add_argument("--classes", type=str, default="../model/classes.npy", help="Path to classes.npy")
    parser.add_argument("--sr", type=int, default=16000)
    parser.add_argument("--segment_sec", type=float, default=2.0)
    # Thêm tham số overlap để khớp với prepare_dataset.py
    parser.add_argument("--overlap", type=float, default=0.5) 
    args = parser.parse_args()

    inter, inp_det, out_det = load_tflite_model(args.tflite)
    classes = np.load(args.classes, allow_pickle=True)

    y, sr = librosa.load(args.wav, sr=args.sr, mono=True)
    
    seg_len = int(args.segment_sec * args.sr)
    hop_len = int(seg_len * args.overlap) # Tính toán hop_length
    
    idx = 0
    predictions = []
    all_probs = []
    
    target_shape = (128, 128) # Phải khớp với lúc huấn luyện

    print(f"--- Bắt đầu phân tích tệp: {args.wav} ---")
    
    # Tạo các đoạn âm thanh (segments) giống như logic trong prepare_dataset.py
    starts = list(range(0, max(1, len(y) - seg_len + 1), hop_len))
    segments = []
    for s in starts:
        seg = y[s:s+seg_len]
        # Đảm bảo đoạn cuối cùng đủ độ dài (padding nếu cần)
        if len(seg) < seg_len:
             seg = np.pad(seg, (0, seg_len - len(seg)), mode='constant')
        segments.append(seg)
    
    if not segments:
        print("Đoạn ghi âm quá ngắn để phân tích.")
    else:
        for i, seg in enumerate(segments):
            # Sử dụng hàm audio_to_spectrogram_image đã định nghĩa ở trên
            img = audio_to_spectrogram_image(seg, sr, target_shape)
            x = np.expand_dims(img, axis=0).astype(np.float32)

            # Xử lý lượng tử hóa (quantization) nếu mô hình yêu cầu
            if inp_det[0]['dtype'] == np.int8:
                in_scale, in_zero_point = inp_det[0]['quantization']
                x = (x / in_scale + in_zero_point).astype(np.int8)

            inter.set_tensor(inp_det[0]['index'], x)
            inter.invoke()
            out = inter.get_tensor(out_det[0]['index'])

            # Xử lý lượng tử hóa (quantization) cho đầu ra
            if out_det[0]['dtype'] == np.int8:
                o_scale, o_zero = out_det[0]['quantization']
                out = (out.astype(np.float32) - o_zero) * o_scale

            prob = tf.nn.softmax(out[0]).numpy()
            cls = np.argmax(prob)

            print(f"Segment {i+1}/{len(segments)} -> class {classes[cls]} (Prob: {prob[cls]:.2f})")

            predictions.append(cls)
            all_probs.append(prob)

        # Tính toán kết quả cuối cùng
        predictions = np.array(predictions)
        all_probs = np.array(all_probs)
        
        # 1. Bỏ phiếu đa số (Majority Vote)
        final_cls_majority = np.bincount(predictions).argmax()
        
        # 2. Trung bình xác suất (Mean Probabilities)
        mean_probs = all_probs.mean(axis=0)
        final_cls_mean = np.argmax(mean_probs)

        print("\n===== FINAL RESULT =====")
        print(f"Majority vote -> {classes[final_cls_majority]}")
        print(f"Mean probs    -> {classes[final_cls_mean]}")
        print("\nDetailed mean probabilities:")
        for i, class_name in enumerate(classes):
            print(f"  {class_name}: {mean_probs[i]:.4f}")