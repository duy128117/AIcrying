# live_from_wav.py (đã cập nhật, không dùng cv2)
import numpy as np
import argparse
import librosa
import tensorflow as tf
import sounddevice as sd
from scipy.io.wavfile import write
import time
# MỚI: Dùng thư viện Pillow thay cho cv2
from PIL import Image

# --- HÀM TẠO ẢNH SPECTROGRAM (Không dùng cv2) ---
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

def load_tflite_model(path):
    inter = tf.lite.Interpreter(model_path=path)
    inter.allocate_tensors()
    inp_details = inter.get_input_details()
    out_details = inter.get_output_details()
    return inter, inp_details, out_details

# --- Hàm ghi âm ---
def record_audio(filename, duration, fs):
    """Ghi âm từ micro và lưu thành file .wav"""
    print("Chuẩn bị... Bắt đầu nói hoặc tạo ra âm thanh sau 3 giây.")
    time.sleep(1); print("3...")
    time.sleep(1); print("2...")
    time.sleep(1); print("1...")
    
    print(f"*** Đang ghi âm trong {duration} giây... ***")
    recording = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='float32')
    sd.wait()
    
    write(filename, fs, recording)
    print(f"Ghi âm hoàn tất. Đã lưu file '{filename}'")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tflite", required=True, help="Path to TFLite model")
    parser.add_argument("--classes", type=str, default="../model/classes.npy")
    parser.add_argument("--sr", type=int, default=16000)
    parser.add_argument("--duration", type=float, default=5.0)
    parser.add_argument("--segment_sec", type=float, default=2.0)
    args = parser.parse_args()

    temp_wav_file = "temp_recording.wav"
    record_audio(temp_wav_file, args.duration, args.sr)

    inter, inp_det, out_det = load_tflite_model(args.tflite)
    classes = np.load(args.classes, allow_pickle=True)

    y, sr = librosa.load(temp_wav_file, sr=args.sr, mono=True)
    seg_len = int(args.segment_sec * args.sr)
    idx = 0
    predictions = []
    all_probs = []
    
    target_shape = (128, 128)

    print("\n--- Bắt đầu phân tích ---")
    while idx + seg_len <= len(y):
        seg = y[idx: idx+seg_len]
        img = audio_to_spectrogram_image(seg, sr, target_shape)
        x = np.expand_dims(img, axis=0).astype(np.float32)

        if inp_det[0]['dtype'] == np.int8:
            in_scale, in_zero_point = inp_det[0]['quantization']
            x = (x / in_scale + in_zero_point).astype(np.int8)

        inter.set_tensor(inp_det[0]['index'], x)
        inter.invoke()
        out = inter.get_tensor(out_det[0]['index'])

        if out_det[0]['dtype'] == np.int8:
            o_scale, o_zero = out_det[0]['quantization']
            out = (out.astype(np.float32) - o_zero) * o_scale

        prob = tf.nn.softmax(out[0]).numpy()
        cls = np.argmax(prob)
        predictions.append(cls)
        all_probs.append(prob)
        idx += int(seg_len * 0.5)

    if not predictions:
        print("Đoạn ghi âm quá ngắn để phân tích.")
    else:
        predictions = np.array(predictions)
        all_probs = np.array(all_probs)
        final_cls_majority = np.bincount(predictions).argmax()
        mean_probs = all_probs.mean(axis=0)
        final_cls_mean = np.argmax(mean_probs)

        print("\n===== FINAL RESULT =====")
        print(f"Majority vote -> {classes[final_cls_majority]} ({final_cls_majority})")
        print(f"Mean probs    -> {classes[final_cls_mean]} ({final_cls_mean}), probs={mean_probs}")